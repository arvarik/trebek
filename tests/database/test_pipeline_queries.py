"""
Tests for PipelineQueryMixin — polling, retry logic, telemetry upserts,
and the SQL injection prevention whitelist.
"""

import sqlite3
import pytest
from pathlib import Path

from trebek.database.writer import DatabaseWriter
from trebek.status import PipelineStatus


@pytest.fixture
async def db_writer_with_episodes(tmp_path: Path) -> DatabaseWriter:
    """Creates a DatabaseWriter with schema and some test episodes."""
    db_path = str(tmp_path / "test.db")
    schema_path = Path(__file__).resolve().parents[2] / "trebek" / "schema.sql"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        # Insert test episodes
        conn.execute(
            "INSERT INTO pipeline_state (episode_id, status, source_filename) VALUES (?, ?, ?)",
            ("ep001", "PENDING", "test1.mp4"),
        )
        conn.execute(
            "INSERT INTO pipeline_state (episode_id, status, source_filename) VALUES (?, ?, ?)",
            ("ep002", "PENDING", "test2.mp4"),
        )
        conn.execute(
            "INSERT INTO pipeline_state (episode_id, status, source_filename) VALUES (?, ?, ?)",
            ("ep003", "TRANSCRIPT_READY", "test3.mp4"),
        )
        conn.execute(
            "INSERT INTO pipeline_state (episode_id, status, source_filename, retry_count) VALUES (?, ?, ?, ?)",
            ("ep_failed", "FAILED", "fail.mp4", 3),
        )
        conn.commit()

    writer = DatabaseWriter(db_path)
    await writer.start()
    yield writer
    await writer.stop()


class TestPollForWork:
    """Atomic polling query tests."""

    async def test_poll_returns_oldest_episode(self, db_writer_with_episodes: DatabaseWriter) -> None:
        result = await db_writer_with_episodes.poll_for_work("PENDING", "TRANSCRIBING")
        assert result == "ep001"

        # Verify the status was updated
        rows = await db_writer_with_episodes.execute(
            "SELECT status FROM pipeline_state WHERE episode_id = ?", ("ep001",)
        )
        assert rows == [("TRANSCRIBING",)]

    async def test_poll_returns_none_when_no_work(self, db_writer_with_episodes: DatabaseWriter) -> None:
        result = await db_writer_with_episodes.poll_for_work("COMPLETED", "PROCESSING")
        assert result is None

    async def test_poll_is_atomic(self, db_writer_with_episodes: DatabaseWriter) -> None:
        """Two concurrent polls should each get a different episode."""
        r1 = await db_writer_with_episodes.poll_for_work("PENDING", "TRANSCRIBING")
        r2 = await db_writer_with_episodes.poll_for_work("PENDING", "TRANSCRIBING")
        assert r1 is not None
        assert r2 is not None
        assert r1 != r2

    async def test_poll_with_enum_values(self, db_writer_with_episodes: DatabaseWriter) -> None:
        """PipelineStatus enum values should work as poll arguments."""
        result = await db_writer_with_episodes.poll_for_work(PipelineStatus.TRANSCRIPT_READY, PipelineStatus.CLEANED)
        assert result == "ep003"


class TestFailEpisodeWithRetry:
    """Retry logic and permanent failure tests."""

    async def test_retry_increments_counter(self, db_writer_with_episodes: DatabaseWriter) -> None:
        is_permanent = await db_writer_with_episodes.fail_episode_with_retry(
            "ep001", "PENDING", "test error", max_retries=3
        )
        assert is_permanent is False

        rows = await db_writer_with_episodes.execute(
            "SELECT status, retry_count, last_error FROM pipeline_state WHERE episode_id = ?",
            ("ep001",),
        )
        assert rows[0][0] == "PENDING"  # Reset to previous status
        assert rows[0][1] == 1  # Retry count incremented
        assert "test error" in rows[0][2]

    async def test_permanent_fail_when_retries_exhausted(self, db_writer_with_episodes: DatabaseWriter) -> None:
        # ep_failed already has retry_count=3
        is_permanent = await db_writer_with_episodes.fail_episode_with_retry(
            "ep_failed", "PENDING", "fatal error", max_retries=3
        )
        assert is_permanent is True

        rows = await db_writer_with_episodes.execute(
            "SELECT status FROM pipeline_state WHERE episode_id = ?",
            ("ep_failed",),
        )
        assert rows[0][0] == PipelineStatus.FAILED

    async def test_error_message_truncated_to_500(self, db_writer_with_episodes: DatabaseWriter) -> None:
        long_error = "x" * 1000
        await db_writer_with_episodes.fail_episode_with_retry("ep001", "PENDING", long_error, max_retries=3)
        rows = await db_writer_with_episodes.execute(
            "SELECT last_error FROM pipeline_state WHERE episode_id = ?", ("ep001",)
        )
        assert len(rows[0][0]) == 500


class TestResetFailedEpisodes:
    """Reset all FAILED episodes."""

    async def test_reset_returns_count(self, db_writer_with_episodes: DatabaseWriter) -> None:
        count = await db_writer_with_episodes.reset_failed_episodes()
        assert count == 1  # ep_failed

    async def test_reset_clears_retry_count_and_error(self, db_writer_with_episodes: DatabaseWriter) -> None:
        await db_writer_with_episodes.reset_failed_episodes()
        rows = await db_writer_with_episodes.execute(
            "SELECT status, retry_count, last_error FROM pipeline_state WHERE episode_id = ?",
            ("ep_failed",),
        )
        assert rows[0][0] == PipelineStatus.PENDING
        assert rows[0][1] == 0
        assert rows[0][2] is None

    async def test_reset_returns_zero_when_none_failed(self, db_writer_with_episodes: DatabaseWriter) -> None:
        await db_writer_with_episodes.reset_failed_episodes()  # Reset first
        count = await db_writer_with_episodes.reset_failed_episodes()  # Now none are FAILED
        assert count == 0


class TestUpdateJobTelemetry:
    """Telemetry upsert and SQL injection prevention."""

    async def test_valid_columns_accepted(self, db_writer_with_episodes: DatabaseWriter) -> None:
        await db_writer_with_episodes.update_job_telemetry(
            "ep001",
            peak_vram_mb=8192.5,
            gemini_total_cost_usd=0.0042,
            pydantic_retry_count=2,
        )
        rows = await db_writer_with_episodes.execute(
            "SELECT peak_vram_mb, gemini_total_cost_usd, pydantic_retry_count FROM job_telemetry WHERE episode_id = ?",
            ("ep001",),
        )
        assert rows[0][0] == 8192.5
        assert abs(rows[0][1] - 0.0042) < 1e-6
        assert rows[0][2] == 2

    async def test_invalid_column_raises_valueerror(self, db_writer_with_episodes: DatabaseWriter) -> None:
        with pytest.raises(ValueError, match="Invalid telemetry column"):
            await db_writer_with_episodes.update_job_telemetry(
                "ep001",
                evil_column="DROP TABLE pipeline_state",
            )

    async def test_sql_injection_blocked(self, db_writer_with_episodes: DatabaseWriter) -> None:
        """Crafted column names should be rejected by the whitelist."""
        with pytest.raises(ValueError, match="Invalid telemetry column"):
            await db_writer_with_episodes.update_job_telemetry(
                "ep001",
                **{"episode_id; DROP TABLE job_telemetry--": 1},
            )

    async def test_empty_kwargs_is_noop(self, db_writer_with_episodes: DatabaseWriter) -> None:
        # Should not raise or create a telemetry row
        await db_writer_with_episodes.update_job_telemetry("ep_nonexistent")
        rows = await db_writer_with_episodes.execute(
            "SELECT COUNT(*) FROM job_telemetry WHERE episode_id = ?", ("ep_nonexistent",)
        )
        assert rows == [(0,)]

    async def test_upsert_creates_row_if_missing(self, db_writer_with_episodes: DatabaseWriter) -> None:
        """First call should INSERT, second should UPDATE the same row."""
        await db_writer_with_episodes.update_job_telemetry("ep001", peak_vram_mb=100.0)
        await db_writer_with_episodes.update_job_telemetry("ep001", peak_vram_mb=200.0)
        rows = await db_writer_with_episodes.execute(
            "SELECT peak_vram_mb FROM job_telemetry WHERE episode_id = ?", ("ep001",)
        )
        assert len(rows) == 1
        assert rows[0][0] == 200.0


class TestInsertJobTelemetry:
    """Full telemetry record insertion."""

    async def test_insert_full_telemetry_record(self, db_writer_with_episodes: DatabaseWriter) -> None:
        from trebek.schemas import JobTelemetry

        telemetry = JobTelemetry(
            episode_id="ep001",
            peak_vram_mb=8000.0,
            avg_gpu_utilization_pct=85.0,
            stage_ingestion_ms=100.0,
            stage_gpu_extraction_ms=5000.0,
            stage_commercial_filtering_ms=200.0,
            stage_structured_extraction_ms=3000.0,
            stage_multimodal_ms=1500.0,
            stage_vectorization_ms=500.0,
            gemini_total_input_tokens=50000,
            gemini_total_output_tokens=10000,
            gemini_total_cached_tokens=5000,
            gemini_total_cost_usd=0.15,
            gemini_api_latency_ms=2500.0,
            pydantic_retry_count=1,
        )
        await db_writer_with_episodes.insert_job_telemetry(telemetry)
        rows = await db_writer_with_episodes.execute(
            "SELECT peak_vram_mb, gemini_total_cost_usd FROM job_telemetry WHERE episode_id = ?",
            ("ep001",),
        )
        assert rows[0][0] == 8000.0
        assert abs(rows[0][1] - 0.15) < 1e-6
