"""
Comprehensive tests for Phase C: UX, live dashboard, and operator tools.

Covers:
- trebek status (queue health, in-flight, errors, --json, --watch)
- trebek inspect <ep_id> (metadata, contestants, clues, telemetry, quality warnings)
- trebek retry [ep_id] (all vs targeted vs forced)
- trebek clean (orphans, tmp, slices, obsolete transcripts, dry-run vs apply)
- trebek export <ep_id> (json, csv, markdown)
- PipelineProgressCoordinator & SessionTelemetry ticker
- Hardware detection and pre-flight diagnostics
"""

import json
import os
import sqlite3
import pytest
from pathlib import Path

from trebek.status import PipelineStatus
from trebek.database.writer import DatabaseWriter
from trebek.analysis.status import get_queue_status
from trebek.ui.status import render_queue_status, run_status_display
from trebek.analysis.inspect import inspect_episode
from trebek.ui.inspect import render_episode_inspection, handle_inspect_command
from trebek.pipeline.cleanup import scan_cleanup_candidates, execute_cleanup
from trebek.ui.cleanup import handle_clean_command
from trebek.analysis.export import (
    export_episode,
    export_episode_json,
    export_episode_csv,
    export_episode_markdown,
)
from trebek.ui.progress import (
    SessionTelemetry,
    PipelineProgressCoordinator,
    create_pipeline_progress,
)
from trebek.llm.client import (
    register_gemini_usage_callback,
    unregister_gemini_usage_callback,
)
from trebek.gpu.hardware import detect_hardware, HardwareInfo
from trebek.cli import build_parser


@pytest.fixture
def populated_db(tmp_path: Path) -> tuple[str, str]:
    """Creates a fully populated SQLite test database with relational and telemetry data."""
    db_path = str(tmp_path / "trebek_test.db")
    output_dir = str(tmp_path / "output")
    os.makedirs(output_dir, exist_ok=True)

    schema_path = Path(__file__).resolve().parents[2] / "trebek" / "schema.sql"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

        # 1. Pipeline state
        conn.execute(
            """
            INSERT INTO pipeline_state (episode_id, status, source_filename, transcript_path, retry_count, last_error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ep_done", PipelineStatus.COMPLETED, "/videos/ep_done.mp4", str(tmp_path / "ep_done.json.gz"), 0, None),
        )
        conn.execute(
            """
            INSERT INTO pipeline_state (episode_id, status, source_filename, transcript_path, retry_count, last_error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ep_failed", PipelineStatus.FAILED, "/videos/ep_failed.mp4", None, 3, "WhisperX OOM on frame 240"),
        )
        conn.execute(
            """
            INSERT INTO pipeline_state (episode_id, status, source_filename, transcript_path, retry_count, last_error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ep_transcribing", PipelineStatus.TRANSCRIBING, "/videos/ep_tx.mp4", None, 1, None),
        )
        conn.execute(
            """
            INSERT INTO pipeline_state (episode_id, status, source_filename, transcript_path, retry_count, last_error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ep_pending", PipelineStatus.PENDING, "/videos/ep_pending.mp4", None, 0, None),
        )

        # 2. Episodes table
        conn.execute(
            """
            INSERT INTO episodes (episode_id, air_date, host_name, is_tournament)
            VALUES (?, ?, ?, ?)
            """,
            ("ep_done", "2024-03-15", "Ken Jennings", 0),
        )

        # 3. Contestants
        conn.execute(
            "INSERT INTO contestants (contestant_id, name, occupational_category, is_returning_champion) VALUES (?, ?, ?, ?)",
            ("c1", "Alice", "Software Engineer", 1),
        )
        conn.execute(
            "INSERT INTO contestants (contestant_id, name, occupational_category, is_returning_champion) VALUES (?, ?, ?, ?)",
            ("c2", "Bob", "Librarian", 0),
        )
        conn.execute(
            "INSERT INTO contestants (contestant_id, name, occupational_category, is_returning_champion) VALUES (?, ?, ?, ?)",
            ("c3", "Carol", "Teacher", 0),
        )

        # 4. Episode performances
        conn.execute(
            "INSERT INTO episode_performances (episode_id, contestant_id, podium_position, coryat_score, final_score) VALUES (?, ?, ?, ?, ?)",
            ("ep_done", "c1", 1, 18400, 24000),
        )
        conn.execute(
            "INSERT INTO episode_performances (episode_id, contestant_id, podium_position, coryat_score, final_score) VALUES (?, ?, ?, ?, ?)",
            ("ep_done", "c2", 2, 12000, 14500),
        )
        conn.execute(
            "INSERT INTO episode_performances (episode_id, contestant_id, podium_position, coryat_score, final_score) VALUES (?, ?, ?, ?, ?)",
            ("ep_done", "c3", 3, 6000, 0),
        )

        # 5. Clues
        conn.execute(
            """
            INSERT INTO clues (clue_id, episode_id, round, category, board_row, board_col, selection_order, clue_text, correct_response, is_verified, is_daily_double, is_triple_stumper)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("clue1", "ep_done", "J!", "SCIENCE", 1, 1, 1, "This element has the symbol Au", "Gold", 1, 0, 0),
        )
        conn.execute(
            """
            INSERT INTO clues (clue_id, episode_id, round, category, board_row, board_col, selection_order, clue_text, correct_response, is_verified, is_daily_double, is_triple_stumper)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("clue2", "ep_done", "J!", "SCIENCE", 2, 1, 2, "Daily double element", "Silver", 1, 1, 0),
        )
        conn.execute(
            """
            INSERT INTO clues (clue_id, episode_id, round, category, board_row, board_col, selection_order, clue_text, correct_response, is_verified, is_daily_double, is_triple_stumper)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "clue_fj",
                "ep_done",
                "Final J!",
                "WORLD CAPITALS",
                None,
                None,
                61,
                "This capital is located on the Seine",
                "Paris",
                1,
                0,
                0,
            ),
        )

        # 6. Buzz attempts & wagers
        conn.execute(
            "INSERT INTO buzz_attempts (attempt_id, clue_id, contestant_id, attempt_order, buzz_timestamp_ms, is_lockout_inferred, response_given, is_correct) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("b1", "clue1", "c1", 1, 1200.0, 0, "Gold", 1),
        )
        conn.execute(
            "INSERT INTO wagers (wager_id, clue_id, contestant_id, actual_wager) VALUES (?, ?, ?, ?)",
            ("w1", "clue2", "c1", 2000),
        )

        # 7. Job telemetry
        conn.execute(
            """
            INSERT INTO job_telemetry (
                episode_id, peak_vram_mb, avg_gpu_utilization_pct,
                stage_ingestion_ms, stage_gpu_extraction_ms, stage_structured_extraction_ms,
                gemini_total_input_tokens, gemini_total_output_tokens, gemini_total_cached_tokens,
                gemini_total_cost_usd, gemini_api_latency_ms, pydantic_retry_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("ep_done", 5800.0, 78.5, 120.0, 4500.0, 8900.0, 15000, 2500, 8000, 0.045, 1250.0, 0),
        )
        conn.commit()

    return db_path, output_dir


# ═════════════════════════════════════════════════════════════════════════════
#  1. Queue Status Tests (`trebek status`)
# ═════════════════════════════════════════════════════════════════════════════


class TestQueueStatus:
    """Tests for get_queue_status and rendering."""

    def test_get_queue_status_missing_db(self, tmp_path: Path) -> None:
        result = get_queue_status(str(tmp_path / "nonexistent.db"))
        assert result["database_found"] is False
        assert result["total"] == 0

    def test_get_queue_status_counts_and_in_flight(self, populated_db: tuple[str, str]) -> None:
        db_path, _ = populated_db
        status = get_queue_status(db_path)

        assert status["database_found"] is True
        assert status["total"] == 4
        assert status["status_counts"]["COMPLETED"] == 1
        assert status["status_counts"]["FAILED"] == 1
        assert status["status_counts"]["TRANSCRIBING"] == 1
        assert status["status_counts"]["PENDING"] == 1

        # In flight check
        assert len(status["in_flight"]) == 1
        assert status["in_flight"][0]["episode_id"] == "ep_transcribing"

        # Recent errors check
        assert len(status["recent_errors"]) == 1
        assert status["recent_errors"][0]["episode_id"] == "ep_failed"
        assert "WhisperX OOM" in status["recent_errors"][0]["last_error"]

    def test_render_queue_status_renders_group(self, populated_db: tuple[str, str]) -> None:
        db_path, _ = populated_db
        status = get_queue_status(db_path)
        group = render_queue_status(status)
        assert group is not None

    @pytest.mark.asyncio
    async def test_run_status_display_json(
        self, populated_db: tuple[str, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        db_path, _ = populated_db
        await run_status_display(db_path, as_json=True)
        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert parsed["total"] == 4
        assert "status_counts" in parsed


# ═════════════════════════════════════════════════════════════════════════════
#  2. Episode Inspection Tests (`trebek inspect <ep_id>`)
# ═════════════════════════════════════════════════════════════════════════════


class TestEpisodeInspection:
    """Tests for inspect_episode and Rich inspection UI."""

    def test_inspect_nonexistent_episode(self, populated_db: tuple[str, str]) -> None:
        db_path, output_dir = populated_db
        result = inspect_episode(db_path, "nonexistent", output_dir)
        assert result is None

    def test_inspect_completed_episode(self, populated_db: tuple[str, str]) -> None:
        db_path, output_dir = populated_db
        data = inspect_episode(db_path, "ep_done", output_dir)
        assert data is not None
        assert data["episode_id"] == "ep_done"
        assert data["status"] == "COMPLETED"
        assert data["air_date"] == "2024-03-15"
        assert data["host_name"] == "Ken Jennings"

        # Contestants
        assert len(data["contestants"]) == 3
        assert data["contestants"][0]["name"] == "Alice"
        assert data["contestants"][0]["coryat_score"] == 18400

        # Clues
        assert data["clues_summary"]["total_clues"] == 3
        assert data["clues_summary"]["daily_doubles"] == 1

        # Telemetry
        assert data["telemetry"] is not None
        assert data["telemetry"]["tokens"]["total"] == 25500
        assert data["telemetry"]["cost_usd"] == 0.045

    def test_inspect_failed_episode(self, populated_db: tuple[str, str]) -> None:
        db_path, output_dir = populated_db
        data = inspect_episode(db_path, "ep_failed", output_dir)
        assert data is not None
        assert data["status"] == "FAILED"
        assert "WhisperX OOM" in data["last_error"]

    def test_render_inspection_group(self, populated_db: tuple[str, str]) -> None:
        db_path, output_dir = populated_db
        data = inspect_episode(db_path, "ep_done", output_dir)
        assert data is not None
        group = render_episode_inspection(data)
        assert group is not None

    def test_handle_inspect_command_json(
        self, populated_db: tuple[str, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        db_path, output_dir = populated_db
        handle_inspect_command(db_path, "ep_done", output_dir, as_json=True)
        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert parsed["episode_id"] == "ep_done"


# ═════════════════════════════════════════════════════════════════════════════
#  3. Targeted and Forced Retry Tests (`trebek retry [ep_id] [--force]`)
# ═════════════════════════════════════════════════════════════════════════════


class TestRetryCommand:
    """Tests for database reset_episode with targeted and forced resets."""

    @pytest.mark.asyncio
    async def test_reset_all_failed_episodes(self, populated_db: tuple[str, str]) -> None:
        db_path, _ = populated_db
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            count = await writer.reset_episode()
            assert count == 1

            rows = await writer.execute(
                "SELECT status, retry_count, last_error FROM pipeline_state WHERE episode_id = 'ep_failed'"
            )
            assert rows[0][0] == PipelineStatus.PENDING
            assert rows[0][1] == 0
            assert rows[0][2] is None
        finally:
            await writer.stop()

    @pytest.mark.asyncio
    async def test_reset_targeted_episode(self, populated_db: tuple[str, str]) -> None:
        db_path, _ = populated_db
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            # Targeting ep_failed should succeed
            count = await writer.reset_episode("ep_failed")
            assert count == 1

            # Targeting ep_done without force should fail (status is COMPLETED, not FAILED)
            count_unforced = await writer.reset_episode("ep_done", force=False)
            assert count_unforced == 0

            # Targeting ep_done WITH force should succeed
            count_forced = await writer.reset_episode("ep_done", force=True)
            assert count_forced == 1

            rows = await writer.execute("SELECT status FROM pipeline_state WHERE episode_id = 'ep_done'")
            assert rows[0][0] == PipelineStatus.PENDING
        finally:
            await writer.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  4. File Cleanup Tests (`trebek clean [--apply]`)
# ═════════════════════════════════════════════════════════════════════════════


class TestCleanupTool:
    """Tests for safe cleanup scanning and execution."""

    def test_scan_cleanup_identifies_candidates(self, tmp_path: Path, populated_db: tuple[str, str]) -> None:
        db_path, output_dir = populated_db

        # Create candidates
        wav_file = Path(output_dir) / "chunk_123.wav"
        wav_file.write_bytes(b"dummy wav content")

        tmp_file = Path(output_dir) / "model.bin.tmp"
        tmp_file.write_bytes(b"interrupted write")

        slice_file = Path(output_dir) / "ep_done_interview_slice.mp3"
        slice_file.write_bytes(b"slice mp3")

        # Obsolete transcript (for completed ep_done)
        obs_tx = Path(output_dir) / "obsolete.json.gz"
        obs_tx.write_bytes(b"gzip data")

        candidates = scan_cleanup_candidates(output_dir, db_path)
        filenames = {c.filename for c in candidates}

        assert "chunk_123.wav" in filenames
        assert "model.bin.tmp" in filenames
        assert "ep_done_interview_slice.mp3" in filenames
        assert "obsolete.json.gz" in filenames

    def test_execute_cleanup_deletes_files(self, tmp_path: Path, populated_db: tuple[str, str]) -> None:
        db_path, output_dir = populated_db

        wav_file = Path(output_dir) / "orphaned.wav"
        wav_file.write_bytes(b"12345")

        candidates = scan_cleanup_candidates(output_dir, db_path)
        count, freed = execute_cleanup(candidates)

        assert count >= 1
        assert freed >= 5
        assert not wav_file.exists()

    def test_handle_clean_command_dry_run_vs_apply(self, populated_db: tuple[str, str]) -> None:
        db_path, output_dir = populated_db
        test_file = Path(output_dir) / "temp.wav"
        test_file.write_bytes(b"audio data")

        # Dry run does not delete
        handle_clean_command(output_dir, db_path, apply=False)
        assert test_file.exists()

        # Apply deletes
        handle_clean_command(output_dir, db_path, apply=True)
        assert not test_file.exists()


# ═════════════════════════════════════════════════════════════════════════════
#  5. Export Tests (`trebek export <ep_id>`)
# ═════════════════════════════════════════════════════════════════════════════


class TestEpisodeExport:
    """Tests for json, csv, and markdown export serialization."""

    def test_export_json(self, populated_db: tuple[str, str]) -> None:
        db_path, output_dir = populated_db
        json_str = export_episode_json(db_path, "ep_done", output_dir)
        data = json.loads(json_str)

        assert data["episode_id"] == "ep_done"
        assert len(data["contestants"]) == 3
        assert len(data["clues"]) == 3

    def test_export_csv(self, populated_db: tuple[str, str]) -> None:
        db_path, _ = populated_db
        csv_str = export_episode_csv(db_path, "ep_done")

        assert "episode_id,round,category" in csv_str
        assert "Gold" in csv_str
        assert "SCIENCE" in csv_str

    def test_export_markdown(self, populated_db: tuple[str, str]) -> None:
        db_path, _ = populated_db
        md_str = export_episode_markdown(db_path, "ep_done")

        assert "# Jeopardy! Episode ep_done" in md_str
        assert "Ken Jennings" in md_str
        assert "## Contestants & Coryat Scores" in md_str
        assert "Alice" in md_str
        assert "## J! Round" in md_str
        assert "## Final J! Round" in md_str

    def test_export_to_file(self, tmp_path: Path, populated_db: tuple[str, str]) -> None:
        db_path, _ = populated_db
        target_file = str(tmp_path / "export" / "episode.md")
        content = export_episode(db_path, "ep_done", format="md", output_path=target_file)

        assert os.path.exists(target_file)
        assert Path(target_file).read_text(encoding="utf-8") == content


# ═════════════════════════════════════════════════════════════════════════════
#  6. Progress Coordinator & Telemetry Callback Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestProgressAndTelemetry:
    """Tests for multi-task progress coordinator and session cost ticker."""

    def test_session_telemetry_accumulation_and_ticker(self) -> None:
        tel = SessionTelemetry()
        tel.record_usage(
            {
                "cost_usd": 0.0015,
                "input_tokens": 1200,
                "output_tokens": 400,
                "cached_tokens": 800,
            }
        )

        assert tel.total_cost_usd == 0.0015
        assert tel.total_input_tokens == 1200
        assert tel.total_output_tokens == 400
        assert tel.total_cached_tokens == 800
        assert tel.total_calls == 1

        rendered = tel.render_ticker()
        assert rendered is not None

    def test_gemini_callback_lifecycle(self) -> None:
        records = []

        def my_callback(u: dict[str, float]) -> None:
            records.append(u)

        register_gemini_usage_callback(my_callback)
        from trebek.llm.client import _notify_gemini_usage

        _notify_gemini_usage({"cost_usd": 0.05})

        assert len(records) == 1
        assert records[0]["cost_usd"] == 0.05

        unregister_gemini_usage_callback(my_callback)
        _notify_gemini_usage({"cost_usd": 0.02})
        # Record count should still be 1 after unregistering
        assert len(records) == 1

    def test_pipeline_progress_coordinator(self) -> None:
        tel = SessionTelemetry()
        prog = create_pipeline_progress(session_telemetry=tel)
        coord = PipelineProgressCoordinator(prog, tel)

        coord.initialize_slots(
            active_stages={"transcribe", "extract", "augment", "verify"}, llm_concurrency=2, total_episodes=5
        )
        assert coord.slots is not None
        assert coord.slots.gpu is not None
        assert len(coord.slots.llm) == 2
        assert coord.slots.augment is not None
        assert coord.slots.commit is not None
        assert coord.slots.verified is not None

        # Updates
        coord.update_gpu("ep123")
        coord.update_llm(1, "ep123", "pass 1")
        coord.update_augment("ep123")
        coord.update_commit("ep123")
        coord.advance_verified()
        coord.update_total_episodes(10)

        # Reset to idle
        coord.update_gpu(None)
        coord.update_llm(1, None)
        coord.update_augment(None)
        coord.update_commit(None)


# ═════════════════════════════════════════════════════════════════════════════
#  7. Hardware & Preflight Diagnostics Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestHardwareAndDiagnostics:
    """Tests for hardware acceleration detection and CLI preflight checks."""

    def test_detect_hardware_returns_valid_object(self) -> None:
        hw = detect_hardware()
        assert isinstance(hw, HardwareInfo)
        assert hw.device in ("cuda", "mps", "cpu")
        assert hw.whisper_device in ("cuda", "cpu")
        assert hw.recommended_compute in ("float16", "int8")

    def test_cli_parser_recognizes_new_commands(self) -> None:
        parser = build_parser()

        # status
        args_status = parser.parse_args(["status", "--watch", "--json"])
        assert args_status.command == "status"
        assert args_status.watch is True
        assert args_status.json is True

        # inspect
        args_inspect = parser.parse_args(["inspect", "ep_test", "--json"])
        assert args_inspect.command == "inspect"
        assert args_inspect.episode_id == "ep_test"
        assert args_inspect.json is True

        # retry
        args_retry = parser.parse_args(["retry", "ep_test", "--force"])
        assert args_retry.command == "retry"
        assert args_retry.episode_id == "ep_test"
        assert args_retry.force is True

        # clean
        args_clean = parser.parse_args(["clean", "--apply"])
        assert args_clean.command == "clean"
        assert args_clean.apply is True

        # export
        args_export = parser.parse_args(["export", "ep_test", "-f", "csv", "-o", "out.csv"])
        assert args_export.command == "export"
        assert args_export.episode_id == "ep_test"
        assert args_export.format == "csv"
        assert args_export.output == "out.csv"

        # run --allow-cpu
        args_run = parser.parse_args(["run", "--allow-cpu"])
        assert args_run.allow_cpu is True
