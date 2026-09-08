"""
Tests for pipeline ingestion worker — scanning, database batching, query deduplication,
in-progress file stability, and worker lifecycle.
"""

import asyncio
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from trebek.pipeline.orchestrator import TrebekPipelineOrchestrator
from trebek.pipeline.workers.ingestion import run_ingestion_pass, ingestion_worker
from trebek.database.writer import DatabaseWriter
from trebek.status import PipelineStatus


@pytest.fixture
def test_db_path(tmp_path: Path) -> str:
    """Creates a temporary SQLite database initialized with schema.sql."""
    db_path = str(tmp_path / "trebek_test.db")
    schema_path = Path(__file__).resolve().parents[2] / "trebek" / "schema.sql"
    with sqlite3.connect(db_path) as conn:
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
    return db_path


@pytest.fixture
def video_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory with test video files."""
    vdir = tmp_path / "videos"
    vdir.mkdir()
    (vdir / "ep1.mp4").write_text("dummy_content_1")
    (vdir / "ep2.mkv").write_text("dummy_content_2")
    return vdir


class TestRunIngestionPass:
    """Tests for run_ingestion_pass logic."""

    @pytest.mark.asyncio
    async def test_ingests_new_files_successfully(self, test_db_path: str, video_dir: Path) -> None:
        writer = DatabaseWriter(test_db_path)
        await writer.start()

        try:
            orchestrator = MagicMock(spec=TrebekPipelineOrchestrator)
            orchestrator.db_writer = writer
            orchestrator.mode = "once"
            orchestrator.gpu_work_ready = asyncio.Event()
            orchestrator.stats = {"total": 0, "completed": 0, "failed": 0}

            count = await run_ingestion_pass(orchestrator, str(video_dir))

            assert count == 2
            assert orchestrator.stats["total"] == 2
            assert orchestrator.gpu_work_ready.is_set()

            # Verify rows in database
            rows = await writer.execute("SELECT episode_id, status FROM pipeline_state ORDER BY episode_id")
            assert len(rows) == 2
            assert rows[0] == ("ep1", PipelineStatus.PENDING)
            assert rows[1] == ("ep2", PipelineStatus.PENDING)

            # Verify telemetry rows
            telemetry = await writer.execute(
                "SELECT episode_id, stage_ingestion_ms FROM job_telemetry ORDER BY episode_id"
            )
            assert len(telemetry) == 2
            assert telemetry[0][0] == "ep1"
            assert telemetry[0][1] is not None
            assert telemetry[0][1] >= 0.0

            # Verify registered_episode_ids cache populated
            assert orchestrator.registered_episode_ids == {"ep1", "ep2"}

        finally:
            await writer.stop()

    @pytest.mark.asyncio
    async def test_second_pass_does_zero_database_queries(self, test_db_path: str, video_dir: Path) -> None:
        """Verifies that subsequent passes with no new files do NOT query the database."""
        writer = DatabaseWriter(test_db_path)
        await writer.start()

        try:
            orchestrator = MagicMock(spec=TrebekPipelineOrchestrator)
            orchestrator.db_writer = writer
            orchestrator.mode = "once"
            orchestrator.gpu_work_ready = asyncio.Event()
            orchestrator.stats = {"total": 0, "completed": 0, "failed": 0}

            # First pass ingests 2 files
            count1 = await run_ingestion_pass(orchestrator, str(video_dir))
            assert count1 == 2

            # Reset event and mock db_writer.execute to track calls on second pass
            orchestrator.gpu_work_ready.clear()
            execute_mock = AsyncMock(wraps=writer.execute)
            orchestrator.db_writer.execute = execute_mock

            # Second pass: zero new files
            count2 = await run_ingestion_pass(orchestrator, str(video_dir))
            assert count2 == 0
            assert not orchestrator.gpu_work_ready.is_set()

            # Zero calls to execute! In-memory cache suppressed all queries
            assert execute_mock.call_count == 0

        finally:
            await writer.stop()

    @pytest.mark.asyncio
    async def test_incremental_new_file_ingestion(self, test_db_path: str, video_dir: Path) -> None:
        """Tests that adding a new file on a later pass only ingests the new file."""
        writer = DatabaseWriter(test_db_path)
        await writer.start()

        try:
            orchestrator = MagicMock(spec=TrebekPipelineOrchestrator)
            orchestrator.db_writer = writer
            orchestrator.mode = "once"
            orchestrator.gpu_work_ready = asyncio.Event()
            orchestrator.stats = {"total": 0, "completed": 0, "failed": 0}

            count1 = await run_ingestion_pass(orchestrator, str(video_dir))
            assert count1 == 2

            # Add a 3rd file
            (video_dir / "ep3.mp4").write_text("dummy_content_3")
            orchestrator.gpu_work_ready.clear()

            count2 = await run_ingestion_pass(orchestrator, str(video_dir))
            assert count2 == 1
            assert orchestrator.stats["total"] == 3
            assert orchestrator.gpu_work_ready.is_set()

            rows = await writer.execute("SELECT episode_id FROM pipeline_state ORDER BY episode_id")
            assert len(rows) == 3
            assert {r[0] for r in rows} == {"ep1", "ep2", "ep3"}

        finally:
            await writer.stop()

    @pytest.mark.asyncio
    async def test_nonexistent_or_empty_directory(self, test_db_path: str, tmp_path: Path) -> None:
        writer = DatabaseWriter(test_db_path)
        await writer.start()

        try:
            orchestrator = MagicMock(spec=TrebekPipelineOrchestrator)
            orchestrator.db_writer = writer
            orchestrator.mode = "once"
            orchestrator.gpu_work_ready = asyncio.Event()
            orchestrator.stats = {"total": 0, "completed": 0, "failed": 0}

            assert await run_ingestion_pass(orchestrator, "/nonexistent/dir") == 0

            empty_dir = tmp_path / "empty"
            empty_dir.mkdir()
            assert await run_ingestion_pass(orchestrator, str(empty_dir)) == 0

        finally:
            await writer.stop()

    @pytest.mark.asyncio
    async def test_filters_unstable_and_temporary_files(self, test_db_path: str, tmp_path: Path) -> None:
        writer = DatabaseWriter(test_db_path)
        await writer.start()

        try:
            vdir = tmp_path / "videos"
            vdir.mkdir()
            (vdir / "ready.mp4").write_text("valid content")
            (vdir / "empty.mp4").touch()  # 0 bytes
            (vdir / "download.mp4.part").write_text("downloading content")
            (vdir / ".hidden.mp4").write_text("hidden")

            orchestrator = MagicMock(spec=TrebekPipelineOrchestrator)
            orchestrator.db_writer = writer
            orchestrator.mode = "once"
            orchestrator.gpu_work_ready = asyncio.Event()
            orchestrator.stats = {"total": 0, "completed": 0, "failed": 0}

            count = await run_ingestion_pass(orchestrator, str(vdir))
            assert count == 1

            rows = await writer.execute("SELECT episode_id FROM pipeline_state")
            assert len(rows) == 1
            assert rows[0][0] == "ready"

        finally:
            await writer.stop()


class TestIngestionWorker:
    """Tests for ingestion_worker loop and termination."""

    @pytest.mark.asyncio
    async def test_worker_once_mode_terminates_immediately(self, tmp_path: Path) -> None:
        orchestrator = MagicMock(spec=TrebekPipelineOrchestrator)
        orchestrator.mode = "once"
        orchestrator.running = True

        with patch("trebek.pipeline.workers.ingestion.run_ingestion_pass", new_callable=AsyncMock) as mock_pass:
            await ingestion_worker(orchestrator, str(tmp_path))
            # In once mode, start_workers already ran run_ingestion_pass, worker breaks immediately
            assert mock_pass.call_count == 0

    @pytest.mark.asyncio
    async def test_worker_daemon_mode_runs_and_cancels(self, tmp_path: Path) -> None:
        orchestrator = MagicMock(spec=TrebekPipelineOrchestrator)
        orchestrator.mode = "daemon"
        orchestrator.running = True

        call_count = 0

        async def fake_pass(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                orchestrator.running = False  # stop after first pass

        with (
            patch("trebek.pipeline.workers.ingestion.run_ingestion_pass", side_effect=fake_pass),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await ingestion_worker(orchestrator, str(tmp_path))
            assert call_count == 1
            mock_sleep.assert_called_once_with(5)
