"""
Tests for Phase A orchestration and concurrency improvements:
1. Intermediate JSON cleanup preserves post-extraction states (SAVING, MULTIMODAL_*, VECTORIZING).
2. Configurable LLM worker concurrency and atomic work polling.
3. Total episodes and progress bar active count math.
4. Signal handling and try/finally shutdown guarantee.
5. In-flight status map and worker retry wake-up coordination.
"""

import asyncio
import gzip
import json
import os
import signal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trebek.config import Settings
from trebek.database import DatabaseWriter
from trebek.pipeline.orchestrator import TrebekPipelineOrchestrator, run_pipeline
from trebek.pipeline.stages import ALL_IN_FLIGHT_STATUSES, UPSTREAM_MAP_FULL
from trebek.pipeline.workers.state_machine import state_machine_worker
from trebek.status import PipelineStatus


@pytest.mark.asyncio
async def test_initialize_preserves_post_extraction_jsons(memory_db_path: str, tmp_path: Path) -> None:
    """Dangerous intermediate JSON deletion bug fix:

    Ensure initialize() only cleans up JSONs for PENDING, TRANSCRIBING,
    TRANSCRIPT_READY, CLEANED, or FAILED. Never delete files for SAVING,
    MULTIMODAL_PROCESSING, MULTIMODAL_DONE, VECTORIZING, or COMPLETED.
    """
    output_dir = tmp_path / "outputs"
    input_dir = tmp_path / "inputs"
    output_dir.mkdir()
    input_dir.mkdir()

    writer = DatabaseWriter(memory_db_path)
    await writer.start()
    try:
        # Insert test episodes across various statuses
        test_episodes = [
            ("ep_pending", PipelineStatus.PENDING),
            ("ep_transcribing", PipelineStatus.TRANSCRIBING),
            ("ep_failed", PipelineStatus.FAILED),
            ("ep_saving", PipelineStatus.SAVING),
            ("ep_multimodal_proc", PipelineStatus.MULTIMODAL_PROCESSING),
            ("ep_multimodal_done", PipelineStatus.MULTIMODAL_DONE),
            ("ep_vectorizing", PipelineStatus.VECTORIZING),
            ("ep_completed", PipelineStatus.COMPLETED),
        ]
        for ep_id, status in test_episodes:
            await writer.execute(
                "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                (ep_id, status),
            )

        # Create corresponding JSON files and an orphaned WAV file
        for ep_id, _ in test_episodes:
            json_file = output_dir / f"episode_{ep_id}.json"
            json_file.write_text('{"test": true}', encoding="utf-8")

        wav_file = output_dir / "temp_orphan.wav"
        wav_file.write_text("dummy audio", encoding="utf-8")
    finally:
        await writer.stop()

    orchestrator = TrebekPipelineOrchestrator(
        db_path=memory_db_path,
        output_dir=str(output_dir),
    )
    await orchestrator.initialize(str(input_dir))
    await orchestrator.shutdown()

    # Pre-extraction / failed files should be removed
    assert not (output_dir / "episode_ep_pending.json").exists()
    assert not (output_dir / "episode_ep_transcribing.json").exists()
    assert not (output_dir / "episode_ep_failed.json").exists()
    assert not (output_dir / "temp_orphan.wav").exists()

    # Post-extraction and completed files MUST be preserved
    assert (output_dir / "episode_ep_saving.json").exists()
    assert (output_dir / "episode_ep_multimodal_proc.json").exists()
    assert (output_dir / "episode_ep_multimodal_done.json").exists()
    assert (output_dir / "episode_ep_vectorizing.json").exists()
    assert (output_dir / "episode_ep_completed.json").exists()


@pytest.mark.asyncio
async def test_get_total_episodes_counts_active_work(memory_db_path: str, tmp_path: Path) -> None:
    """Progress bar math bug fix:

    _get_total_episodes() should calculate active work items remaining,
    excluding previously completed episodes.
    """
    writer = DatabaseWriter(memory_db_path)
    await writer.start()
    try:
        # Insert 80 COMPLETED episodes
        for i in range(80):
            await writer.execute(
                "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                (f"ep_done_{i}", PipelineStatus.COMPLETED),
            )
        # Insert 5 PENDING episodes
        for i in range(5):
            await writer.execute(
                "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                (f"ep_pending_{i}", PipelineStatus.PENDING),
            )
        # Insert 3 TRANSCRIPT_READY episodes
        for i in range(3):
            await writer.execute(
                "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                (f"ep_llm_{i}", PipelineStatus.TRANSCRIPT_READY),
            )
        # Insert 2 SAVING episodes
        for i in range(2):
            await writer.execute(
                "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                (f"ep_augment_{i}", PipelineStatus.SAVING),
            )
        # Insert 1 MULTIMODAL_DONE episode
        await writer.execute(
            "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
            ("ep_verify_1", PipelineStatus.MULTIMODAL_DONE),
        )
    finally:
        await writer.stop()

    output_dir = str(tmp_path / "outputs")

    # When stage == 'all', total should be all non-completed (5 + 3 + 2 + 1 = 11)
    orch_all = TrebekPipelineOrchestrator(db_path=memory_db_path, output_dir=output_dir, stage="all")
    await orch_all.db_writer.start()
    try:
        assert await orch_all._get_total_episodes() == 11
    finally:
        await orch_all.db_writer.stop()

    # When stage == 'transcribe', only PENDING and TRANSCRIBING are counted (5)
    orch_tx = TrebekPipelineOrchestrator(db_path=memory_db_path, output_dir=output_dir, stage="transcribe")
    await orch_tx.db_writer.start()
    try:
        assert await orch_tx._get_total_episodes() == 5
    finally:
        await orch_tx.db_writer.stop()

    # When stage == 'extract', only TRANSCRIPT_READY and CLEANED are counted (3)
    orch_ex = TrebekPipelineOrchestrator(db_path=memory_db_path, output_dir=output_dir, stage="extract")
    await orch_ex.db_writer.start()
    try:
        assert await orch_ex._get_total_episodes() == 3
    finally:
        await orch_ex.db_writer.stop()

    # When stage == 'augment', only SAVING and MULTIMODAL_PROCESSING are counted (2)
    orch_aug = TrebekPipelineOrchestrator(db_path=memory_db_path, output_dir=output_dir, stage="augment")
    await orch_aug.db_writer.start()
    try:
        assert await orch_aug._get_total_episodes() == 2
    finally:
        await orch_aug.db_writer.stop()

    # When stage == 'verify', only MULTIMODAL_DONE and VECTORIZING are counted (1)
    orch_ver = TrebekPipelineOrchestrator(db_path=memory_db_path, output_dir=output_dir, stage="verify")
    await orch_ver.db_writer.start()
    try:
        assert await orch_ver._get_total_episodes() == 1
    finally:
        await orch_ver.db_writer.stop()


@pytest.mark.asyncio
async def test_start_workers_spawns_llm_concurrency_tasks(memory_db_path: str, tmp_path: Path) -> None:
    """Verify that start_workers spawns the configured number of llm_worker tasks."""
    output_dir = tmp_path / "outputs"
    input_dir = tmp_path / "inputs"
    output_dir.mkdir()
    input_dir.mkdir()

    orchestrator = TrebekPipelineOrchestrator(
        db_path=memory_db_path,
        output_dir=str(output_dir),
        stage="extract",
        llm_concurrency=4,
    )
    await orchestrator.initialize(str(input_dir))
    try:
        progress = MagicMock()
        task_id = 1
        await orchestrator.start_workers(str(input_dir), progress, task_id)

        # 4 llm_worker tasks should have been spawned
        assert len(orchestrator.tasks) == 4
        task_names = [t.get_name() for t in orchestrator.tasks]
        assert task_names == ["llm_worker_1", "llm_worker_2", "llm_worker_3", "llm_worker_4"]
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_concurrent_llm_workers_poll_atomically(memory_db_path: str, tmp_path: Path) -> None:
    """Verify that multiple concurrent llm_worker instances consume jobs atomically without conflicts."""
    output_dir = tmp_path / "outputs"
    input_dir = tmp_path / "inputs"
    output_dir.mkdir()
    input_dir.mkdir()

    # Create dummy gzipped transcript files
    transcript_paths = {}
    for i in range(4):
        t_path = output_dir / f"transcript_{i}.json.gz"
        with gzip.open(t_path, "wt", encoding="utf-8") as f:
            json.dump({"transcript": {"segments": [{"text": "Sample J! text"}]}}, f)
        transcript_paths[i] = str(t_path)

    writer = DatabaseWriter(memory_db_path)
    await writer.start()
    try:
        for i in range(4):
            await writer.execute(
                "INSERT INTO pipeline_state (episode_id, status, transcript_path) VALUES (?, ?, ?)",
                (f"ep_{i}", PipelineStatus.TRANSCRIPT_READY, transcript_paths[i]),
            )
    finally:
        await writer.stop()

    orchestrator = TrebekPipelineOrchestrator(
        db_path=memory_db_path,
        output_dir=str(output_dir),
        mode="once",
        stage="extract",
        llm_concurrency=3,
    )
    await orchestrator.initialize(str(input_dir))

    async def mock_execute_pass_1(path: str):
        await asyncio.sleep(0.02)
        return {"SPEAKER_00": "Ken Jennings"}, {"cost_usd": 0.001, "latency_ms": 20.0}

    fake_output = MagicMock()
    fake_output.clues = []
    fake_output.contestants = []
    fake_output.model_dump_json.return_value = '{"clues": [], "contestants": []}'

    async def mock_execute_pass_2(segments, speaker_mapping, model=None):
        await asyncio.sleep(0.02)
        return fake_output, {"cost_usd": 0.002, "latency_ms": 20.0}, 0, "HIGH"

    progress = MagicMock()
    task_id = 1

    with (
        patch("trebek.pipeline.workers.llm.execute_pass_1_speaker_anchoring", side_effect=mock_execute_pass_1),
        patch("trebek.pipeline.workers.llm.execute_pass_2_data_extraction", side_effect=mock_execute_pass_2),
        patch("os.path.exists", return_value=True),
    ):
        await orchestrator.start_workers(str(input_dir), progress, task_id)
        await asyncio.gather(*orchestrator.tasks, return_exceptions=True)

    # Verify all 4 episodes were processed to SAVING and stats updated
    rows = await orchestrator.db_writer.execute("SELECT episode_id, status FROM pipeline_state")
    statuses = {r[0]: r[1] for r in rows}
    for i in range(4):
        assert statuses[f"ep_{i}"] == PipelineStatus.SAVING

    assert orchestrator.stats["completed"] == 4
    await orchestrator.shutdown()


def test_upstream_map_full_covers_all_in_flight_statuses() -> None:
    """Verify Issue 5 fix: UPSTREAM_MAP_FULL prevents premature worker exits in stage=='all'."""
    assert ALL_IN_FLIGHT_STATUSES == [
        PipelineStatus.PENDING,
        PipelineStatus.TRANSCRIBING,
        PipelineStatus.TRANSCRIPT_READY,
        PipelineStatus.CLEANED,
        PipelineStatus.SAVING,
        PipelineStatus.MULTIMODAL_PROCESSING,
        PipelineStatus.MULTIMODAL_DONE,
        PipelineStatus.VECTORIZING,
    ]

    # Every worker in UPSTREAM_MAP_FULL should check all in-flight statuses
    for status, upstream_list in UPSTREAM_MAP_FULL.items():
        assert upstream_list == ALL_IN_FLIGHT_STATUSES


@pytest.mark.asyncio
async def test_state_machine_worker_retry_signals_llm_worker(memory_db_path: str, tmp_path: Path) -> None:
    """Issue 5 fix: verify that when state_machine_worker retries an episode,

    it triggers orchestrator.llm_work_ready.set() and does not prematurely advance progress.
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    orchestrator = TrebekPipelineOrchestrator(
        db_path=memory_db_path,
        output_dir=str(output_dir),
        mode="once",
        stage="all",
    )
    await orchestrator.db_writer.start()
    try:
        await orchestrator.db_writer.execute(
            "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
            ("ep_retry_test", PipelineStatus.MULTIMODAL_DONE),
        )

        orchestrator.running = True
        orchestrator.llm_work_ready.clear()
        progress = MagicMock()
        task_id = 1

        # Run worker with poll returning the episode once
        call_count = 0

        async def mock_poll(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "ep_retry_test"
            orchestrator.running = False
            return None

        orchestrator.db_writer.poll_for_work = AsyncMock(side_effect=mock_poll)  # type: ignore[method-assign]
        # Simulate non-permanent failure (retry scheduled)
        orchestrator.db_writer.fail_episode_with_retry = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await state_machine_worker(orchestrator, progress, task_id)

        # Worker should have called fail_episode_with_retry with TRANSCRIPT_READY
        orchestrator.db_writer.fail_episode_with_retry.assert_called_once()
        args, _ = orchestrator.db_writer.fail_episode_with_retry.call_args
        assert args[0] == "ep_retry_test"
        assert args[1] == PipelineStatus.TRANSCRIPT_READY

        # Progress should NOT have advanced for retried episode
        progress.advance.assert_not_called()
        # llm_work_ready MUST have been signaled
        assert orchestrator.llm_work_ready.is_set()
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_run_pipeline_once_mode_signal_handler_triggers_shutdown(tmp_path: Path) -> None:
    """Issue 4 fix: test that signal handler in once mode cancels tasks and triggers clean shutdown."""
    input_dir = str(tmp_path / "inputs")
    os.makedirs(input_dir, exist_ok=True)

    with (
        patch.object(Settings, "require_gemini_api_key", return_value="dummy-key"),
        patch("trebek.ui.render_startup_banner"),
        patch("trebek.ui.render_system_diagnostics"),
    ):
        mock_orch = MagicMock(spec=TrebekPipelineOrchestrator)
        mock_orch.initialize = AsyncMock()
        mock_orch.shutdown = AsyncMock()
        mock_orch.running = True

        fake_task = asyncio.create_task(asyncio.sleep(10))
        mock_orch.tasks = [fake_task]

        async def fake_start_workers(*args):
            pass

        mock_orch.start_workers = AsyncMock(side_effect=fake_start_workers)

        with patch("trebek.pipeline.orchestrator.TrebekPipelineOrchestrator", return_value=mock_orch):
            loop = asyncio.get_running_loop()
            handlers: dict[int, Any] = {}

            def fake_add_signal_handler(sig, callback):
                handlers[sig] = callback

            def fake_remove_signal_handler(sig):
                handlers.pop(sig, None)

            with (
                patch.object(loop, "add_signal_handler", side_effect=fake_add_signal_handler),
                patch.object(loop, "remove_signal_handler", side_effect=fake_remove_signal_handler),
            ):
                # Run run_pipeline in background and trigger the installed signal handler
                run_coro = run_pipeline(mode="once", input_dir_override=input_dir)
                run_task = asyncio.create_task(run_coro)
                await asyncio.sleep(0.01)

                # Signal handler should be installed for SIGINT
                assert signal.SIGINT in handlers
                handlers[signal.SIGINT]()

                await run_task

                # Verify task was cancelled and shutdown was called
                assert fake_task.cancelled()
                mock_orch.shutdown.assert_called_once()
                # Handlers removed
                assert len(handlers) == 0
