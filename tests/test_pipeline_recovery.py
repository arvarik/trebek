import pytest
from unittest.mock import AsyncMock, MagicMock
from trebek.pipeline.orchestrator import TrebekPipelineOrchestrator
from trebek.pipeline.workers.state_machine import state_machine_worker
from trebek.database.writer import DatabaseWriter


@pytest.mark.asyncio
async def test_state_machine_worker_exception_recovery(memory_db_path: str) -> None:
    """Test that an unexpected exception in the worker prevents a zombie job by failing it."""
    # Set up DB with a job in MULTIMODAL_DONE
    writer = DatabaseWriter(memory_db_path)
    await writer.start()

    try:
        # We need the pipeline schema to insert a job
        import os

        schema_path = os.path.join(os.path.dirname(__file__), "..", "trebek", "schema.sql")
        with open(schema_path) as f:
            await writer.execute_transaction([(q, ()) for q in f.read().split(";") if q.strip()])

        await writer.execute(
            "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)", ("ep_zombie", "MULTIMODAL_DONE")
        )

        orchestrator = MagicMock(spec=TrebekPipelineOrchestrator)
        orchestrator.db_writer = writer
        orchestrator.running = True
        orchestrator.output_dir = "/tmp/does_not_exist"  # This will cause an exception when trying to read the file
        orchestrator.stats = {"completed": 0, "failed": 0}

        # We want to run the worker loop exactly once
        async def poll_side_effect(*args):
            if orchestrator.running:
                orchestrator.running = False  # Stop the loop after one pass
                return "ep_zombie"
            return None

        writer.poll_for_work = AsyncMock(side_effect=poll_side_effect)
        writer.fail_episode_with_retry = AsyncMock(return_value=True)  # It fails permanently

        progress = MagicMock()
        task_id = 1

        # Run worker
        await state_machine_worker(orchestrator, progress, task_id)

        # Verify fail_episode_with_retry was called
        writer.fail_episode_with_retry.assert_called_once()
        args, _ = writer.fail_episode_with_retry.call_args
        assert args[0] == "ep_zombie"
        assert args[1] == "MULTIMODAL_DONE"
        assert "Episode data file not found" in args[2]  # Exception message

        assert orchestrator.stats["failed"] == 1
    finally:
        await writer.stop()
