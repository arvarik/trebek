import os
import time
import asyncio
import structlog
from typing import Any, TYPE_CHECKING
from trebek.ui import get_stage_display
from trebek.schemas import Episode
from trebek.state_machine import TrebekStateMachine
from trebek.database import commit_episode_to_relational_tables

if TYPE_CHECKING:
    from trebek.pipeline.orchestrator import TrebekPipelineOrchestrator

logger = structlog.get_logger()


async def state_machine_worker(orchestrator: "TrebekPipelineOrchestrator", progress: Any, task_id: Any) -> None:
    """Polls for MULTIMODAL_DONE, verifies game state, and commits relational data to DB."""
    current_episode_id: str | None = None
    try:
        while orchestrator.running:
            episode_id = await orchestrator.db_writer.poll_for_work("MULTIMODAL_DONE", "VECTORIZING")
            current_episode_id = episode_id
            if episode_id:
                logger.info(
                    "State Machine: Verifying game state",
                    episode_id=episode_id,
                    stage=get_stage_display("VECTORIZING"),
                )
                try:
                    episode_data_path = os.path.join(orchestrator.output_dir, f"episode_{episode_id}.json")
                    if not os.path.exists(episode_data_path):
                        raise ValueError(f"Episode data file not found: {episode_data_path}")

                    # Offload heavy Pydantic validation to a thread to protect the event loop
                    from pathlib import Path
                    episode_json = await asyncio.to_thread(Path(episode_data_path).read_text, encoding="utf-8")
                    episode_data = await asyncio.to_thread(Episode.model_validate_json, episode_json)

                    # Run the deterministic state machine verification
                    start_vec_t = time.perf_counter()
                    state_machine = TrebekStateMachine()
                    state_machine.load_adjustments(episode_data.score_adjustments)
                    for clue in episode_data.clues:
                        state_machine.process_clue(clue)

                    # Commit relational data to the analytical tables
                    await commit_episode_to_relational_tables(
                        orchestrator.db_writer, episode_id, episode_data, state_machine
                    )
                    stage_vectorization_ms = (time.perf_counter() - start_vec_t) * 1000
                    await orchestrator.db_writer.update_job_telemetry(
                        episode_id, stage_vectorization_ms=stage_vectorization_ms
                    )

                    await orchestrator.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'COMPLETED', updated_at = CURRENT_TIMESTAMP "
                        "WHERE episode_id = ?",
                        (episode_id,),
                    )
                    current_episode_id = None
                    orchestrator.stats["completed"] += 1
                    progress.advance(task_id)
                    logger.info(
                        "Episode completed successfully",
                        episode_id=episode_id,
                        stage=get_stage_display("COMPLETED"),
                    )

                except Exception as e:
                    logger.error("State Machine Verification failed", error=str(e))
                    permanently_failed = await orchestrator.db_writer.fail_episode_with_retry(
                        episode_id, "MULTIMODAL_DONE", str(e)
                    )
                    current_episode_id = None
                    if permanently_failed:
                        orchestrator.stats["failed"] += 1
                    progress.advance(task_id)
            else:
                current_episode_id = None
                if orchestrator.mode == "once" and await orchestrator._no_work_remaining("MULTIMODAL_DONE"):
                    break
                orchestrator.state_machine_work_ready.clear()
                if orchestrator.mode == "daemon":
                    await orchestrator.state_machine_work_ready.wait()
                else:
                    try:
                        await asyncio.wait_for(orchestrator.state_machine_work_ready.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
    except asyncio.CancelledError:
        if current_episode_id:
            logger.warning("State machine worker cancelled, resetting episode", episode_id=current_episode_id)
            try:
                await orchestrator.db_writer.execute(
                    "UPDATE pipeline_state SET status = 'MULTIMODAL_DONE', updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                    (current_episode_id,),
                )
            except Exception:
                pass
