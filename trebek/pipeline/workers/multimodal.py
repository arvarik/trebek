import os
import time
import asyncio
import structlog
from typing import Any, TYPE_CHECKING
from trebek.ui import get_stage_display
from trebek.schemas import Episode
from trebek.llm import execute_pass_3_multimodal_augmentation
from trebek.config import MODEL_PRICING

if TYPE_CHECKING:
    from trebek.pipeline.orchestrator import TrebekPipelineOrchestrator

logger = structlog.get_logger()


async def multimodal_worker(orchestrator: "TrebekPipelineOrchestrator", progress: Any, task_id: Any) -> None:
    """Polls for SAVING, executes temporal sniping (Pass 3), and sets to MULTIMODAL_DONE."""
    current_episode_id: str | None = None
    try:
        while orchestrator.running:
            episode_id = await orchestrator.db_writer.poll_for_work("SAVING", "MULTIMODAL_PROCESSING")
            current_episode_id = episode_id
            if episode_id:
                logger.info(
                    "Multimodal Worker: Processing episode",
                    episode_id=episode_id,
                    stage=get_stage_display("MULTIMODAL_PROCESSING"),
                )
                try:
                    episode_data_path = os.path.join(orchestrator.output_dir, f"episode_{episode_id}.json")
                    from pathlib import Path
                    episode_json = await asyncio.to_thread(Path(episode_data_path).read_text, encoding="utf-8")
                    episode_data = await asyncio.to_thread(Episode.model_validate_json, episode_json)

                    # Look up the actual source filename
                    rows = await orchestrator.db_writer.execute(
                        "SELECT source_filename FROM pipeline_state WHERE episode_id = ?",
                        (episode_id,),
                    )
                    source_filename = rows[0][0] if rows and rows[0][0] else f"{episode_id}.mp4"
                    video_filepath = source_filename

                    start_multi_t = time.perf_counter()
                    episode_data, multi_usage = await execute_pass_3_multimodal_augmentation(
                        episode_data, video_filepath, orchestrator.output_dir,
                        model=orchestrator.llm_model,
                    )
                    stage_multimodal_ms = (time.perf_counter() - start_multi_t) * 1000

                    # Re-save episode data with augmented multimodal info
                    with open(episode_data_path, "w", encoding="utf-8") as f:
                        f.write(episode_data.model_dump_json())

                    # Update telemetry (dynamic pricing from config)
                    actual_model = orchestrator.llm_model
                    model_pricing = MODEL_PRICING.get(actual_model, {"input": 1.25, "output": 5.00})
                    cost_3 = (
                        multi_usage.get("input_tokens", 0) * model_pricing["input"]
                        + multi_usage.get("output_tokens", 0) * model_pricing["output"]
                    ) / 1_000_000

                    await orchestrator.db_writer.execute(
                        "UPDATE job_telemetry SET "
                        "stage_multimodal_ms = ?, "
                        "gemini_total_cost_usd = COALESCE(gemini_total_cost_usd, 0) + ?, "
                        "gemini_total_input_tokens = COALESCE(gemini_total_input_tokens, 0) + ?, "
                        "gemini_total_output_tokens = COALESCE(gemini_total_output_tokens, 0) + ? "
                        "WHERE episode_id = ?",
                        (
                            stage_multimodal_ms,
                            cost_3,
                            multi_usage.get("input_tokens", 0),
                            multi_usage.get("output_tokens", 0),
                            episode_id,
                        ),
                    )

                    await orchestrator.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'MULTIMODAL_DONE', updated_at = CURRENT_TIMESTAMP "
                        "WHERE episode_id = ?",
                        (episode_id,),
                    )
                    current_episode_id = None
                    if orchestrator.is_stage_active("verify"):
                        orchestrator.state_machine_work_ready.set()
                    else:
                        # Augment is the last active stage — advance progress
                        orchestrator.stats["completed"] += 1
                        progress.advance(task_id)
                    logger.info("Multimodal extraction complete", episode_id=episode_id)
                except Exception as e:
                    logger.error("Multimodal worker failed", error=str(e))
                    permanently_failed = await orchestrator.db_writer.fail_episode_with_retry(
                        episode_id, "SAVING", str(e)
                    )
                    current_episode_id = None
                    if permanently_failed:
                        orchestrator.stats["failed"] += 1
                    progress.advance(task_id)
            else:
                current_episode_id = None
                if orchestrator.mode == "once" and await orchestrator._no_work_remaining("SAVING"):
                    break
                orchestrator.multimodal_work_ready.clear()
                if orchestrator.mode == "daemon":
                    await orchestrator.multimodal_work_ready.wait()
                else:
                    try:
                        await asyncio.wait_for(orchestrator.multimodal_work_ready.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
    except asyncio.CancelledError:
        if current_episode_id:
            logger.warning("Multimodal worker cancelled, resetting episode", episode_id=current_episode_id)
            try:
                await orchestrator.db_writer.execute(
                    "UPDATE pipeline_state SET status = 'SAVING', updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                    (current_episode_id,),
                )
            except Exception:
                pass
