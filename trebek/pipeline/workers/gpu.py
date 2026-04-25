import os
import time
import asyncio
import structlog
from typing import Any, TYPE_CHECKING
from trebek.ui import get_stage_display

if TYPE_CHECKING:
    from trebek.pipeline.orchestrator import TrebekPipelineOrchestrator

logger = structlog.get_logger()


async def extractor_worker(orchestrator: "TrebekPipelineOrchestrator", progress: Any, task_id: Any) -> None:
    """Polls for PENDING episodes and sends to GPU for transcription."""
    current_episode_id: str | None = None
    try:
        while orchestrator.running:
            episode_id = await orchestrator.db_writer.poll_for_work("PENDING", "TRANSCRIBING")
            current_episode_id = episode_id
            if episode_id:
                logger.info(
                    "Extractor: Processing episode",
                    episode_id=episode_id,
                    stage=get_stage_display("TRANSCRIBING"),
                )

                # Look up the actual source filename from the database
                rows = await orchestrator.db_writer.execute(
                    "SELECT source_filename FROM pipeline_state WHERE episode_id = ?",
                    (episode_id,),
                )
                source_filename = rows[0][0] if rows and rows[0][0] else f"{episode_id}.mp4"
                # source_filename is the full walk path from ingestion — use directly
                video_filepath = source_filename

                if not os.path.exists(video_filepath):
                    logger.error("Video file not found", filepath=video_filepath)
                    await orchestrator.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                        (episode_id,),
                    )
                    orchestrator.stats["failed"] += 1
                    progress.advance(task_id)
                    continue

                try:
                    start_t = time.perf_counter()
                    (
                        transcript_path,
                        peak_vram_mb,
                        avg_gpu_utilization_pct,
                    ) = await orchestrator.gpu_orchestrator.execute_gpu_work(video_filepath)
                    stage_gpu_extraction_ms = (time.perf_counter() - start_t) * 1000

                    await orchestrator.db_writer.update_job_telemetry(
                        episode_id,
                        stage_gpu_extraction_ms=stage_gpu_extraction_ms,
                        peak_vram_mb=peak_vram_mb,
                        avg_gpu_utilization_pct=avg_gpu_utilization_pct,
                    )

                    await orchestrator.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'TRANSCRIPT_READY', transcript_path = ?, "
                        "updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                        (transcript_path, episode_id),
                    )
                    current_episode_id = None
                    if not orchestrator.nollm:
                        orchestrator.llm_work_ready.set()
                    else:
                        # In --nollm mode, GPU is the final worker — advance the progress bar
                        orchestrator.stats["completed"] += 1
                        progress.advance(task_id)
                    logger.info("Transcription complete", episode_id=episode_id)
                except Exception as e:
                    logger.error("GPU Orchestrator failed", error=str(e))
                    await orchestrator.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                        (episode_id,),
                    )
                    current_episode_id = None
                    orchestrator.stats["failed"] += 1
                    progress.advance(task_id)
            else:
                current_episode_id = None
                if orchestrator.mode == "once" and await orchestrator._no_work_remaining("PENDING"):
                    break
                orchestrator.gpu_work_ready.clear()
                if orchestrator.mode == "daemon":
                    await orchestrator.gpu_work_ready.wait()
                else:
                    try:
                        await asyncio.wait_for(orchestrator.gpu_work_ready.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
    except asyncio.CancelledError:
        if current_episode_id:
            logger.warning("GPU worker cancelled, resetting episode", episode_id=current_episode_id)
            try:
                await orchestrator.db_writer.execute(
                    "UPDATE pipeline_state SET status = 'PENDING', updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                    (current_episode_id,),
                )
            except Exception:
                pass
