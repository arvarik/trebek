import os
import time
import asyncio
from typing import TYPE_CHECKING
from trebek.config import SUPPORTED_VIDEO_EXTENSIONS

if TYPE_CHECKING:
    from trebek.pipeline.orchestrator import TrebekPipelineOrchestrator


async def run_ingestion_pass(orchestrator: "TrebekPipelineOrchestrator", input_dir: str) -> None:
    """Performs a single pass to scan for video files."""
    if os.path.exists(input_dir):
        for dirpath, _dirnames, filenames in os.walk(input_dir):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in SUPPORTED_VIDEO_EXTENSIONS:
                    continue

                start_t = time.perf_counter()
                # Include parent folder in episode_id for uniqueness
                # e.g. "Season 41/S41E01.mp4" → "Season_41_S41E01"
                rel = os.path.relpath(os.path.join(dirpath, fname), input_dir)
                episode_id = os.path.splitext(rel)[0].replace(os.sep, "_").replace(" ", "_")
                source_path = os.path.join(dirpath, fname)

                await orchestrator.db_writer.execute(
                    "INSERT OR IGNORE INTO pipeline_state (episode_id, status, source_filename) VALUES (?, ?, ?)",
                    (episode_id, "PENDING", source_path),
                )

                # Notify GPU worker that work is ready
                orchestrator.gpu_work_ready.set()

                stage_ingestion_ms = (time.perf_counter() - start_t) * 1000
                await orchestrator.db_writer.update_job_telemetry(episode_id, stage_ingestion_ms=stage_ingestion_ms)

                orchestrator.stats["total"] += 1


async def ingestion_worker(orchestrator: "TrebekPipelineOrchestrator", input_dir: str) -> None:
    """Polls input_dir recursively for new video files across all supported formats."""
    while orchestrator.running:
        await run_ingestion_pass(orchestrator, input_dir)

        if orchestrator.mode == "once":
            # In once mode, stop polling after first scan
            break
        await asyncio.sleep(5)
