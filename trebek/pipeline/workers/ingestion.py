import os
import time
import asyncio
import structlog
from typing import TYPE_CHECKING
from trebek.status import PipelineStatus
from trebek.pipeline.discovery import async_scan_video_files

if TYPE_CHECKING:
    from trebek.pipeline.orchestrator import TrebekPipelineOrchestrator

logger = structlog.get_logger()


async def run_ingestion_pass(orchestrator: "TrebekPipelineOrchestrator", input_dir: str) -> int:
    """Performs a single pass to scan for video files.

    Scans input_dir asynchronously off the event loop, filters incomplete or unstable files,
    checks an in-memory registered cache to avoid redundant database roundtrips, and
    registers new video files as PENDING in pipeline_state.

    Returns the count of newly registered episodes.
    """
    if not os.path.exists(input_dir):
        return 0

    # In daemon mode, min_age_seconds=1.0 ensures active file copies have settled
    min_age = 1.0 if orchestrator.mode == "daemon" else 0.0
    scanned_files = await async_scan_video_files(input_dir, check_stability=True, min_age_seconds=min_age)

    if not scanned_files:
        return 0

    # Ensure in-memory cache of registered episode IDs exists
    if not hasattr(orchestrator, "registered_episode_ids"):
        rows = await orchestrator.db_writer.execute("SELECT episode_id FROM pipeline_state")
        orchestrator.registered_episode_ids = {r[0] for r in rows} if rows else set()

    # Filter out files whose derived episode_id is already registered
    new_candidates = [f for f in scanned_files if f["episode_id"] not in orchestrator.registered_episode_ids]
    if not new_candidates:
        return 0

    start_t = time.perf_counter()
    newly_inserted_ids: list[str] = []

    for f in new_candidates:
        ep_id = f["episode_id"]
        source_path = f["filepath"]

        result = await orchestrator.db_writer.execute(
            "INSERT OR IGNORE INTO pipeline_state (episode_id, status, source_filename) "
            "VALUES (?, ?, ?) RETURNING episode_id",
            (ep_id, PipelineStatus.PENDING, source_path),
        )

        is_new = isinstance(result, list) and len(result) > 0
        if is_new:
            newly_inserted_ids.append(ep_id)
            orchestrator.registered_episode_ids.add(ep_id)
        else:
            # Already existed in DB (e.g. from previous run before cache init)
            orchestrator.registered_episode_ids.add(ep_id)

    if newly_inserted_ids:
        stage_ingestion_ms = (time.perf_counter() - start_t) * 1000
        for ep_id in newly_inserted_ids:
            await orchestrator.db_writer.update_job_telemetry(ep_id, stage_ingestion_ms=stage_ingestion_ms)

        orchestrator.stats["total"] += len(newly_inserted_ids)
        # Notify GPU worker that work is ready
        orchestrator.gpu_work_ready.set()
        logger.info("Ingested new video files", count=len(newly_inserted_ids), episodes=newly_inserted_ids)

    return len(newly_inserted_ids)


async def ingestion_worker(orchestrator: "TrebekPipelineOrchestrator", input_dir: str) -> None:
    """Polls input_dir recursively for new video files across all supported formats."""
    while orchestrator.running:
        if orchestrator.mode == "once":
            # In once mode, start_workers already ran the initial ingestion pass
            break

        await asyncio.sleep(5)
        if not orchestrator.running:
            break
        await run_ingestion_pass(orchestrator, input_dir)
