"""
Pipeline orchestrator — coordinates all worker stages, manages lifecycle,
and provides the ``run_pipeline()`` entry point called by the CLI.

Worker dispatch, event-driven coordination, and shutdown sequencing
are all managed here. Stage definitions and logging configuration
are imported from their dedicated modules.
"""

import asyncio
import sqlite3
import signal
import structlog
import os
from pathlib import Path
from typing import Any, List, Optional

from trebek.database import DatabaseWriter
from trebek.gpu import GPUOrchestrator
from trebek.config import settings, MODEL_PRO
from trebek.ui import (
    console,
    create_pipeline_progress,
)
from trebek.pipeline.logging import configure_logging
from trebek.pipeline.stages import (
    ACTIVE_STAGES,
    UPSTREAM_MAP_FULL,
    UPSTREAM_MAP_ISOLATED,
)

from trebek.pipeline.workers.ingestion import ingestion_worker, run_ingestion_pass
from trebek.pipeline.workers.gpu import extractor_worker
from trebek.pipeline.workers.llm import llm_worker
from trebek.pipeline.workers.multimodal import multimodal_worker
from trebek.pipeline.workers.state_machine import state_machine_worker

configure_logging()
logger = structlog.get_logger()


class TrebekPipelineOrchestrator:
    def __init__(
        self,
        db_path: str,
        output_dir: str,
        mode: str = "daemon",
        stage: str = "all",
        llm_model: str = MODEL_PRO,
        max_retries: int = 3,
    ) -> None:
        self.db_path = db_path
        self.output_dir = output_dir
        self.mode = mode
        self.stage = stage
        self.llm_model = llm_model
        self.max_retries = max_retries
        self.db_writer = DatabaseWriter(db_path)
        self.gpu_orchestrator = GPUOrchestrator(
            output_dir,
            batch_size=settings.whisper_batch_size,
            compute_type=settings.whisper_compute_type,
        )
        self.running = False
        self.tasks: List[asyncio.Task[Any]] = []

        # Events for worker orchestration to avoid database polling
        self.gpu_work_ready = asyncio.Event()
        self.llm_work_ready = asyncio.Event()
        self.multimodal_work_ready = asyncio.Event()
        self.state_machine_work_ready = asyncio.Event()

        # Stats for shutdown summary
        self.stats = {"total": 0, "completed": 0, "failed": 0}

    def is_stage_active(self, stage_name: str) -> bool:
        """Check if a logical stage is active for the current --stage configuration."""
        return stage_name in ACTIVE_STAGES.get(self.stage, set())

    async def initialize(self, input_dir: str) -> None:
        # Ensure input/output directories exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create schema if it doesn't exist using safe relative path
        schema_path = Path(__file__).parent.parent / "schema.sql"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())

        await self.db_writer.start()

        # Cleanup intermediate files
        try:
            rows = await self.db_writer.execute("SELECT episode_id FROM pipeline_state WHERE status != 'COMPLETED'")
            incomplete_episodes = {row[0] for row in rows} if rows else set()

            for filename in os.listdir(self.output_dir):
                filepath = os.path.join(self.output_dir, filename)

                # Unmapped audio chunks from crashed gpu workers
                if filename.endswith(".wav"):
                    try:
                        os.remove(filepath)
                        logger.info("Cleaned up orphaned audio chunk", file=filename)
                    except OSError:
                        pass
                    continue

                # Intermediate JSONs for incomplete episodes
                if filename.startswith("episode_") and filename.endswith(".json"):
                    ep_id = filename[len("episode_") : -len(".json")]
                    if ep_id in incomplete_episodes:
                        try:
                            os.remove(filepath)
                            logger.info("Cleaned up intermediate JSON file", file=filename)
                        except OSError:
                            pass
        except Exception as e:
            logger.warning("Cleanup routine failed", error=str(e))

    async def shutdown(self) -> None:
        from trebek.ui.dashboard.summary import render_shutdown_summary

        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.gpu_orchestrator.shutdown()

        telemetry_stats = {}
        try:
            rows = await self.db_writer.execute("""
                SELECT 
                    SUM(gemini_total_input_tokens + gemini_total_output_tokens + gemini_total_cached_tokens),
                    SUM(gemini_total_cost_usd),
                    AVG(peak_vram_mb),
                    AVG(stage_gpu_extraction_ms)
                FROM job_telemetry
            """)
            if rows and rows[0][0] is not None:
                telemetry_stats = {
                    "total_tokens": float(rows[0][0] or 0),
                    "total_cost": float(rows[0][1] or 0.0),
                    "avg_peak_vram": float(rows[0][2] or 0.0),
                    "avg_extraction_ms": float(rows[0][3] or 0.0),
                }
        except Exception as e:
            logger.warning("Failed to fetch telemetry aggregates", error=str(e))

        await self.db_writer.stop()

        render_shutdown_summary(self.stats, telemetry_stats)
        logger.info("Pipeline Orchestrator shut down cleanly.")

    async def _get_total_episodes(self) -> int:
        """Returns the total number of episodes in the pipeline."""
        result = await self.db_writer.execute("SELECT COUNT(*) FROM pipeline_state")
        return result[0][0] if result else 0

    async def _no_work_remaining(self, target_status: str) -> bool:
        """Check if there are no more items in the target status or upstream (for --once mode).

        When running all stages, uses full upstream checking (waits for upstream workers).
        When running a single stage, uses isolated checking (only own statuses).
        """
        if self.stage == "all":
            upstream_map = UPSTREAM_MAP_FULL
        else:
            upstream_map = UPSTREAM_MAP_ISOLATED

        statuses_to_check = upstream_map.get(target_status, [target_status])

        placeholders = ",".join(["?"] * len(statuses_to_check))
        query = f"SELECT COUNT(*) FROM pipeline_state WHERE status IN ({placeholders})"
        result = await self.db_writer.execute(query, tuple(statuses_to_check))
        return result[0][0] == 0 if result else True

    async def start_workers(self, input_dir: str, progress: Any, task_id: Any) -> None:
        self.running = True

        active_stages: list[str] = []

        # ── Auto-reset FAILED episodes for stage-targeted re-runs ────
        # When running a specific stage, reset FAILED episodes back to the
        # status that makes them visible to that stage's worker.
        # This ensures `trebek run --stage transcribe --once` picks up
        # previously failed files without needing `trebek retry` first.
        _STAGE_RESET_STATUS: dict[str, str] = {
            "transcribe": "PENDING",
            "extract": "TRANSCRIPT_READY",
            "augment": "SAVING",
            "verify": "MULTIMODAL_DONE",
        }
        if self.stage != "all" and self.stage in _STAGE_RESET_STATUS:
            reset_to = _STAGE_RESET_STATUS[self.stage]
            reset_result = await self.db_writer.execute(
                "UPDATE pipeline_state SET status = ?, retry_count = 0, "
                "last_error = NULL, updated_at = CURRENT_TIMESTAMP "
                "WHERE status = 'FAILED' RETURNING episode_id",
                (reset_to,),
            )
            reset_count = len(reset_result) if isinstance(reset_result, list) else 0
            if reset_count > 0:
                logger.info(
                    "Auto-reset FAILED episodes for stage re-run",
                    count=reset_count,
                    stage=self.stage,
                    reset_to=reset_to,
                )

        # ── Transcription stage (ingestion + GPU) ────────────────────
        if self.is_stage_active("transcribe"):
            self.gpu_work_ready.set()
            await run_ingestion_pass(self, input_dir)
            total = await self._get_total_episodes()
            progress.update(task_id, total=total)
            self.tasks.append(asyncio.create_task(ingestion_worker(self, input_dir)))
            self.tasks.append(asyncio.create_task(extractor_worker(self, progress, task_id)))
            active_stages.append("transcribe")

        # ── LLM extraction stage ────────────────────────────────────
        if self.is_stage_active("extract"):
            self.llm_work_ready.set()
            # When running extract without transcribe, count episodes with work available
            if not self.is_stage_active("transcribe"):
                total = await self._get_total_episodes()
                progress.update(task_id, total=total)
            self.tasks.append(asyncio.create_task(llm_worker(self, progress, task_id)))
            active_stages.append("extract")

        # ── Multimodal augmentation stage ────────────────────────────
        if self.is_stage_active("augment"):
            self.multimodal_work_ready.set()
            if not self.is_stage_active("transcribe") and not self.is_stage_active("extract"):
                total = await self._get_total_episodes()
                progress.update(task_id, total=total)
            self.tasks.append(asyncio.create_task(multimodal_worker(self, progress, task_id)))
            active_stages.append("augment")

        # ── State machine verification stage ─────────────────────────
        if self.is_stage_active("verify"):
            self.state_machine_work_ready.set()
            if len(active_stages) == 0:
                total = await self._get_total_episodes()
                progress.update(task_id, total=total)
            self.tasks.append(asyncio.create_task(state_machine_worker(self, progress, task_id)))
            active_stages.append("verify")

        logger.info(
            "Pipeline workers started",
            stage=self.stage,
            active_stages=active_stages,
            mode=self.mode,
            llm_model=self.llm_model,
        )

        if not active_stages:
            logger.warning("No stages are active — nothing to do")


async def run_pipeline(
    mode: str = "daemon",
    input_dir_override: Optional[str] = None,
    stage: str = "all",
    llm_model: str = MODEL_PRO,
    max_retries: int = 3,
) -> None:
    """Main pipeline entry point, called by cli.py."""
    from trebek.ui import render_startup_banner, render_system_diagnostics

    input_dir = input_dir_override or settings.input_dir

    # ── Branded startup ──
    stage_label = f"{mode} • stage={stage}" if stage != "all" else mode
    render_startup_banner(mode=stage_label)
    render_system_diagnostics(settings)

    orchestrator = TrebekPipelineOrchestrator(
        db_path=settings.db_path,
        output_dir=settings.output_dir,
        mode=mode,
        stage=stage,
        llm_model=llm_model,
        max_retries=max_retries,
    )

    await orchestrator.initialize(input_dir)

    progress = create_pipeline_progress()

    with progress:
        task_id = progress.add_task("Processing episodes", total=None)

        await orchestrator.start_workers(input_dir, progress, task_id)

        if mode == "daemon":
            # Daemon mode — wait for signal
            loop = asyncio.get_running_loop()
            stop_event = asyncio.Event()

            def signal_handler() -> None:
                logger.info("Received shutdown signal.")
                stop_event.set()

            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, signal_handler)

            console.rule("[dim]Watching for new episodes • Press Ctrl+C to stop[/dim]", style="dim cyan")
            console.print()
            await stop_event.wait()
        else:
            # Once mode — wait for all workers to finish naturally
            console.rule("[dim]Processing queued episodes[/dim]", style="dim cyan")
            console.print()
            await asyncio.gather(*orchestrator.tasks, return_exceptions=True)

    await orchestrator.shutdown()


# Backward compatibility — allow `python src/main.py` to still work
if __name__ == "__main__":
    asyncio.run(run_pipeline(mode="daemon"))
