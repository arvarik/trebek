import asyncio
import sqlite3
import signal
import structlog
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from trebek.database import DatabaseWriter
from trebek.gpu import GPUOrchestrator
from trebek.config import settings
from trebek.ui import (
    console,
    create_pipeline_progress,
)

from trebek.pipeline.workers.ingestion import ingestion_worker, run_ingestion_pass
from trebek.pipeline.workers.gpu import extractor_worker
from trebek.pipeline.workers.llm import llm_worker
from trebek.pipeline.workers.multimodal import multimodal_worker
from trebek.pipeline.workers.state_machine import state_machine_worker

# ──────────────────────────────────────────────────────────────
# Structlog Configuration — Rich in TTY, JSON when piped
# ──────────────────────────────────────────────────────────────


def _configure_logging() -> None:
    """Configures structlog with Rich ConsoleRenderer for TTY, JSONRenderer for piped output."""
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if sys.stderr.isatty():
        # Interactive terminal — beautiful Rich output
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
        )
    else:
        # Piped / redirected — machine-parseable JSON lines
        renderer = structlog.processors.JSONRenderer()  # type: ignore[assignment]

    processors: list[Any] = [*shared_processors, renderer]
    structlog.configure(processors=processors)


_configure_logging()
logger = structlog.get_logger()


class TrebekPipelineOrchestrator:
    def __init__(self, db_path: str, output_dir: str, mode: str = "daemon") -> None:
        self.db_path = db_path
        self.output_dir = output_dir
        self.mode = mode
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
        """Check if there are no more items in the target status OR any upstream status (for --once mode)."""
        upstream_map = {
            "PENDING": ["PENDING"],
            "TRANSCRIPT_READY": ["PENDING", "TRANSCRIBING", "TRANSCRIPT_READY"],
            "SAVING": ["PENDING", "TRANSCRIBING", "TRANSCRIPT_READY", "CLEANED", "SAVING"],
            "MULTIMODAL_DONE": [
                "PENDING",
                "TRANSCRIBING",
                "TRANSCRIPT_READY",
                "CLEANED",
                "SAVING",
                "MULTIMODAL_PROCESSING",
                "MULTIMODAL_DONE",
            ],
        }
        statuses_to_check = upstream_map.get(target_status, [target_status])

        placeholders = ",".join(["?"] * len(statuses_to_check))
        query = f"SELECT COUNT(*) FROM pipeline_state WHERE status IN ({placeholders})"
        result = await self.db_writer.execute(query, tuple(statuses_to_check))
        return result[0][0] == 0 if result else True

    async def start_workers(self, input_dir: str, progress: Any, task_id: Any) -> None:
        self.running = True

        # Wake up all workers for an initial database poll in case there's backlogged work
        self.gpu_work_ready.set()
        self.llm_work_ready.set()
        self.multimodal_work_ready.set()
        self.state_machine_work_ready.set()

        # Run ingestion first so we know the total count
        await run_ingestion_pass(self, input_dir)
        total = await self._get_total_episodes()
        progress.update(task_id, total=total)

        self.tasks.append(asyncio.create_task(ingestion_worker(self, input_dir)))
        self.tasks.append(asyncio.create_task(extractor_worker(self, progress, task_id)))
        self.tasks.append(asyncio.create_task(llm_worker(self, progress, task_id)))
        self.tasks.append(asyncio.create_task(multimodal_worker(self, progress, task_id)))
        self.tasks.append(asyncio.create_task(state_machine_worker(self, progress, task_id)))


async def run_pipeline(mode: str = "daemon", input_dir_override: Optional[str] = None) -> None:
    """Main pipeline entry point, called by cli.py."""
    from trebek.ui import render_startup_banner, render_system_diagnostics

    input_dir = input_dir_override or settings.input_dir

    # ── Branded startup ──
    render_startup_banner(mode=mode)
    render_system_diagnostics(settings)

    orchestrator = TrebekPipelineOrchestrator(
        db_path=settings.db_path,
        output_dir=settings.output_dir,
        mode=mode,
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
