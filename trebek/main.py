import asyncio
import sqlite3
import signal
import structlog
import os
import gzip
import json
import sys
import uuid
from pathlib import Path
from typing import Any, List, Optional

from trebek.core_database import DatabaseWriter
from trebek.gpu_orchestrator import GPUOrchestrator
from trebek.config import settings, SUPPORTED_VIDEO_EXTENSIONS
from trebek.llm_pipeline import execute_pass_1_speaker_anchoring, execute_pass_2_data_extraction
from trebek.state_machine import TrebekStateMachine
from trebek.schemas import Episode
from trebek.console import (
    console,
    create_pipeline_progress,
    get_stage_display,
    render_shutdown_summary,
)

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
        self.gpu_orchestrator = GPUOrchestrator(output_dir)
        self.running = False
        self.tasks: List[asyncio.Task[Any]] = []

        # Stats for shutdown summary
        self.stats = {"total": 0, "completed": 0, "failed": 0}

    async def initialize(self, input_dir: str) -> None:
        # Ensure input/output directories exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create schema if it doesn't exist using safe relative path
        schema_path = Path(__file__).parent / "schema.sql"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())

        await self.db_writer.start()

    async def shutdown(self) -> None:
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

    async def _ingestion_worker(self, input_dir: str) -> None:
        """Polls input_dir for new video files across all supported formats."""
        import time
        while self.running:
            if os.path.exists(input_dir):
                for entry in os.scandir(input_dir):
                    if not entry.is_file():
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
                        continue

                    start_t = time.perf_counter()
                    episode_id = os.path.basename(os.path.splitext(entry.name)[0])
                    source_filename = entry.name

                    await self.db_writer.execute(
                        "INSERT OR IGNORE INTO pipeline_state (episode_id, status, source_filename) "
                        "VALUES (?, ?, ?)",
                        (episode_id, "PENDING", source_filename),
                    )
                    
                    stage_ingestion_ms = (time.perf_counter() - start_t) * 1000
                    await self.db_writer.update_job_telemetry(episode_id, stage_ingestion_ms=stage_ingestion_ms)
                    
                    self.stats["total"] += 1

            if self.mode == "once":
                # In once mode, stop polling after first scan
                break
            await asyncio.sleep(5)

    async def _extractor_worker(self, progress: Any, task_id: Any) -> None:
        """Polls for PENDING episodes and sends to GPU for transcription."""
        while self.running:
            episode_id = await self.db_writer.poll_for_work("PENDING", "TRANSCRIBING")
            if episode_id:
                logger.info(
                    "Extractor: Processing episode",
                    episode_id=episode_id,
                    stage=get_stage_display("TRANSCRIBING"),
                )

                # Look up the actual source filename from the database
                rows = await self.db_writer.execute(
                    "SELECT source_filename FROM pipeline_state WHERE episode_id = ?",
                    (episode_id,),
                )
                source_filename = rows[0][0] if rows and rows[0][0] else f"{episode_id}.mp4"
                video_filepath = os.path.join(settings.input_dir, source_filename)

                if not os.path.exists(video_filepath):
                    logger.error("Video file not found", filepath=video_filepath)
                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP "
                        "WHERE episode_id = ?",
                        (episode_id,),
                    )
                    self.stats["failed"] += 1
                    progress.advance(task_id)
                    continue

                try:
                    import time
                    start_t = time.perf_counter()
                    transcript_path, peak_vram_mb, avg_gpu_utilization_pct = await self.gpu_orchestrator.execute_gpu_work(video_filepath)
                    stage_gpu_extraction_ms = (time.perf_counter() - start_t) * 1000
                    
                    await self.db_writer.update_job_telemetry(
                        episode_id, 
                        stage_gpu_extraction_ms=stage_gpu_extraction_ms,
                        peak_vram_mb=peak_vram_mb,
                        avg_gpu_utilization_pct=avg_gpu_utilization_pct
                    )
                    
                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'TRANSCRIPT_READY', transcript_path = ?, "
                        "updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                        (transcript_path, episode_id),
                    )
                    logger.info("Transcription complete", episode_id=episode_id)
                except Exception as e:
                    logger.error("GPU Orchestrator failed", error=str(e))
                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP "
                        "WHERE episode_id = ?",
                        (episode_id,),
                    )
                    self.stats["failed"] += 1
                    progress.advance(task_id)
            else:
                if self.mode == "once" and await self._no_work_remaining("PENDING"):
                    break
                await asyncio.sleep(2)

    async def _llm_worker(self, progress: Any, task_id: Any) -> None:
        """Polls for TRANSCRIPT_READY, extracts data via LLM, and saves structured output."""
        while self.running:
            episode_id = await self.db_writer.poll_for_work("TRANSCRIPT_READY", "CLEANED")
            if episode_id:
                logger.info(
                    "LLM Worker: Processing episode",
                    episode_id=episode_id,
                    stage=get_stage_display("CLEANED"),
                )
                try:
                    rows = await self.db_writer.execute(
                        "SELECT transcript_path FROM pipeline_state WHERE episode_id = ?", (episode_id,)
                    )
                    if rows and rows[0][0]:
                        transcript_path = rows[0][0]
                        with gzip.open(transcript_path, "rt", encoding="utf-8") as f:
                            gpu_data = json.load(f)

                        transcript_data = gpu_data.get("transcript", {})
                        full_transcript = json.dumps(transcript_data)

                        # Use diarization speaker boundaries to locate interview segment
                        segments = transcript_data.get("segments", [])
                        interview_text = ""
                        if segments:
                            for seg in segments:
                                if isinstance(seg, dict):
                                    start_time = seg.get("start", 0)
                                    if start_time <= 120.0:
                                        interview_text += seg.get("text", "") + " "
                                    else:
                                        break
                        if not interview_text:
                            interview_text = full_transcript[:3000]

                        import time
                        start_llm_t = time.perf_counter()
                        speaker_mapping, usage1 = await execute_pass_1_speaker_anchoring(interview_text)
                        data, usage2, retries = await execute_pass_2_data_extraction(full_transcript, speaker_mapping)
                        
                        stage_structured_extraction_ms = (time.perf_counter() - start_llm_t) * 1000
                        
                        total_input = int(usage1.get("input_tokens", 0) + usage2.get("input_tokens", 0))
                        total_output = int(usage1.get("output_tokens", 0) + usage2.get("output_tokens", 0))
                        total_cached = int(usage1.get("cached_tokens", 0) + usage2.get("cached_tokens", 0))
                        total_latency = usage1.get("latency_ms", 0.0) + usage2.get("latency_ms", 0.0)
                        
                        cost_1 = (usage1.get("input_tokens", 0) * 0.075 + usage1.get("output_tokens", 0) * 0.30) / 1000000
                        cost_2 = (usage2.get("input_tokens", 0) * 1.25 + usage2.get("output_tokens", 0) * 5.00) / 1000000
                        total_cost = cost_1 + cost_2
                        
                        await self.db_writer.update_job_telemetry(
                            episode_id,
                            gemini_total_input_tokens=total_input,
                            gemini_total_output_tokens=total_output,
                            gemini_total_cached_tokens=total_cached,
                            gemini_api_latency_ms=total_latency,
                            gemini_total_cost_usd=total_cost,
                            pydantic_retry_count=retries,
                            stage_structured_extraction_ms=stage_structured_extraction_ms
                        )

                        episode_data_path = os.path.join(self.output_dir, f"episode_{episode_id}.json")
                        with open(episode_data_path, "w", encoding="utf-8") as f:
                            f.write(data.model_dump_json())

                        await self.db_writer.execute(
                            "UPDATE pipeline_state SET status = 'SAVING', updated_at = CURRENT_TIMESTAMP "
                            "WHERE episode_id = ?",
                            (episode_id,),
                        )
                        logger.info("LLM extraction complete", episode_id=episode_id)
                    else:
                        raise ValueError("Transcript path not found in database")
                except Exception as e:
                    logger.error("LLM Pipeline failed", error=str(e))
                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP "
                        "WHERE episode_id = ?",
                        (episode_id,),
                    )
                    self.stats["failed"] += 1
                    progress.advance(task_id)
            else:
                if self.mode == "once" and await self._no_work_remaining("TRANSCRIPT_READY"):
                    break
                await asyncio.sleep(2)

    async def _state_machine_worker(self, progress: Any, task_id: Any) -> None:
        """Polls for SAVING, verifies game state, and commits relational data to DB."""
        while self.running:
            episode_id = await self.db_writer.poll_for_work("SAVING", "VECTORIZING")
            if episode_id:
                logger.info(
                    "State Machine: Verifying game state",
                    episode_id=episode_id,
                    stage=get_stage_display("VECTORIZING"),
                )
                try:
                    episode_data_path = os.path.join(self.output_dir, f"episode_{episode_id}.json")
                    if not os.path.exists(episode_data_path):
                        raise ValueError(f"Episode data file not found: {episode_data_path}")

                    # Offload heavy Pydantic validation to a thread to protect the event loop
                    with open(episode_data_path, "r", encoding="utf-8") as f:
                        episode_json = f.read()
                    episode_data = await asyncio.to_thread(Episode.model_validate_json, episode_json)

                    # Run the deterministic state machine verification
                    import time
                    start_vec_t = time.perf_counter()
                    state_machine = TrebekStateMachine()
                    state_machine.load_adjustments(episode_data.score_adjustments)
                    for clue in episode_data.clues:
                        state_machine.process_clue(clue)

                    # Commit relational data to the analytical tables
                    await self._commit_relational_data(episode_id, episode_data, state_machine)
                    stage_vectorization_ms = (time.perf_counter() - start_vec_t) * 1000
                    await self.db_writer.update_job_telemetry(episode_id, stage_vectorization_ms=stage_vectorization_ms)

                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'COMPLETED', updated_at = CURRENT_TIMESTAMP "
                        "WHERE episode_id = ?",
                        (episode_id,),
                    )
                    self.stats["completed"] += 1
                    progress.advance(task_id)
                    logger.info(
                        "Episode completed successfully",
                        episode_id=episode_id,
                        stage=get_stage_display("COMPLETED"),
                    )

                except Exception as e:
                    logger.error("State Machine Verification failed", error=str(e))
                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP "
                        "WHERE episode_id = ?",
                        (episode_id,),
                    )
                    self.stats["failed"] += 1
                    progress.advance(task_id)
            else:
                if self.mode == "once" and await self._no_work_remaining("SAVING"):
                    break
                await asyncio.sleep(2)

    async def _commit_relational_data(
        self, episode_id: str, episode_data: Episode, state_machine: TrebekStateMachine
    ) -> None:
        """
        Commits verified episode data into the normalized relational tables.
        This is the Stage 8 relational commit — writing to episodes, contestants,
        clues, buzz_attempts, wagers, score_adjustments, and episode_performances.
        """
        # 1. Insert episode metadata
        await self.db_writer.execute(
            "INSERT OR IGNORE INTO episodes (episode_id, air_date, host_name, is_tournament) VALUES (?, ?, ?, ?)",
            (episode_id, episode_data.episode_date, episode_data.host_name, episode_data.is_tournament),
        )

        # 2. Insert contestants
        for contestant in episode_data.contestants:
            contestant_id = f"{episode_id}_{contestant.name.replace(' ', '_').lower()}"
            await self.db_writer.execute(
                "INSERT OR IGNORE INTO contestants "
                "(contestant_id, name, occupational_category, is_returning_champion) VALUES (?, ?, ?, ?)",
                (contestant_id, contestant.name, contestant.occupational_category, contestant.is_returning_champion),
            )

            # 3. Insert episode_performances with final scores from state machine
            final_score = state_machine.scores.get(contestant.name, 0)
            await self.db_writer.execute(
                "INSERT OR IGNORE INTO episode_performances "
                "(episode_id, contestant_id, podium_position, final_score) VALUES (?, ?, ?, ?)",
                (episode_id, contestant_id, contestant.podium_position, final_score),
            )

        # 4. Insert clues and buzz_attempts
        for clue in episode_data.clues:
            clue_id = f"{episode_id}_c{clue.selection_order}"
            is_triple_stumper = len(clue.attempts) == 0 or all(not a.is_correct for a in clue.attempts)

            dd_wager_str = str(clue.daily_double_wager) if clue.daily_double_wager is not None else None

            await self.db_writer.execute(
                "INSERT OR IGNORE INTO clues "
                "(clue_id, episode_id, round, category, board_row, board_col, selection_order, "
                "clue_text, correct_response, is_daily_double, is_triple_stumper, "
                "daily_double_wager, wagerer_name, requires_visual_context, "
                "host_start_timestamp_ms, host_finish_timestamp_ms, clue_syllable_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    clue_id, episode_id, clue.round, clue.category,
                    clue.board_row, clue.board_col, clue.selection_order,
                    clue.clue_text, clue.correct_response, clue.is_daily_double,
                    is_triple_stumper, dd_wager_str, clue.wagerer_name,
                    clue.requires_visual_context, clue.host_start_timestamp_ms,
                    clue.host_finish_timestamp_ms, clue.clue_syllable_count,
                ),
            )

            # Insert buzz_attempts for this clue
            for attempt in clue.attempts:
                attempt_id = f"{clue_id}_a{attempt.attempt_order}_{uuid.uuid4().hex[:8]}"
                contestant_id = f"{episode_id}_{attempt.speaker.replace(' ', '_').lower()}"
                await self.db_writer.execute(
                    "INSERT OR IGNORE INTO buzz_attempts "
                    "(attempt_id, clue_id, contestant_id, attempt_order, buzz_timestamp_ms, "
                    "response_given, is_correct, response_start_timestamp_ms, is_lockout_inferred) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        attempt_id, clue_id, contestant_id, attempt.attempt_order,
                        attempt.buzz_timestamp_ms, attempt.response_given,
                        attempt.is_correct, attempt.response_start_timestamp_ms,
                        attempt.is_lockout_inferred,
                    ),
                )

        # 5. Insert score_adjustments
        for adj in episode_data.score_adjustments:
            adjustment_id = f"{episode_id}_adj_{adj.effective_after_clue_selection_order}_{uuid.uuid4().hex[:8]}"
            contestant_id = f"{episode_id}_{adj.contestant.replace(' ', '_').lower()}"
            await self.db_writer.execute(
                "INSERT OR IGNORE INTO score_adjustments "
                "(adjustment_id, episode_id, contestant_id, points_adjusted, reason, "
                "effective_after_clue_selection_order) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    adjustment_id, episode_id, contestant_id,
                    adj.points_adjusted, adj.reason, adj.effective_after_clue_selection_order,
                ),
            )

        logger.info(
            "Relational commit complete",
            episode_id=episode_id,
            clues=len(episode_data.clues),
            contestants=len(episode_data.contestants),
        )

    async def _no_work_remaining(self, status: str) -> bool:
        """Check if there are no more items in the given status (for --once mode)."""
        result = await self.db_writer.execute(
            "SELECT COUNT(*) FROM pipeline_state WHERE status = ?", (status,)
        )
        return result[0][0] == 0 if result else True

    async def start_workers(self, input_dir: str, progress: Any, task_id: Any) -> None:
        self.running = True
        self.tasks.append(asyncio.create_task(self._ingestion_worker(input_dir)))
        self.tasks.append(asyncio.create_task(self._extractor_worker(progress, task_id)))
        self.tasks.append(asyncio.create_task(self._llm_worker(progress, task_id)))
        self.tasks.append(asyncio.create_task(self._state_machine_worker(progress, task_id)))


async def run_pipeline(mode: str = "daemon", input_dir_override: Optional[str] = None) -> None:
    """Main pipeline entry point, called by cli.py."""
    from console import render_startup_banner, render_system_diagnostics

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

