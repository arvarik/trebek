import asyncio
import sqlite3
import signal
import structlog
import os
import glob
import gzip
import json
from pathlib import Path
from typing import List, Any

from core_database import DatabaseWriter
from gpu_orchestrator import GPUOrchestrator
from config import settings
from llm_pipeline import execute_pass_1_speaker_anchoring, execute_pass_2_data_extraction
from state_machine import TrebekStateMachine
from schemas import Episode

# Configure structlog globally (basic configuration suitable for enterprise JSON lines or rich console)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger()


class TrebekPipelineOrchestrator:
    def __init__(self, db_path: str, output_dir: str) -> None:
        self.db_path = db_path
        self.output_dir = output_dir
        self.db_writer = DatabaseWriter(db_path)
        self.gpu_orchestrator = GPUOrchestrator(output_dir)
        self.running = False
        self.tasks: List[asyncio.Task[Any]] = []

    async def initialize(self) -> None:
        # Create schema if it doesn't exist using safe relative path
        schema_path = Path(__file__).parent / "schema.sql"
        with sqlite3.connect(self.db_path) as conn:
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())

        await self.db_writer.start()

    async def shutdown(self) -> None:
        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.gpu_orchestrator.shutdown()
        await self.db_writer.stop()
        logger.info("Pipeline Orchestrator shut down cleanly.")

    async def _ingestion_worker(self) -> None:
        """Polls input_dir for new videos and adds them to DB."""
        while self.running:
            input_dir = settings.input_dir
            if os.path.exists(input_dir):
                for video_file in glob.glob(os.path.join(input_dir, "*.mp4")):
                    episode_id = os.path.splitext(os.path.basename(video_file))[0]
                    await self.db_writer.execute(
                        "INSERT OR IGNORE INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                        (episode_id, "PENDING")
                    )
            await asyncio.sleep(5)

    async def _extractor_worker(self) -> None:
        """Polls for new files (PENDING) and sends to GPU."""
        while self.running:
            episode_id = await self.db_writer.poll_for_work("PENDING", "TRANSCRIBING")
            if episode_id:
                logger.info("Extractor: Processing episode", episode_id=episode_id)
                video_filepath = os.path.join(settings.input_dir, f"{episode_id}.mp4")
                if not os.path.exists(video_filepath):
                    logger.error("Video file not found", filepath=video_filepath)
                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'FAILED' WHERE episode_id = ?", (episode_id,)
                    )
                    continue

                try:
                    transcript_path = await self.gpu_orchestrator.execute_gpu_work(video_filepath)
                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'TRANSCRIPT_READY', transcript_path = ? WHERE episode_id = ?",
                        (transcript_path, episode_id),
                    )
                except Exception as e:
                    logger.error("GPU Orchestrator failed", error=str(e))
                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'FAILED' WHERE episode_id = ?", (episode_id,)
                    )
            else:
                await asyncio.sleep(2)

    async def _llm_worker(self) -> None:
        """Polls for TRANSCRIPT_READY, extracts data, and saves to state."""
        while self.running:
            episode_id = await self.db_writer.poll_for_work("TRANSCRIPT_READY", "CLEANED")
            if episode_id:
                logger.info("LLM Worker: Processing episode", episode_id=episode_id)
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
                        host_interview_segment = full_transcript[:1000]

                        speaker_mapping = await execute_pass_1_speaker_anchoring(host_interview_segment)
                        data = await execute_pass_2_data_extraction(full_transcript, speaker_mapping)
                        
                        episode_data_path = os.path.join(self.output_dir, f"episode_{episode_id}.json")
                        with open(episode_data_path, "w", encoding="utf-8") as f:
                            f.write(data.model_dump_json())

                        await self.db_writer.execute(
                            "UPDATE pipeline_state SET status = 'SAVING' WHERE episode_id = ?", (episode_id,)
                        )
                    else:
                        raise ValueError("Transcript path not found in database")
                except Exception as e:
                    logger.error("LLM Pipeline failed", error=str(e))
                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'FAILED' WHERE episode_id = ?", (episode_id,)
                    )
            else:
                await asyncio.sleep(2)

    async def _state_machine_worker(self) -> None:
        """Polls for SAVING, verifies game state, and inserts to DB."""
        while self.running:
            episode_id = await self.db_writer.poll_for_work("SAVING", "VECTORIZING")
            if episode_id:
                logger.info("State Machine: Verifying game state for episode", episode_id=episode_id)
                try:
                    episode_data_path = os.path.join(self.output_dir, f"episode_{episode_id}.json")
                    if os.path.exists(episode_data_path):
                        with open(episode_data_path, "r", encoding="utf-8") as f:
                            episode_json = f.read()
                        episode_data = Episode.model_validate_json(episode_json)

                        state_machine = TrebekStateMachine()
                        state_machine.load_adjustments(episode_data.score_adjustments)
                        for clue in episode_data.clues:
                            state_machine.process_clue(clue)
                        
                        # Assuming state machine passes without error, we mark as COMPLETED
                        await self.db_writer.execute(
                            "UPDATE pipeline_state SET status = 'COMPLETED' WHERE episode_id = ?", (episode_id,)
                        )
                    else:
                        raise ValueError(f"Episode data file not found: {episode_data_path}")
                except Exception as e:
                    logger.error("State Machine Verification failed", error=str(e))
                    await self.db_writer.execute(
                        "UPDATE pipeline_state SET status = 'FAILED' WHERE episode_id = ?", (episode_id,)
                    )
            else:
                await asyncio.sleep(2)

    async def start_workers(self) -> None:
        self.running = True
        self.tasks.append(asyncio.create_task(self._ingestion_worker()))
        self.tasks.append(asyncio.create_task(self._extractor_worker()))
        self.tasks.append(asyncio.create_task(self._llm_worker()))
        self.tasks.append(asyncio.create_task(self._state_machine_worker()))


async def main() -> None:
    orchestrator = TrebekPipelineOrchestrator(db_path=settings.db_path, output_dir=settings.output_dir)

    await orchestrator.initialize()
    await orchestrator.start_workers()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Received shutdown signal.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    logger.info("Trebek is running...")

    try:
        await stop_event.wait()
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
