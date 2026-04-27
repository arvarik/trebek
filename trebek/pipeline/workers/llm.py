import os
import time
import asyncio
import gzip
import json
import structlog
from typing import Any, TYPE_CHECKING
from trebek.ui import get_stage_display
from trebek.llm import execute_pass_1_speaker_anchoring, execute_pass_2_data_extraction

if TYPE_CHECKING:
    from trebek.pipeline.orchestrator import TrebekPipelineOrchestrator

logger = structlog.get_logger()


async def llm_worker(orchestrator: "TrebekPipelineOrchestrator", progress: Any, task_id: Any) -> None:
    """Polls for TRANSCRIPT_READY, extracts data via LLM, and saves structured output."""
    current_episode_id: str | None = None
    try:
        while orchestrator.running:
            episode_id = await orchestrator.db_writer.poll_for_work("TRANSCRIPT_READY", "CLEANED")
            current_episode_id = episode_id
            if episode_id:
                logger.info(
                    "LLM Worker: Processing episode",
                    episode_id=episode_id,
                    stage=get_stage_display("CLEANED"),
                )
                try:
                    rows = await orchestrator.db_writer.execute(
                        "SELECT transcript_path FROM pipeline_state WHERE episode_id = ?", (episode_id,)
                    )
                    if rows and rows[0][0]:
                        transcript_path = rows[0][0]
                        logger.info("Loading transcript", episode_id=episode_id, path=transcript_path)

                        def _load_transcript(path: str) -> dict[str, Any]:
                            with gzip.open(path, "rt", encoding="utf-8") as f:
                                return json.load(f)  # type: ignore[no-any-return, unused-ignore]

                        gpu_data: dict[str, Any] = await asyncio.to_thread(_load_transcript, transcript_path)

                        transcript_data = gpu_data.get("transcript", {})
                        segments = transcript_data.get("segments", [])
                        logger.info(
                            "Transcript loaded",
                            episode_id=episode_id,
                            segments=len(segments),
                        )

                        # Audio-Anchoring Domain Flaw: Extract exact 5-minute block
                        # starting around the 6-minute mark (host interview)
                        rows_src = await orchestrator.db_writer.execute(
                            "SELECT source_filename FROM pipeline_state WHERE episode_id = ?", (episode_id,)
                        )
                        source_filename = rows_src[0][0] if rows_src and rows_src[0][0] else f"{episode_id}.mp4"
                        video_filepath = source_filename

                        audio_slice_path = os.path.join(orchestrator.output_dir, f"{episode_id}_interview_slice.mp3")
                        if not os.path.exists(audio_slice_path):
                            logger.info("Extracting host interview audio slice", episode_id=episode_id)
                            proc = await asyncio.create_subprocess_exec(
                                "ffmpeg",
                                "-y",
                                "-ss",
                                "00:06:00",
                                "-t",
                                "00:05:00",
                                "-i",
                                video_filepath,
                                "-vn",
                                "-acodec",
                                "libmp3lame",
                                audio_slice_path,
                                stdout=asyncio.subprocess.DEVNULL,
                                stderr=asyncio.subprocess.DEVNULL,
                            )
                            await proc.wait()
                            if proc.returncode != 0:
                                logger.error("Failed to extract audio slice with ffmpeg")
                                raise RuntimeError("ffmpeg audio extraction failed")

                        start_llm_t = time.perf_counter()
                        speaker_mapping, usage1 = await execute_pass_1_speaker_anchoring(audio_slice_path)
                        pass1_ms = (time.perf_counter() - start_llm_t) * 1000
                        logger.info(
                            "Pass 1 complete",
                            episode_id=episode_id,
                            speakers=len(speaker_mapping),
                            speaker_mapping=speaker_mapping,
                            pass1_ms=round(pass1_ms, 0),
                            cost_usd=round(usage1.get("cost_usd", 0.0), 6),
                        )

                        pass2_start = time.perf_counter()
                        data, usage2, retries = await execute_pass_2_data_extraction(
                            segments,
                            speaker_mapping,
                            model=orchestrator.llm_model,
                        )
                        pass2_ms = (time.perf_counter() - pass2_start) * 1000

                        stage_structured_extraction_ms = (time.perf_counter() - start_llm_t) * 1000

                        total_input = int(usage1.get("input_tokens", 0) + usage2.get("input_tokens", 0))
                        total_output = int(usage1.get("output_tokens", 0) + usage2.get("output_tokens", 0))
                        total_cached = int(usage1.get("cached_tokens", 0) + usage2.get("cached_tokens", 0))
                        total_latency = usage1.get("latency_ms", 0.0) + usage2.get("latency_ms", 0.0)

                        total_cost = usage1.get("cost_usd", 0.0) + usage2.get("cost_usd", 0.0)

                        logger.info(
                            "Pass 2 complete",
                            episode_id=episode_id,
                            clues_extracted=len(data.clues),
                            contestants=[c.name for c in data.contestants],
                            pass2_ms=round(pass2_ms, 0),
                            total_input_tokens=total_input,
                            total_output_tokens=total_output,
                            retries_used=retries,
                            cost_usd=round(total_cost, 6),
                            model=orchestrator.llm_model,
                        )

                        await orchestrator.db_writer.update_job_telemetry(
                            episode_id,
                            gemini_total_input_tokens=total_input,
                            gemini_total_output_tokens=total_output,
                            gemini_total_cached_tokens=total_cached,
                            gemini_api_latency_ms=total_latency,
                            gemini_total_cost_usd=total_cost,
                            pydantic_retry_count=retries,
                            stage_structured_extraction_ms=stage_structured_extraction_ms,
                        )

                        episode_data_path = os.path.join(orchestrator.output_dir, f"episode_{episode_id}.json")
                        with open(episode_data_path, "w", encoding="utf-8") as f:
                            f.write(data.model_dump_json())

                        await orchestrator.db_writer.execute(
                            "UPDATE pipeline_state SET status = 'SAVING', updated_at = CURRENT_TIMESTAMP "
                            "WHERE episode_id = ?",
                            (episode_id,),
                        )
                        current_episode_id = None
                        if orchestrator.is_stage_active("augment"):
                            orchestrator.multimodal_work_ready.set()
                        else:
                            # Extract is the last active stage — advance progress
                            orchestrator.stats["completed"] += 1
                            progress.advance(task_id)
                        logger.info("LLM extraction complete", episode_id=episode_id)
                    else:
                        raise ValueError("Transcript path not found in database")
                except Exception as e:
                    logger.error(
                        "LLM Pipeline failed",
                        episode_id=episode_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,
                    )
                    permanently_failed = await orchestrator.db_writer.fail_episode_with_retry(
                        episode_id, "TRANSCRIPT_READY", str(e)
                    )
                    current_episode_id = None
                    if permanently_failed:
                        orchestrator.stats["failed"] += 1
                    progress.advance(task_id)
            else:
                current_episode_id = None
                if orchestrator.mode == "once" and await orchestrator._no_work_remaining("TRANSCRIPT_READY"):
                    break
                orchestrator.llm_work_ready.clear()
                if orchestrator.mode == "daemon":
                    await orchestrator.llm_work_ready.wait()
                else:
                    try:
                        await asyncio.wait_for(orchestrator.llm_work_ready.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
    except asyncio.CancelledError:
        if current_episode_id:
            logger.warning("LLM worker cancelled, resetting episode", episode_id=current_episode_id)
            try:
                await orchestrator.db_writer.execute(
                    "UPDATE pipeline_state SET status = 'TRANSCRIPT_READY', updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                    (current_episode_id,),
                )
            except Exception:
                pass
