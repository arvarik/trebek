import os
import asyncio
import structlog
from trebek.schemas import Episode, Clue
from trebek.llm.client import _get_client

logger = structlog.get_logger()
GEMINI_CONCURRENCY = 3


async def execute_pass_3_multimodal_augmentation(
    episode: Episode, video_filepath: str, output_dir: str
) -> "tuple[Episode, dict[str, float]]":
    """
    Pass 3: Multimodal Vision Augmentation (Gemini 3.1 Pro Preview).
    Executes Temporal Sniping to find precise visual cues (like podium lights)
    without diluting the context window with the full video.
    """

    client = _get_client()
    total_usage: dict[str, float] = {"input_tokens": 0.0, "output_tokens": 0.0, "cached_tokens": 0.0, "latency_ms": 0.0}

    # Use a bounded semaphore to prevent ffmpeg/API floods
    semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)

    async def process_clue_multimodal(clue: Clue) -> None:
        async with semaphore:
            # 1. Temporal Sniping for podium lockouts
            if clue.attempts:
                clip_path = os.path.join(output_dir, f"podium_{clue.selection_order}.mp4")
                start_time = clue.host_finish_timestamp_ms / 1000.0

                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start_time:.3f}",
                    "-t",
                    "3.0",
                    "-i",
                    video_filepath,
                    "-c",
                    "copy",
                    clip_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()

                if proc.returncode == 0:
                    try:
                        uploaded_file = await client.upload_file(clip_path)

                        # Wait for file to become ACTIVE (video processing takes 1-5s)
                        for poll_i in range(15):
                            file_info = await asyncio.to_thread(client.client.files.get, name=uploaded_file.name)
                            state_name = file_info.state.name if file_info.state else "UNKNOWN"
                            if state_name == "ACTIVE":
                                break
                            if state_name == "FAILED":
                                raise RuntimeError(f"File processing failed: {uploaded_file.name}")
                            await asyncio.sleep(1)
                        else:
                            logger.warning("File never became ACTIVE, skipping", file=uploaded_file.name)
                            await client.delete_file(uploaded_file.name)
                            return

                        prompt = [
                            "Watch this 3-second clip immediately following the host finishing the clue. "
                            "Determine if any contestant's podium indicator light illuminates, indicating a buzz. "
                            "Return ONLY the float timestamp (in seconds, relative to the clip start) when the light turns on. "
                            "If no light turns on, return -1.0.",
                            uploaded_file,
                        ]
                        response, usage = await client.generate_content(
                            model="gemini-3.1-pro-preview",
                            prompt=prompt,
                            system_instruction="You are a precise temporal grounding model. Return ONLY a float.",
                            max_output_tokens=64,
                        )
                        for k in total_usage:
                            total_usage[k] += usage.get(k, 0.0)

                        await client.delete_file(uploaded_file.name)
                    except Exception as e:
                        logger.warning(
                            "Temporal sniping failed for clue", clue_order=clue.selection_order, error=str(e)
                        )

                    try:
                        os.remove(clip_path)
                    except OSError:
                        pass

    # For demonstration, only process a few visual context or podium items
    # to avoid overwhelming the API in a single run. We'll filter for visual context clues.
    tasks = [process_clue_multimodal(c) for c in episode.clues if c.requires_visual_context]
    if tasks:
        logger.info("Executing multimodal temporal sniping", tasks=len(tasks))
        await asyncio.gather(*tasks)

    return episode, total_usage
