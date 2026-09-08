import os
import re
import uuid
import asyncio
import contextlib
import structlog

from trebek.schemas import Episode, Clue
from trebek.llm.client import _get_client
from trebek.config import MODEL_PRO
from trebek.analysis.buzzer import calculate_true_buzzer_latency

logger = structlog.get_logger()
GEMINI_CONCURRENCY = 3


async def _extract_video_clip(
    video_filepath: str,
    clip_path: str,
    start_time: float,
    duration: float,
    ep_id: str,
    clue_order: int,
) -> bool:
    """Invokes ffmpeg to snip a short clip from video_filepath."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            video_filepath,
            "-c",
            "copy",
            clip_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await proc.communicate()
        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace")[-200:] if stderr_bytes else "no stderr"
            logger.warning(
                "Pass 3: ffmpeg clip extraction failed",
                episode_id=ep_id,
                clue_order=clue_order,
                returncode=proc.returncode,
                stderr=stderr_text,
            )
            return False
        return True
    except Exception as e:
        logger.warning(
            "Pass 3: ffmpeg subprocess error",
            episode_id=ep_id,
            clue_order=clue_order,
            error=str(e)[:200],
        )
        return False


async def _wait_for_file_active(client: object, uploaded_file: object) -> bool:
    """Polls Gemini Files API until the uploaded file state is ACTIVE."""
    for _ in range(15):
        if getattr(client, "client", None) is None:
            break
        file_info = await asyncio.to_thread(getattr(client, "client").files.get, name=getattr(uploaded_file, "name"))
        state_name = file_info.state.name if file_info.state else "UNKNOWN"
        if state_name == "ACTIVE":
            return True
        if state_name == "FAILED":
            return False
        await asyncio.sleep(1)
    return True


async def extract_visual_clue_context(
    clue: Clue,
    video_filepath: str,
    output_dir: str,
    ep_id: str,
    client: object,
    semaphore: asyncio.Semaphore,
    model: str = MODEL_PRO,
) -> dict[str, float]:
    """Extracts on-screen visual clue context during host reading.

    Runs when clue.requires_visual_context is True, regardless of whether
    contestants buzzed (supports triple stumpers). The clip is extracted
    while the host reads the clue (host_start to host_finish).
    """
    async with semaphore:
        start_ms = clue.host_start_timestamp_ms or 0.0
        finish_ms = clue.host_finish_timestamp_ms or (start_ms + 3000.0)
        start_time = max(0.0, start_ms / 1000.0)
        duration = max(1.0, min(6.0, (finish_ms - start_ms) / 1000.0)) if finish_ms > start_ms else 3.0

        clip_token = uuid.uuid4().hex[:8]
        clip_path = os.path.join(output_dir, f"visual_{ep_id}_{clue.selection_order}_{clip_token}.mp4")

        logger.info(
            "Pass 3: extracting visual clue clip",
            episode_id=ep_id,
            clue_order=clue.selection_order,
            clip_start_s=round(start_time, 3),
            clip_duration_s=round(duration, 3),
        )

        uploaded_file = None
        try:
            success = await _extract_video_clip(
                video_filepath, clip_path, start_time, duration, ep_id, clue.selection_order
            )
            if not success:
                return {}

            try:
                uploaded_file = await getattr(client, "upload_file")(clip_path)
                if not await _wait_for_file_active(client, uploaded_file):
                    logger.warning("File failed to become ACTIVE", file=getattr(uploaded_file, "name", "unknown"))
                    return {}

                prompt = [
                    "Watch this video clip of the Jeopardy! clue displayed on screen while the host reads it. "
                    "Describe the visual clue, picture, artwork, map, diagram, or video shown on the screen concisely in 1-2 sentences. "
                    "Identify the key subject, painting, map location, or text visible on screen.",
                    uploaded_file,
                ]
                response, usage = await getattr(client, "generate_content")(
                    model=model,
                    prompt=prompt,
                    system_instruction="You are an expert at identifying and describing visual Jeopardy! clues.",
                    max_output_tokens=65536,
                    invocation_context=f"Pass 3 Visual Context (clue {clue.selection_order})",
                    thinking_level="low",
                )

                result_text = str(response.text).strip() if response.text else ""
                clue.visual_context_description = result_text
                logger.info(
                    "Pass 3: visual context extracted",
                    episode_id=ep_id,
                    clue_order=clue.selection_order,
                    description=result_text[:80],
                )
                return usage
            except Exception as e:
                logger.warning(
                    "Visual context extraction failed for clue",
                    episode_id=ep_id,
                    clue_order=clue.selection_order,
                    error=str(e)[:200],
                )
                return {}
            finally:
                if uploaded_file is not None:
                    with contextlib.suppress(Exception):
                        await getattr(client, "delete_file")(uploaded_file.name)
        finally:
            if os.path.exists(clip_path):
                with contextlib.suppress(OSError):
                    os.remove(clip_path)


async def extract_podium_lockout_sniping(
    clue: Clue,
    video_filepath: str,
    output_dir: str,
    ep_id: str,
    client: object,
    semaphore: asyncio.Semaphore,
    model: str = MODEL_PRO,
) -> dict[str, float]:
    """Extracts post-read podium clip (host_finish to host_finish + 3s) for buzzer lockout detection."""
    if not clue.attempts or not clue.host_finish_timestamp_ms or clue.host_finish_timestamp_ms <= 0:
        return {}

    async with semaphore:
        start_time = clue.host_finish_timestamp_ms / 1000.0
        clip_token = uuid.uuid4().hex[:8]
        clip_path = os.path.join(output_dir, f"podium_{ep_id}_{clue.selection_order}_{clip_token}.mp4")

        logger.info(
            "Pass 3: extracting podium lockout clip",
            episode_id=ep_id,
            clue_order=clue.selection_order,
            clip_start_s=round(start_time, 3),
            clip_duration_s=3.0,
        )

        uploaded_file = None
        try:
            success = await _extract_video_clip(video_filepath, clip_path, start_time, 3.0, ep_id, clue.selection_order)
            if not success:
                return {}

            try:
                uploaded_file = await getattr(client, "upload_file")(clip_path)
                if not await _wait_for_file_active(client, uploaded_file):
                    return {}

                prompt = [
                    "Watch this 3-second clip immediately following the host finishing the clue. "
                    "Determine if any contestant's podium indicator light illuminates, indicating a buzz. "
                    "Return ONLY the float timestamp (in seconds, relative to the clip start) when the light turns on. "
                    "If no light turns on, return -1.0.",
                    uploaded_file,
                ]
                response, usage = await getattr(client, "generate_content")(
                    model=model,
                    prompt=prompt,
                    system_instruction="You are a precise temporal grounding model. Return ONLY a float.",
                    max_output_tokens=65536,
                    invocation_context=f"Pass 3 Temporal Sniping (clue {clue.selection_order})",
                    thinking_level="low",
                )

                result_text = str(response.text).strip() if response.text else "-1.0"
                match = re.search(r"[-+]?\d*\.?\d+", result_text)
                if match:
                    offset_s = float(match.group())
                    if offset_s >= 0.0:
                        podium_ms = (start_time + offset_s) * 1000.0
                        clue.attempts[0].podium_light_timestamp_ms = podium_ms
                        clue.attempts[0].true_buzzer_latency_ms = calculate_true_buzzer_latency(
                            clue.attempts[0].buzz_timestamp_ms, podium_ms
                        )
                return usage
            except Exception as e:
                logger.warning(
                    "Podium lockout sniping failed for clue",
                    episode_id=ep_id,
                    clue_order=clue.selection_order,
                    error=str(e)[:200],
                )
                return {}
            finally:
                if uploaded_file is not None:
                    with contextlib.suppress(Exception):
                        await getattr(client, "delete_file")(uploaded_file.name)
        finally:
            if os.path.exists(clip_path):
                with contextlib.suppress(OSError):
                    os.remove(clip_path)


async def execute_pass_3_multimodal_augmentation(
    episode: Episode,
    video_filepath: str,
    output_dir: str,
    model: str = MODEL_PRO,
    episode_id: str | None = None,
    enable_podium_sniping: bool = False,
) -> "tuple[Episode, dict[str, float]]":
    """
    Pass 3: Multimodal Vision Augmentation (Gemini 3.1 Pro Preview).

    Differentiates between:
    1. Visual Clue Context Extraction (during-read clip, host_start to host_finish) for
       all clues marked requires_visual_context, including triple stumpers.
    2. Podium Lockout Sniping (post-read clip, host_finish to host_finish + 3s) for
       clues with buzz attempts when enable_podium_sniping is True.
    """
    from trebek.config import settings

    ep_id = episode_id or getattr(episode, "episode_id", None) or os.path.splitext(os.path.basename(video_filepath))[0]
    total_usage: dict[str, float] = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "thinking_tokens": 0.0,
        "cached_tokens": 0.0,
        "total_tokens": 0.0,
        "cost_usd": 0.0,
        "latency_ms": 0.0,
    }

    if getattr(settings, "mock_llm", False):
        logger.info("Pass 3: Mock LLM mode active, populating synthetic multimodal data", episode_id=ep_id)
        for clue in episode.clues:
            if clue.requires_visual_context and not clue.visual_context_description:
                clue.visual_context_description = (
                    f"Synthetic visual context description for clue {clue.selection_order}."
                )
            if clue.attempts and clue.host_finish_timestamp_ms > 0:
                if not clue.attempts[0].podium_light_timestamp_ms:
                    podium_ms = clue.host_finish_timestamp_ms + 150.0
                    clue.attempts[0].podium_light_timestamp_ms = podium_ms
                    clue.attempts[0].true_buzzer_latency_ms = calculate_true_buzzer_latency(
                        clue.attempts[0].buzz_timestamp_ms, podium_ms
                    )
        return episode, total_usage

    client = _get_client()
    semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)

    tasks = []
    # 1. Visual clue context extraction for all visual clues (including triple stumpers)
    for c in episode.clues:
        if c.requires_visual_context:
            tasks.append(
                extract_visual_clue_context(
                    clue=c,
                    video_filepath=video_filepath,
                    output_dir=output_dir,
                    ep_id=ep_id,
                    client=client,
                    semaphore=semaphore,
                    model=model,
                )
            )

    # 2. Podium lockout sniping for clues with contestant buzz attempts if enabled
    if enable_podium_sniping:
        for c in episode.clues:
            if c.attempts and c.host_finish_timestamp_ms and c.host_finish_timestamp_ms > 0:
                tasks.append(
                    extract_podium_lockout_sniping(
                        clue=c,
                        video_filepath=video_filepath,
                        output_dir=output_dir,
                        ep_id=ep_id,
                        client=client,
                        semaphore=semaphore,
                        model=model,
                    )
                )

    if tasks:
        logger.info("Executing Pass 3 multimodal vision extraction", tasks=len(tasks))
        clue_usages = await asyncio.gather(*tasks)
        for clue_usage in clue_usages:
            if clue_usage:
                for k in total_usage:
                    total_usage[k] += clue_usage.get(k, 0.0)

        logger.info(
            "Pass 3 multimodal complete",
            tasks_processed=len(tasks),
            total_cost_usd=round(total_usage.get("cost_usd", 0.0), 6),
            total_latency_ms=round(total_usage.get("latency_ms", 0), 0),
            total_thinking_tokens=int(total_usage.get("thinking_tokens", 0)),
        )
    else:
        logger.info("Pass 3: no multimodal tasks to process, skipping")

    return episode, total_usage
