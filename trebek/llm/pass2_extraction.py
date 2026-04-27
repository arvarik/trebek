"""
Pass 2: Map-Reduce structured data extraction from Jeopardy transcripts.

Orchestrates the full extraction pipeline:
1. Episode metadata extraction (contestants, FJ, score adjustments)
2. Semantic chunking by round boundaries
3. Semaphore-bounded concurrent clue extraction per chunk (Line-Indexed)
4. Timestamp reconstruction from parsed WhisperX JSON
5. Composite-key deduplication for overlapping chunk regions
6. Deterministic integrity validation encoding Jeopardy domain rules
"""

import json
import re
import structlog
import asyncio
from typing import Any, Dict, Literal, Union, Optional

from trebek.schemas import Episode, Clue, BuzzAttempt
from trebek.llm.schemas import PartialEpisodeMeta, PartialClues
from trebek.llm.chunking import _chunk_by_semantic_boundaries
from trebek.llm.validation import _validate_extraction_integrity, _deduplicate_clues
from trebek.llm.utils import _extract_part
from trebek.llm.transcript import _abbreviate_speaker, _format_transcript_compressed
from trebek.llm.speaker_normalization import _normalize_speaker_names
from trebek.config import MODEL_PRO

logger = structlog.get_logger()
GEMINI_CONCURRENCY = 3


def _count_syllables(text: str) -> int:
    """
    Vowel-cluster syllable counting heuristic.

    More accurate than word_count * constant for Jeopardy's academic vocabulary.
    Handles silent-e and guarantees a minimum of 1 syllable per word.
    """
    words = text.lower().split()
    total = 0
    for word in words:
        syllables = len(re.findall(r"[aeiouy]+", word))
        # Silent-e heuristic: trailing 'e' with >1 syllable
        if word.endswith("e") and syllables > 1:
            syllables -= 1
        total += max(1, syllables)
    return max(1, total)


async def execute_pass_2_data_extraction(
    segments: list[Dict[str, Any]],
    speaker_mapping: Dict[str, str],
    max_retries: int = 4,
    model: str = MODEL_PRO,
) -> "tuple[Episode, dict[str, float], int]":
    """
    Pass 2: Production-grade map-reduce extraction.

    Architecture:
    1. Extract Episode Meta (contestants, FJ, score adjustments) — full transcript
    2. Semantic chunking by round boundaries (not arbitrary line counts)
    3. Semaphore-bounded concurrent clue extraction per chunk (Line-Indexed)
    4. Reconstruct clue texts and timestamps directly from parsed WhisperX JSON
    5. Composite-key deduplication for overlapping chunk regions
    6. Deterministic integrity validation encoding Jeopardy domain rules

    Cost optimizations:
    - Prompt compression: timestamps stripped, speaker IDs abbreviated
    - Inline chunk extraction: each chunk sent independently to avoid
      the 2.5x cost penalty of context caching in map-reduce patterns
    """
    logger.info(
        "Starting Map-Reduce extraction pipeline",
        total_segments=len(segments),
        speaker_mapping=speaker_mapping,
        max_retries=max_retries,
        model=model,
    )

    # ── Prompt Compression ───────────────────────────────────────────
    # Build compressed speaker mapping for the system prompt
    compressed_mapping: Dict[str, str] = {}
    for speaker_id, name in speaker_mapping.items():
        compressed_mapping[_abbreviate_speaker(speaker_id)] = name

    full_transcript = _format_transcript_compressed(segments)
    logger.info(
        "Transcript formatted (compressed)",
        total_lines=len(segments),
        transcript_chars=len(full_transcript),
        compression="timestamps_removed, speakers_abbreviated",
    )

    base_system = (
        "You are Trebek, an expert data extraction pipeline for Jeopardy game transcripts. "
        "Extract game events into the provided JSON schema with surgical precision. "
        f"CRITICAL CONSTRAINT: Map speakers using this exact dictionary: {json.dumps(compressed_mapping)}. "
        "Speaker IDs are abbreviated (e.g., S0 = SPEAKER_00, S1 = SPEAKER_01). "
        "Do NOT hallucinate names outside this mapping. "
        "Do NOT perform any running score math — extract only observable facts. "
        "Line IDs (e.g., L0, L105) reference exact transcript positions. "
        "Use Line IDs for all timestamp-related fields."
    )

    total_usage: dict[str, float] = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "thinking_tokens": 0.0,
        "cached_tokens": 0.0,
        "total_tokens": 0.0,
        "cost_usd": 0.0,
        "latency_ms": 0.0,
    }
    max_attempt_reached = 0

    def _accumulate_usage(usage: dict[str, float]) -> None:
        for k in total_usage:
            total_usage[k] += usage.get(k, 0.0)

    # ── Stage 1: Meta Extraction (Full Transcript) ──────────────────
    logger.info("Extracting Episode Metadata...")
    meta_prompt = (
        f"Transcript Data:\n{full_transcript}\n\n"
        "Output strict JSON matching the requested schema. "
        "EXTRACT ONLY: episode_date, host_name, is_tournament, contestants, final_jeopardy, and score_adjustments. "
        "DO NOT extract individual clues."
    )
    meta_data, meta_usage, meta_att = await _extract_part(
        meta_prompt,
        base_system,
        PartialEpisodeMeta,
        max_retries,
        model=model,
        invocation_context="Pass 2 Meta",
        thinking_level="medium",  # Meta extraction needs reasoning for FJ, score adjustments, tournament detection
    )
    _accumulate_usage(meta_usage)
    max_attempt_reached = max(max_attempt_reached, meta_att)
    logger.info(
        "Episode metadata extracted",
        host=meta_data.host_name,
        contestants=[c.name for c in meta_data.contestants],
        is_tournament=meta_data.is_tournament,
        score_adjustments=len(meta_data.score_adjustments),
        meta_retries=meta_att,
    )

    # ── Stage 2: Semantic Chunking ──────────────────────────────────
    lines = full_transcript.split("\n")
    chunks = _chunk_by_semantic_boundaries(lines)
    logger.info(
        "Chunked transcript for concurrent clue extraction",
        num_chunks=len(chunks),
        strategy="semantic_round_boundaries",
        mode="inline_chunks",
    )

    # ── Stage 3: Semaphore-Bounded Concurrent Clue Extraction ──────
    semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)

    async def extract_chunk(chunk_idx: int, chunk_text: str) -> "tuple[PartialClues, dict[str, float], int]":
        async with semaphore:
            chunk_lines_list = chunk_text.split("\n")
            chunk_line_count = len(chunk_lines_list)

            # Extract Line ID range for debugging
            first_line_id = chunk_lines_list[0].split(" ")[0] if chunk_lines_list else "?"
            last_line_id = chunk_lines_list[-1].split(" ")[0] if chunk_lines_list else "?"

            ctx_label = f"Pass 2 Chunk {chunk_idx + 1}/{len(chunks)}"
            logger.info(
                f"Extracting clues from chunk {chunk_idx + 1}/{len(chunks)}",
                lines=chunk_line_count,
                line_range=f"{first_line_id}–{last_line_id}",
            )

            prompt = (
                f"Transcript Chunk ({chunk_idx + 1} of {len(chunks)}):\n{chunk_text}\n\n"
                "Extract ALL Jeopardy and Double Jeopardy clues found in this chunk. "
                "Do NOT extract Final Jeopardy — it is handled separately. "
                "If a clue appears cut off at the start or end of this chunk, SKIP IT completely "
                "(overlapping chunks will capture it whole). "
                "If no clues are found in this chunk, return an empty array for clues. "
                "Output ONLY Line IDs for clue reading and buzz attempts (e.g. 'L105'). "
                "Ensure you extract ALL buzz attempts."
            )
            return await _extract_part(
                prompt,
                base_system,
                PartialClues,
                max_retries,
                model=model,
                invocation_context=ctx_label,
            )

    chunk_tasks = [extract_chunk(idx, chunk) for idx, chunk in enumerate(chunks)]
    raw_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

    # Filter out failed chunks — log them but don't crash the entire episode
    chunk_results = []
    failed_chunks = 0
    for idx, result in enumerate(raw_results):
        if isinstance(result, BaseException):
            failed_chunks += 1
            logger.error(
                "Chunk extraction failed, skipping chunk",
                chunk=idx + 1,
                total_chunks=len(chunks),
                error_type=type(result).__name__,
                error=str(result)[:300],
            )
            continue
        chunk_data, _, _ = result
        logger.info(
            "Chunk extraction succeeded",
            chunk=idx + 1,
            clues_extracted=len(chunk_data.clues),
        )
        chunk_results.append(result)

    logger.info(
        "All chunk extractions complete",
        succeeded=len(chunk_results),
        failed=failed_chunks,
        total=len(chunks),
    )

    # ── Stage 4: Reconstruct & Deduplicate ──────────────────────────
    all_clues: list[Clue] = []
    dropped_clue_count = 0
    clamped_line_id_count = 0
    buzz_fallback_count = 0
    fj_filtered_count = 0
    nonstandard_line_id_count = 0

    for chunk_data, chunk_usage, chunk_att in chunk_results:
        for ext_clue in chunk_data.clues:
            # Filter out Final Jeopardy clues — the prompt says skip FJ but
            # the LLM sometimes extracts them anyway. FJ is handled by meta.
            if ext_clue.round == "Final Jeopardy":
                fj_filtered_count += 1
                continue

            # Resolve Line IDs — track non-standard formats
            raw_start_lid = ext_clue.host_read_start_line_id
            raw_end_lid = ext_clue.host_read_end_line_id
            start_id = raw_start_lid.replace("L", "").replace("[", "").replace("]", "").strip()
            end_id = raw_end_lid.replace("L", "").replace("[", "").replace("]", "").strip()

            # Track non-standard Line ID formats for prompt engineering feedback
            standard_pattern = re.compile(r"^L\d+$")
            if not standard_pattern.match(raw_start_lid.strip()) or not standard_pattern.match(raw_end_lid.strip()):
                nonstandard_line_id_count += 1

            try:
                s_idx = int(start_id)
                e_idx = int(end_id)
            except ValueError:
                dropped_clue_count += 1
                logger.warning(
                    "Invalid Line ID — dropping clue",
                    start=start_id,
                    end=end_id,
                    category=ext_clue.category,
                    round=ext_clue.round,
                    dropped_total=dropped_clue_count,
                )
                continue

            # Bounds check with logging
            s_clamped = max(0, min(s_idx, len(segments) - 1))
            e_clamped = max(0, min(e_idx, len(segments) - 1))
            if s_clamped != s_idx or e_clamped != e_idx:
                clamped_line_id_count += 1
                logger.warning(
                    "Line ID out of bounds — clamped",
                    original_start=s_idx,
                    original_end=e_idx,
                    clamped_start=s_clamped,
                    clamped_end=e_clamped,
                    max_segment=len(segments) - 1,
                    category=ext_clue.category,
                )
            s_idx = s_clamped
            e_idx = e_clamped
            if e_idx < s_idx:
                e_idx = s_idx

            clue_text = " ".join([seg.get("text", "").strip() for seg in segments[s_idx : e_idx + 1]])

            # Explicit None-check on segment timestamps — silent 0.0 default
            # would create phantom clues at t=0
            raw_start = segments[s_idx].get("start")
            raw_end = segments[e_idx].get("end")
            if raw_start is None:
                logger.warning(
                    "Segment missing 'start' timestamp, defaulting to 0.0",
                    segment_idx=s_idx,
                    category=ext_clue.category,
                )
                raw_start = 0.0
            if raw_end is None:
                logger.warning(
                    "Segment missing 'end' timestamp, defaulting to 0.0",
                    segment_idx=e_idx,
                    category=ext_clue.category,
                )
                raw_end = 0.0
            host_start_ms = float(raw_start) * 1000.0
            host_finish_ms = float(raw_end) * 1000.0

            # Map Buzz Attempts
            attempts = []
            for ext_att in ext_clue.attempts:
                buzz_id_str = ext_att.buzz_line_id.replace("L", "").replace("[", "").replace("]", "").strip()
                try:
                    b_idx = int(buzz_id_str)
                    b_idx = max(0, min(b_idx, len(segments) - 1))
                    raw_buzz_start = segments[b_idx].get("start")
                    if raw_buzz_start is None:
                        logger.warning(
                            "Buzz segment missing 'start' timestamp, defaulting to 0.0",
                            segment_idx=b_idx,
                            speaker=ext_att.speaker,
                            category=ext_clue.category,
                        )
                        raw_buzz_start = 0.0
                    buzz_timestamp_ms = float(raw_buzz_start) * 1000.0
                except ValueError:
                    buzz_fallback_count += 1
                    logger.warning(
                        "Invalid buzz Line ID — falling back to host_finish",
                        buzz_line_id=ext_att.buzz_line_id,
                        speaker=ext_att.speaker,
                        category=ext_clue.category,
                        fallback_total=buzz_fallback_count,
                    )
                    buzz_timestamp_ms = host_finish_ms  # fallback

                # Account for lockout penalty in response start offset
                lockout_penalty = 250.0 if ext_att.is_lockout_inferred else 0.0

                attempts.append(
                    BuzzAttempt(
                        attempt_order=ext_att.attempt_order,
                        speaker=ext_att.speaker,
                        response_given=ext_att.response_given,
                        is_correct=ext_att.is_correct,
                        buzz_timestamp_ms=buzz_timestamp_ms,
                        response_start_timestamp_ms=buzz_timestamp_ms + 250.0 + lockout_penalty,
                        is_lockout_inferred=ext_att.is_lockout_inferred,
                    )
                )

            wager_val: Optional[Union[int, Literal["True Daily Double"]]] = None
            if ext_clue.daily_double_wager is not None:
                if ext_clue.daily_double_wager == "True Daily Double":
                    wager_val = "True Daily Double"
                else:
                    try:
                        wager_val = int(ext_clue.daily_double_wager)
                    except ValueError:
                        logger.warning(
                            "Invalid DD wager value — dropping",
                            raw_wager=ext_clue.daily_double_wager,
                            category=ext_clue.category,
                        )

            syllable_estimate = _count_syllables(clue_text)

            all_clues.append(
                Clue(
                    round=ext_clue.round,
                    category=ext_clue.category,
                    board_row=ext_clue.board_row,
                    board_col=ext_clue.board_col,
                    selection_order=0,  # Computed later via chronological sort
                    is_daily_double=ext_clue.is_daily_double,
                    requires_visual_context=ext_clue.requires_visual_context,
                    host_start_timestamp_ms=host_start_ms,
                    host_finish_timestamp_ms=host_finish_ms,
                    clue_syllable_count=syllable_estimate,
                    daily_double_wager=wager_val,
                    wagerer_name=ext_clue.wagerer_name,
                    clue_text=clue_text,
                    correct_response=ext_clue.correct_response,
                    attempts=attempts,
                )
            )

        _accumulate_usage(chunk_usage)
        max_attempt_reached = max(max_attempt_reached, chunk_att)

    # ── Assembly quality summary ─────────────────────────────────────
    logger.info(
        "Clue assembly quality summary",
        clues_assembled=len(all_clues),
        clues_dropped_invalid_line_id=dropped_clue_count,
        buzz_fallback_to_host_finish=buzz_fallback_count,
        line_ids_clamped=clamped_line_id_count,
        fj_clues_filtered=fj_filtered_count,
        nonstandard_line_id_format=nonstandard_line_id_count,
    )
    if nonstandard_line_id_count > 0:
        logger.warning(
            "Non-standard Line ID formats detected — consider prompt adjustment",
            count=nonstandard_line_id_count,
            total_clues=len(all_clues),
            pct=round(nonstandard_line_id_count / max(1, len(all_clues)) * 100, 1),
        )

    logger.info("Raw clues before deduplication", count=len(all_clues))
    unique_clues = _deduplicate_clues(all_clues)

    # Sort chronologically and re-index selection_order
    sorted_clues = sorted(unique_clues, key=lambda c: c.host_start_timestamp_ms)
    for i, clue in enumerate(sorted_clues):
        clue.selection_order = i + 1

    duplicates_removed = len(all_clues) - len(sorted_clues)
    logger.info(
        "Clues after deduplication",
        count=len(sorted_clues),
        duplicates_removed=duplicates_removed,
    )

    # ── Stage 4b: Speaker Name Normalization ────────────────────────
    contestant_names = [c.name for c in meta_data.contestants]
    _normalize_speaker_names(
        sorted_clues,
        speaker_mapping,
        contestant_names,
        host_name=meta_data.host_name,
        score_adjustments=meta_data.score_adjustments,
        fj_wagers=meta_data.final_jeopardy.wagers_and_responses,
    )

    # ── Stage 5: Assemble Episode ───────────────────────────────────
    episode = Episode(
        episode_date=meta_data.episode_date,
        host_name=meta_data.host_name,
        is_tournament=meta_data.is_tournament,
        contestants=meta_data.contestants,
        clues=sorted_clues,
        final_jeopardy=meta_data.final_jeopardy,
        score_adjustments=meta_data.score_adjustments,
    )

    # ── Stage 6: Deterministic Integrity Validation ─────────────────
    integrity_warnings = _validate_extraction_integrity(episode)
    if integrity_warnings:
        for w in integrity_warnings:
            logger.warning("Extraction integrity check", issue=w)
        logger.warning(
            "Extraction completed with integrity warnings",
            warning_count=len(integrity_warnings),
            total_clues=len(sorted_clues),
        )
    else:
        logger.info(
            "Extraction passed all integrity checks",
            total_clues=len(sorted_clues),
        )

    logger.info(
        "Map-Reduce extraction complete",
        total_clues_extracted=len(sorted_clues),
        chunks_processed=len(chunks),
        max_retries_used=max_attempt_reached,
        model=model,
    )

    # ── Episode Quality Score ────────────────────────────────────────
    quality_score = "PASS" if not integrity_warnings else ("DEGRADED" if len(integrity_warnings) <= 3 else "FAIL")
    logger.info(
        "Episode quality assessment",
        quality=quality_score,
        warning_count=len(integrity_warnings),
        clue_count=len(sorted_clues),
        contestant_count=len(meta_data.contestants),
        dropped_clues=dropped_clue_count,
        clamped_line_ids=clamped_line_id_count,
        buzz_fallbacks=buzz_fallback_count,
    )

    return episode, total_usage, max_attempt_reached
