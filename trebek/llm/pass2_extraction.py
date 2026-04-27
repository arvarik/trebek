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

    total_usage: dict[str, float] = {"input_tokens": 0.0, "output_tokens": 0.0, "cached_tokens": 0.0, "latency_ms": 0.0}
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

            ctx_label = f"Pass 2 Chunk {chunk_idx + 1}/{len(chunks)}"
            logger.info(f"Extracting clues from chunk {chunk_idx + 1}/{len(chunks)}", lines=chunk_line_count)

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
    for chunk_data, chunk_usage, chunk_att in chunk_results:
        for ext_clue in chunk_data.clues:
            # Resolve Line IDs
            start_id = ext_clue.host_read_start_line_id.replace("L", "").replace("[", "").replace("]", "").strip()
            end_id = ext_clue.host_read_end_line_id.replace("L", "").replace("[", "").replace("]", "").strip()

            try:
                s_idx = int(start_id)
                e_idx = int(end_id)
            except ValueError:
                logger.warning("Invalid Line ID extracted", start=start_id, end=end_id)
                continue

            # Bounds check
            s_idx = max(0, min(s_idx, len(segments) - 1))
            e_idx = max(0, min(e_idx, len(segments) - 1))
            if e_idx < s_idx:
                e_idx = s_idx

            clue_text = " ".join([seg.get("text", "").strip() for seg in segments[s_idx : e_idx + 1]])
            host_start_ms = float(segments[s_idx].get("start", 0.0) * 1000)
            host_finish_ms = float(segments[e_idx].get("end", 0.0) * 1000)

            # Map Buzz Attempts
            attempts = []
            for ext_att in ext_clue.attempts:
                buzz_id_str = ext_att.buzz_line_id.replace("L", "").replace("[", "").replace("]", "").strip()
                try:
                    b_idx = int(buzz_id_str)
                    b_idx = max(0, min(b_idx, len(segments) - 1))
                    buzz_timestamp_ms = float(segments[b_idx].get("start", 0.0) * 1000)
                except ValueError:
                    buzz_timestamp_ms = host_finish_ms  # fallback

                attempts.append(
                    BuzzAttempt(
                        attempt_order=ext_att.attempt_order,
                        speaker=ext_att.speaker,
                        response_given=ext_att.response_given,
                        is_correct=ext_att.is_correct,
                        buzz_timestamp_ms=buzz_timestamp_ms,
                        response_start_timestamp_ms=buzz_timestamp_ms + 250.0,  # Approximate
                        is_lockout_inferred=ext_att.is_lockout_inferred,
                    )
                )

            wager_val: Optional[Union[int, Literal["True Daily Double"]]] = None
            if ext_clue.daily_double_wager == "True Daily Double":
                wager_val = "True Daily Double"
            elif isinstance(ext_clue.daily_double_wager, int):
                wager_val = ext_clue.daily_double_wager
            elif isinstance(ext_clue.daily_double_wager, str) and ext_clue.daily_double_wager.isdigit():
                wager_val = int(ext_clue.daily_double_wager)

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
                    clue_syllable_count=len(clue_text.split()) * 2,  # Approximate
                    daily_double_wager=wager_val,
                    wagerer_name=ext_clue.wagerer_name,
                    clue_text=clue_text,
                    correct_response=ext_clue.correct_response,
                    attempts=attempts,
                )
            )

        _accumulate_usage(chunk_usage)
        max_attempt_reached = max(max_attempt_reached, chunk_att)

    logger.info("Raw clues before deduplication", count=len(all_clues))
    unique_clues = _deduplicate_clues(all_clues)

    # Sort chronologically and re-index selection_order
    sorted_clues = sorted(unique_clues, key=lambda c: c.host_start_timestamp_ms)
    for i, clue in enumerate(sorted_clues):
        clue.selection_order = i + 1

    logger.info("Clues after deduplication", count=len(sorted_clues))

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
    return episode, total_usage, max_attempt_reached
