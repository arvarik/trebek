"""
Pass 2: Map-Reduce structured data extraction from J! transcripts.

Orchestrates the full extraction pipeline:
1. Episode metadata extraction (contestants, FJ, score adjustments)
2. Semantic chunking by round boundaries
3. Semaphore-bounded concurrent clue extraction per chunk (Line-Indexed)
4. Timestamp reconstruction from parsed WhisperX JSON
5. Composite-key deduplication for overlapping chunk regions
6. Deterministic integrity validation encoding J! domain rules
"""

import json
import re
import structlog
import asyncio
from typing import Any, Dict, Literal, Union, Optional

from trebek.schemas import Episode, Clue, BuzzAttempt
from trebek.llm.schemas import PartialEpisodeMeta
from trebek.llm.chunking import _chunk_by_semantic_boundaries
from trebek.llm.validation import _validate_extraction_integrity, _deduplicate_clues
from trebek.llm.utils import _extract_part
from trebek.llm.transcript import _abbreviate_speaker, _format_transcript_compressed
from trebek.llm.speaker_normalization import (
    _normalize_speaker_names,
    _reconcile_speaker_mapping,
    _resolve_host_from_pass1,
)
from trebek.config import MODEL_PRO, KNOWN_HOSTS

logger = structlog.get_logger()
GEMINI_CONCURRENCY = 3


def _count_syllables(text: str) -> int:
    """
    Vowel-cluster syllable counting heuristic.

    More accurate than word_count * constant for J!'s academic vocabulary.
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
) -> "tuple[Episode, dict[str, float], int, str]":
    """
    Pass 2: Production-grade map-reduce extraction.

    Architecture:
    1. Extract Episode Meta (contestants, FJ, score adjustments) — full transcript
    2. Speaker mapping reconciliation (Pass 1 → Pass 2 name alignment)
    3. Semantic chunking by round boundaries
    4. Semaphore-bounded concurrent clue extraction per chunk (Line-Indexed)
    5. Reconstruct clue texts and timestamps directly from parsed WhisperX JSON
    6. Composite-key deduplication for overlapping chunk regions
    7. Deterministic integrity validation encoding J! domain rules

    Returns:
        Tuple of (Episode, usage_dict, max_retries_used, quality_score).
        quality_score is one of "PASS", "DEGRADED", or "FAIL".

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

    full_transcript = _format_transcript_compressed(segments)
    logger.info(
        "Transcript formatted (compressed)",
        total_lines=len(segments),
        transcript_chars=len(full_transcript),
        compression="timestamps_removed, speakers_abbreviated",
    )

    # Build initial compressed mapping for meta extraction (pre-reconciliation).
    # This will be rebuilt after meta extraction with reconciled names.
    compressed_mapping: Dict[str, str] = {}
    for speaker_id, name in speaker_mapping.items():
        compressed_mapping[_abbreviate_speaker(speaker_id)] = name

    base_system = (
        "You are Trebek, an expert data extraction pipeline for J! game transcripts. "
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
        "EXTRACT ONLY: episode_date, host_name, is_tournament, contestants, jeopardy_categories, double_jep_categories, final_jep, and score_adjustments. "
        "CRITICAL: For each contestant, extract their FULL NAME (first AND last name) as introduced by the host during the interview segment. "
        "Do NOT use first names only. "
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

    # Warn on single-word contestant names (likely missing last names)
    for c in meta_data.contestants:
        if len(c.name.split()) < 2:
            logger.warning(
                "Contestant has single-word name (possible missing last name)",
                contestant=c.name,
            )

    # ── Stage 1b: Host Validation ───────────────────────────────────
    # The LLM sometimes misidentifies a contestant as the host (e.g.,
    # "Lisa" instead of "Ken Jennings"). Validate against KNOWN_HOSTS
    # and fall back to Pass 1 mapping or the default modern host.
    known_hosts_lower = {h.lower() for h in KNOWN_HOSTS}
    extracted_host = meta_data.host_name.strip()
    if extracted_host.lower() not in known_hosts_lower:
        contestant_names_lower = {c.name.lower() for c in meta_data.contestants}
        is_contestant_confusion = extracted_host.lower() in contestant_names_lower
        pass1_host = _resolve_host_from_pass1(speaker_mapping)
        corrected_host = pass1_host or "Ken Jennings"
        logger.error(
            "Host misidentified — overriding",
            extracted_host=extracted_host,
            is_contestant_confusion=is_contestant_confusion,
            corrected_host=corrected_host,
            source="pass1" if pass1_host else "default",
        )
        meta_data.host_name = corrected_host

    # ── Stage 1c: Speaker Mapping Reconciliation ────────────────────
    # Pass 1 (audio anchoring) may spell names differently than Pass 2
    # (transcript analysis). e.g., "Paulo" vs "Paolo Pasco".
    # Reconcile before building the clue extraction prompt to ensure
    # the system prompt and schema constraints are self-consistent.
    contestant_names = [c.name for c in meta_data.contestants]
    reconciled_mapping = _reconcile_speaker_mapping(speaker_mapping, contestant_names, host_name=meta_data.host_name)
    if reconciled_mapping != speaker_mapping:
        logger.info(
            "Speaker mapping reconciled (Pass 1 → Pass 2 alignment)",
            original=speaker_mapping,
            reconciled=reconciled_mapping,
        )

    # Rebuild compressed mapping with reconciled names for clue extraction.
    # Filter out speakers that are not contestants, not the host, and not
    # resolvable — these are typically commercial/news voices (e.g.,
    # "Lindsay Davis", "Aaron Katersky") that waste prompt tokens.
    valid_speakers = {meta_data.host_name.lower()} | {c.name.lower() for c in meta_data.contestants}
    compressed_mapping = {}
    for speaker_id, name in reconciled_mapping.items():
        if name.lower() in valid_speakers:
            compressed_mapping[_abbreviate_speaker(speaker_id)] = name
        else:
            logger.info(
                "Filtering unresolvable speaker from clue extraction prompt",
                speaker_id=speaker_id,
                name=name,
            )
    base_system = (
        "You are Trebek, an expert data extraction pipeline for J! game transcripts. "
        "Extract game events into the provided JSON schema with surgical precision. "
        f"CRITICAL CONSTRAINT: Map speakers using this exact dictionary: {json.dumps(compressed_mapping)}. "
        "Speaker IDs are abbreviated (e.g., S0 = SPEAKER_00, S1 = SPEAKER_01). "
        "Do NOT hallucinate names outside this mapping. "
        "Do NOT perform any running score math — extract only observable facts. "
        "Line IDs (e.g., L0, L105) reference exact transcript positions. "
        "Use Line IDs for all timestamp-related fields."
    )

    # ── Stage 2: Semantic Chunking ──────────────────────────────────
    from trebek.llm.schemas import create_dynamic_clue_schema

    all_categories = meta_data.jeopardy_categories + meta_data.double_jep_categories
    DynamicPartialClues = create_dynamic_clue_schema(all_categories, contestant_names)

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

    async def extract_chunk(
        chunk_idx: int, chunk_text: str, is_retry: bool = False
    ) -> "tuple[Any, dict[str, float], int]":
        async with semaphore:
            chunk_lines_list = chunk_text.split("\n")
            chunk_line_count = len(chunk_lines_list)

            # Extract Line ID range for debugging
            first_line_id = chunk_lines_list[0].split(" ")[0] if chunk_lines_list else "?"
            last_line_id = chunk_lines_list[-1].split(" ")[0] if chunk_lines_list else "?"

            ctx_label = f"Pass 2 Chunk {chunk_idx + 1}/{len(chunks)}" + (" (Retry)" if is_retry else "")
            logger.info(
                f"Extracting clues from chunk {chunk_idx + 1}/{len(chunks)}",
                lines=chunk_line_count,
                line_range=f"{first_line_id}–{last_line_id}",
                is_retry=is_retry,
            )

            prompt = (
                f"Transcript Chunk ({chunk_idx + 1} of {len(chunks)}):\n{chunk_text}\n\n"
                "Extract ALL J! and Double J! clues found in this chunk. "
                "Do NOT extract Final J! — it is handled separately. "
                "Skip clues cut off at chunk boundaries. "
                "Use Line IDs for timestamps (e.g. 'L105')."
            )
            return await _extract_part(
                prompt,
                base_system,
                DynamicPartialClues,
                max_retries,
                model=model,
                invocation_context=ctx_label,
                thinking_level="low",  # Clue extraction is mechanical, not reasoning-heavy
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
        chunk_results.append((result, idx))

    # ── Auto-retry under-extracted chunks ──────────────────────────
    # Dynamic threshold: expect ~1 clue per 8-10 transcript lines.
    # A 480-line transcript should yield ~50-60 clues; the old static
    # threshold of 15 missed S42E04 which produced 45 clues (> 15)
    # but was still catastrophically under-extracted (15/30 J! clues).
    MIN_CHUNK_LINES_FOR_RETRY = 50
    retried_chunks = 0

    final_chunk_results = []
    for result_tuple, original_idx in chunk_results:
        chunk_data, chunk_usage, chunk_att = result_tuple
        chunk_text = chunks[original_idx]
        chunk_lines_list = chunk_text.split("\n")
        chunk_line_count = len(chunk_lines_list)

        # Dynamic threshold: scale with chunk size, minimum 15
        expected_clues = max(15, chunk_line_count // 10)

        if len(chunk_data.clues) < expected_clues and chunk_line_count > MIN_CHUNK_LINES_FOR_RETRY:
            retried_chunks += 1

            # ── Round-aware retry targeting ──────────────────────
            # Determine which round is under-extracted. J! clues
            # live in the first half, DJ! in the second half. If J!
            # is severely under-represented, retrying the tail is
            # useless — we need to re-extract the first half.
            j_count = sum(1 for c in chunk_data.clues if c.round == "J!")
            dj_count = sum(1 for c in chunk_data.clues if c.round == "Double J!")
            j_deficit = max(0, 25 - j_count)
            dj_deficit = max(0, 25 - dj_count)

            if j_deficit > dj_deficit and j_deficit >= 10:
                # J! round is the bottleneck — retry the FIRST half
                retry_region = "first_half"
                retry_text = "\n".join(chunk_lines_list[: chunk_line_count // 2])
            elif dj_deficit > j_deficit and dj_deficit >= 10:
                # DJ! round is the bottleneck — retry the SECOND half
                retry_region = "second_half"
                retry_text = "\n".join(chunk_lines_list[chunk_line_count // 2 :])
            else:
                # Balanced deficit — use the old heuristic (retry from
                # the last extracted line or the second half)
                retry_region = "tail"
                last_extracted_line_id = None
                if chunk_data.clues:
                    try:
                        sorted_clues = sorted(
                            chunk_data.clues,
                            key=lambda c: int(
                                c.host_read_start_line_id.replace("L", "").replace("[", "").replace("]", "").strip()
                            ),
                        )
                        last_extracted_line_id = sorted_clues[-1].host_read_end_line_id
                    except ValueError:
                        pass

                if not last_extracted_line_id:
                    retry_text = "\n".join(chunk_lines_list[chunk_line_count // 2 :])
                else:
                    split_idx = chunk_line_count // 2
                    for i, line in enumerate(chunk_lines_list):
                        if line.startswith(last_extracted_line_id):
                            split_idx = min(i + 1, chunk_line_count - 1)
                            break
                    retry_text = "\n".join(chunk_lines_list[split_idx:])

            logger.warning(
                "Chunk under-extracted, triggering round-aware retry",
                chunk=original_idx + 1,
                clues_found=len(chunk_data.clues),
                j_clues=j_count,
                dj_clues=dj_count,
                retry_region=retry_region,
                chunk_lines=chunk_line_count,
            )

            try:
                retry_result = await extract_chunk(original_idx, retry_text, is_retry=True)
                retry_data, retry_usage, retry_att = retry_result
                logger.info(
                    "Chunk retry result",
                    chunk=original_idx + 1,
                    original_clues=len(chunk_data.clues),
                    additional_clues=len(retry_data.clues),
                )
                # Combine both results
                chunk_data.clues.extend(retry_data.clues)
                final_chunk_results.append((chunk_data, chunk_usage, max(chunk_att, retry_att)))
            except Exception as retry_err:
                logger.warning(
                    "Chunk retry failed, keeping original",
                    chunk=original_idx + 1,
                    error=str(retry_err)[:200],
                )
                final_chunk_results.append(result_tuple)
        else:
            final_chunk_results.append(result_tuple)

    if retried_chunks > 0:
        logger.info(
            "Under-extraction retry pass complete",
            retried_chunks=retried_chunks,
            total_chunks=len(chunks),
        )

    logger.info(
        "All chunk extractions complete",
        succeeded=len(final_chunk_results),
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

    for chunk_data, chunk_usage, chunk_att in final_chunk_results:
        for ext_clue in chunk_data.clues:
            # Filter out Final J! clues — the prompt says skip FJ but
            # the LLM sometimes extracts them anyway. FJ is handled by meta.
            if ext_clue.round == "Final J!":
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

            # Clamp hallucinated board positions to valid J! grid ranges.
            # The LLM can't see the physical board — these fields are always
            # inferred from category order, so out-of-range values are common.
            clamped_row = max(1, min(ext_clue.board_row, 5))
            clamped_col = max(1, min(ext_clue.board_col, 6))

            all_clues.append(
                Clue(
                    round=ext_clue.round,
                    category=ext_clue.category,
                    board_row=clamped_row,
                    board_col=clamped_col,
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
    # Use reconciled mapping (not raw Pass 1 mapping) for normalization
    _normalize_speaker_names(
        sorted_clues,
        reconciled_mapping,
        contestant_names,
        host_name=meta_data.host_name,
        score_adjustments=meta_data.score_adjustments,
        fj_wagers=meta_data.final_jep.wagers_and_responses,
    )

    # ── Stage 5: Assemble Episode ───────────────────────────────────
    episode = Episode(
        episode_date=meta_data.episode_date,
        host_name=meta_data.host_name,
        is_tournament=meta_data.is_tournament,
        contestants=meta_data.contestants,
        clues=sorted_clues,
        final_jep=meta_data.final_jep,
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

    # Hard fail: severe under-extraction (standard episode = 60 clues)
    if len(sorted_clues) < 45:
        quality_score = "FAIL"
        logger.error(
            "Severe under-extraction detected",
            clue_count=len(sorted_clues),
            minimum_expected=45,
        )

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

    return episode, total_usage, max_attempt_reached, quality_score
