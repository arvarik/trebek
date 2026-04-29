"""
Pass 2: Manifest-Verify-Fill structured data extraction from J! transcripts.

Orchestrates the full extraction pipeline:
1. Episode metadata extraction (contestants, FJ, score adjustments)
2. Board manifest construction (category lists, value grids)
3. Round-split concurrent clue extraction (J! and DJ! independently)
4. Category gap detection and targeted re-extraction
5. Timestamp reconstruction from parsed WhisperX JSON
6. Board row inference from clue dollar values in transcript
7. Composite-key deduplication for overlapping regions
8. Deterministic integrity validation encoding J! domain rules
"""

import json
import re
import structlog
import asyncio
from typing import Any, Dict, Literal, Union, Optional

from trebek.schemas import Episode, Clue, BuzzAttempt
from trebek.llm.schemas import PartialEpisodeMeta
from trebek.llm.validation import _validate_extraction_integrity, _deduplicate_clues, normalize_response_format
from trebek.llm.utils import _extract_part
from trebek.llm.transcript import _abbreviate_speaker, _format_transcript_compressed
from trebek.llm.speaker_normalization import (
    _normalize_speaker_names,
    _reconcile_speaker_mapping,
    _resolve_host_from_pass1,
)
from trebek.config import MODEL_PRO, MODEL_FLASH, KNOWN_HOSTS

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


# ── Regex for stripping selection preamble from assembled clue text ──
# Matches patterns like:
#   "Numeric words and phrases for 600. On CB radio..."
#   "I can adapt for $200. This semi-aquatic..."
#   "Tree Pre for 800. Ars El-Rab..."
#   "Let's stick with trees for 600. Andrew Jackson's..."
#   "I can adapt, 600. I can adapt, sort of..."
# Group 1 captures everything up to and including the preamble separator.
_PREAMBLE_PATTERN = re.compile(
    r"^(?:"
    # Pattern A: "Category for [$]value[.]" (with optional period)
    r"(?:.*?)\bfor\s+\$?[\d,]+\.?\s+"
    r"|"
    # Pattern B: "Let's stick with CATEGORY for [$]value[.]"
    r"(?:Let'?s\s+(?:stick\s+with|go\s+(?:with|to|back\s+to)|try|do)\s+.*?)\bfor\s+\$?[\d,]+\.?\s+"
    r"|"
    # Pattern C: "One more time, we have CATEGORY[.]" (verbose announcements)
    r"(?:One\s+more\s+time,?\s+we\s+have\s+.*?\.)\s+"
    r")",
    re.IGNORECASE,
)


def _strip_selection_preamble(clue_text: str, category: str) -> str:
    """Strip category+value selection preamble from the start of clue text.

    When the LLM's host_read_start_line_id points to the segment where
    the host announces the category selection (e.g., "I can adapt for 400.
    It's armored up..."), the assembled clue_text starts with this preamble.

    This function detects and removes it, returning only the actual clue content.
    Returns the original text if no preamble is detected.
    """
    if not clue_text:
        return clue_text

    # Quick check: does the clue start with the category name (case-insensitive)?
    # If not, there's likely no preamble to strip.
    cat_lower = category.lower().strip()
    clue_lower = clue_text.lower().strip()

    # Check for category-name-prefixed text or "Let's" preambles
    has_preamble = (
        clue_lower.startswith(cat_lower[:10])
        or clue_lower.startswith("let's ")
        or clue_lower.startswith("let's ")
        or clue_lower.startswith("one more time")
    )
    if not has_preamble:
        return clue_text

    match = _PREAMBLE_PATTERN.match(clue_text)
    if match:
        stripped = clue_text[match.end() :]
        # Safety: don't strip if the result is too short (< 10 chars)
        # — that means the "preamble" was actually the entire clue
        if len(stripped.strip()) >= 10:
            return stripped.strip()

    return clue_text


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
        "RESPONSE FORMAT: Every correct_response MUST be in J! question form "
        "(e.g., 'What is Paris?', 'Who is Shakespeare?'). Never output a bare answer word. "
        "Line IDs (e.g., L0, L105) reference exact transcript positions. "
        "Use Line IDs for all timestamp-related fields."
    )

    # ── Stage 2: Manifest-Verify-Fill Extraction ─────────────────────
    # Instead of extracting all 60 clues in one massive call, we:
    # 1. Split transcript into J! and DJ! regions
    # 2. Extract each round independently (concurrent, category-aware)
    # 3. Detect category-level gaps (expected 5 per category)
    # 4. Surgically re-extract only the missing clues

    from trebek.llm.schemas import create_dynamic_clue_schema
    from trebek.llm.chunking import split_transcript_by_round
    from trebek.llm.board import (
        detect_board_format,
        build_manifests,
        RoundManifest,
    )

    board_format = detect_board_format(
        is_tournament=meta_data.is_tournament,
        j_categories=meta_data.jeopardy_categories,
        dj_categories=meta_data.double_jep_categories,
        transcript_text=full_transcript,
    )

    # ── Stage 2a: Category Discovery Fallback ────────────────────────
    # Standard J! boards have 6 categories per round. If the meta
    # extraction found fewer (often because the category intro was cut
    # off by a news broadcast or commercial), we use a cheap Flash call
    # to rediscover missing categories from the gameplay in the transcript.

    EXPECTED_CATEGORIES = 6  # Standard J! board

    async def _discover_missing_categories(
        round_name: str,
        known_categories: list[str],
        transcript_text: str,
    ) -> list[str]:
        """Use Flash to find categories the meta extraction missed."""
        if len(known_categories) >= EXPECTED_CATEGORIES:
            return known_categories

        missing_count = EXPECTED_CATEGORIES - len(known_categories)
        known_list = ", ".join(f'"{c}"' for c in known_categories)
        discovery_prompt = (
            f"Transcript ({round_name} round):\n{transcript_text}\n\n"
            f"I already know these {len(known_categories)} categories for the {round_name} round: [{known_list}].\n"
            f"A standard J! board has {EXPECTED_CATEGORIES} categories. "
            f"I'm missing {missing_count} category name(s).\n"
            f"Look through the transcript for category selections by contestants "
            f"(e.g., 'Cards for 800', 'Geography for 200') to find the missing category names.\n"
            f"Return ONLY the complete list of ALL {EXPECTED_CATEGORIES} categories for this round "
            f"(including the ones I already know)."
        )

        from trebek.llm.schemas import EpisodeSkeleton

        try:
            skeleton, skel_usage, _ = await _extract_part(
                discovery_prompt,
                "You are a J! category name extractor. Return the complete list of category names.",
                EpisodeSkeleton,
                max_retries=2,
                model=MODEL_FLASH,
                invocation_context=f"Category Discovery {round_name}",
                thinking_level="low",
            )
            _accumulate_usage(skel_usage)

            # Use the round-appropriate category list
            if round_name == "J!":
                discovered = skeleton.jeopardy_categories
            else:
                discovered = skeleton.double_jep_categories

            if len(discovered) > len(known_categories):
                logger.info(
                    "Category discovery found missing categories",
                    round=round_name,
                    previously_known=len(known_categories),
                    now_known=len(discovered),
                    new_categories=[c for c in discovered if c not in known_categories],
                )
                return discovered
            else:
                logger.warning(
                    "Category discovery did not find additional categories",
                    round=round_name,
                    known=len(known_categories),
                    discovered=len(discovered),
                )
                return known_categories
        except Exception as e:
            logger.warning(
                "Category discovery failed — using original categories",
                round=round_name,
                error=str(e)[:200],
            )
            return known_categories

    # Run category discovery for under-populated rounds
    j_cats = meta_data.jeopardy_categories
    dj_cats = meta_data.double_jep_categories

    if len(j_cats) < EXPECTED_CATEGORIES or len(dj_cats) < EXPECTED_CATEGORIES:
        logger.warning(
            "Under-populated category manifest — running category discovery fallback",
            j_categories=len(j_cats),
            dj_categories=len(dj_cats),
            expected=EXPECTED_CATEGORIES,
        )
        # Run discovery calls concurrently for both rounds if needed
        discovery_tasks = []
        if len(j_cats) < EXPECTED_CATEGORIES:
            discovery_tasks.append(("j", _discover_missing_categories("J!", j_cats, full_transcript)))
        if len(dj_cats) < EXPECTED_CATEGORIES:
            discovery_tasks.append(("dj", _discover_missing_categories("Double J!", dj_cats, full_transcript)))

        for label, task in discovery_tasks:
            result = await task
            if label == "j":
                j_cats = result
                meta_data.jeopardy_categories = j_cats
            else:
                dj_cats = result
                meta_data.double_jep_categories = dj_cats

    j_manifest, dj_manifest = build_manifests(
        j_categories=j_cats,
        dj_categories=dj_cats,
        board_format=board_format,
    )

    logger.info(
        "Board manifest built",
        format=board_format.name,
        j_categories=len(j_manifest.categories),
        dj_categories=len(dj_manifest.categories),
        expected_j_clues=j_manifest.expected_total,
        expected_dj_clues=dj_manifest.expected_total,
    )

    lines = full_transcript.split("\n")
    j_text, dj_text, _fj_text = split_transcript_by_round(lines)

    # Build round-scoped dynamic schemas for category validation
    j_schema = create_dynamic_clue_schema(meta_data.jeopardy_categories, contestant_names)
    dj_schema = create_dynamic_clue_schema(meta_data.double_jep_categories, contestant_names)

    semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)

    async def _extract_round(
        round_name: str,
        transcript_text: str,
        categories: list[str],
        schema_cls: Any,
        manifest: RoundManifest,
        is_gap_fill: bool = False,
        gap_context: str = "",
    ) -> "tuple[Any, dict[str, float], int]":
        """Extract clues for a single round with category-manifest awareness."""
        async with semaphore:
            if is_gap_fill:
                prompt = (
                    f"Transcript (targeted excerpt for {round_name}):\n{transcript_text}\n\n"
                    f"You are filling GAPS in a previous extraction. {gap_context}\n"
                    f"Extract ONLY the missing clues listed above. "
                    f"Use Line IDs for timestamps (e.g. 'L105')."
                )
                ctx_label = f"Pass 2 Gap-Fill {round_name}"
            else:
                cat_list = ", ".join(f'"{c}"' for c in categories)
                prompt = (
                    f"Transcript ({round_name} round):\n{transcript_text}\n\n"
                    f"Extract ALL {round_name} clues from this transcript section. "
                    f"The {round_name} board has these {len(categories)} categories: [{cat_list}]. "
                    f"Each category has exactly {manifest.clues_per_category} clues (rows 1-{manifest.clues_per_category}). "
                    f"You should find approximately {manifest.expected_total} clues total. "
                    f"Do NOT extract clues from other rounds (Double J!, Final J!, etc.). "
                    f"Use Line IDs for timestamps (e.g. 'L105')."
                )
                ctx_label = f"Pass 2 {round_name}"

            logger.info(
                f"Extracting {round_name} clues",
                categories=len(categories),
                transcript_chars=len(transcript_text),
                is_gap_fill=is_gap_fill,
            )

            return await _extract_part(
                prompt,
                base_system,
                schema_cls,
                max_retries,
                model=model,
                invocation_context=ctx_label,
                thinking_level="low",
            )

    # ── Stage 2b: Concurrent Round Extraction ───────────────────────
    # Extract J! and DJ! in parallel — same total latency as single call
    if dj_text:
        j_task = _extract_round("J!", j_text, meta_data.jeopardy_categories, j_schema, j_manifest)
        dj_task = _extract_round("Double J!", dj_text, meta_data.double_jep_categories, dj_schema, dj_manifest)
        (j_result, dj_result) = await asyncio.gather(j_task, dj_task)
    else:
        # Fallback: no DJ! boundary found — extract everything in one call
        logger.warning("No DJ! boundary found, falling back to full-transcript extraction")
        all_schema = create_dynamic_clue_schema(
            meta_data.jeopardy_categories + meta_data.double_jep_categories, contestant_names
        )
        j_result = await _extract_round(
            "J! + Double J!",
            j_text,
            meta_data.jeopardy_categories + meta_data.double_jep_categories,
            all_schema,
            j_manifest,
        )
        dj_result = None

    j_data, j_usage, j_att = j_result
    _accumulate_usage(j_usage)
    max_attempt_reached = max(max_attempt_reached, j_att)

    if dj_result:
        dj_data, dj_usage, dj_att = dj_result
        _accumulate_usage(dj_usage)
        max_attempt_reached = max(max_attempt_reached, dj_att)
        extracted_clues = list(j_data.clues) + list(dj_data.clues)
    else:
        extracted_clues = list(j_data.clues)

    logger.info(
        "Round extraction complete",
        j_clues=len(j_data.clues),
        dj_clues=len(dj_result[0].clues) if dj_result else 0,
        total=len(extracted_clues),
    )

    # ── Stage 3: Category Gap Detection & Targeted Fill ─────────────
    def _detect_gaps(
        clues: list[Any],
        manifest: "RoundManifest",
    ) -> dict[str, list[int]]:
        """Identify categories with missing clues.

        Returns a dict of category → list of missing row numbers.
        E.g., {"Songwriters": [1, 2]} means rows 1 and 2 are missing.
        """
        category_rows: dict[str, set[int]] = {}
        for clue in clues:
            if clue.round == manifest.round_name:
                cat_key = clue.category.lower().strip()
                category_rows.setdefault(cat_key, set()).add(clue.board_row)

        gaps: dict[str, list[int]] = {}
        for cat in manifest.categories:
            cat_key = cat.lower().strip()
            found_rows = category_rows.get(cat_key, set())
            expected_rows = set(range(1, manifest.clues_per_category + 1))
            missing = sorted(expected_rows - found_rows)
            if missing:
                gaps[cat] = missing

        return gaps

    j_gaps = _detect_gaps(extracted_clues, j_manifest)
    dj_gaps = _detect_gaps(extracted_clues, dj_manifest)

    total_gaps = sum(len(v) for v in j_gaps.values()) + sum(len(v) for v in dj_gaps.values())
    if total_gaps > 0:
        logger.warning(
            "Category gaps detected — triggering targeted re-extraction",
            j_gaps={k: v for k, v in j_gaps.items()},
            dj_gaps={k: v for k, v in dj_gaps.items()},
            total_missing_clues=total_gaps,
        )

        gap_fill_tasks = []

        for cat, missing_rows in j_gaps.items():
            gap_context = (
                f"Category '{cat}' in {j_manifest.round_name} round should have "
                f"{j_manifest.clues_per_category} clues. "
                f"Missing clues at rows: {missing_rows}. "
                f"Find ONLY these {len(missing_rows)} missing clue(s)."
            )
            gap_fill_tasks.append(
                _extract_round("J!", j_text, [cat], j_schema, j_manifest, is_gap_fill=True, gap_context=gap_context)
            )

        for cat, missing_rows in dj_gaps.items():
            gap_context = (
                f"Category '{cat}' in {dj_manifest.round_name} round should have "
                f"{dj_manifest.clues_per_category} clues. "
                f"Missing clues at rows: {missing_rows}. "
                f"Find ONLY these {len(missing_rows)} missing clue(s)."
            )
            gap_fill_tasks.append(
                _extract_round(
                    "Double J!",
                    dj_text if dj_text else j_text,
                    [cat],
                    dj_schema,
                    dj_manifest,
                    is_gap_fill=True,
                    gap_context=gap_context,
                )
            )

        if gap_fill_tasks:
            gap_results = await asyncio.gather(*gap_fill_tasks, return_exceptions=True)
            gap_filled = 0
            for result in gap_results:  # type: ignore[assignment]
                if isinstance(result, BaseException):
                    logger.warning("Gap-fill call failed", error=str(result)[:200])
                    continue
                gap_data, gap_usage, gap_att = result
                _accumulate_usage(gap_usage)  # type: ignore[arg-type]
                max_attempt_reached = max(max_attempt_reached, gap_att)  # type: ignore[assignment]
                extracted_clues.extend(list(gap_data.clues))  # type: ignore[attr-defined]
                gap_filled += len(gap_data.clues)  # type: ignore[attr-defined]

            logger.info(
                "Gap-fill extraction complete",
                gap_fill_calls=len(gap_fill_tasks),
                new_clues_found=gap_filled,
                total_clues=len(extracted_clues),
            )
    else:
        logger.info(
            "No category gaps detected — all categories fully extracted",
            j_categories=len(j_manifest.categories),
            dj_categories=len(dj_manifest.categories),
        )

    # Package results for Stage 4 (reconstruction)
    class _SyntheticChunkData:
        def __init__(self, clues: list[Any]) -> None:
            self.clues = clues

    final_chunk_results = [(_SyntheticChunkData(extracted_clues), total_usage, max_attempt_reached)]

    logger.info(
        "Manifest-Verify-Fill extraction complete",
        total_clues=len(extracted_clues),
        gap_fills_triggered=total_gaps,
    )

    # ── Stage 3.5: Verify & Correct (Flash) ─────────────────────────
    # Cross-validate extracted clue_text and correct_response against
    # the surrounding transcript context using cheap Flash calls.
    # This catches ASR artifacts, selection preamble bleed, and
    # unverified correct_responses for triple stumpers.
    from trebek.llm.verify import verify_and_correct_clues, verify_final_jeopardy

    verify_corrections, verify_usage = await verify_and_correct_clues(
        extracted_clues,
        segments,
        contestant_names,
        model=MODEL_FLASH,
    )
    _accumulate_usage(verify_usage)

    # Verify Final Jeopardy correct_response
    fj_verified_response, fj_verify_usage = await verify_final_jeopardy(
        meta_data.final_jep,
        segments,
        contestant_names,
        model=MODEL_FLASH,
    )
    _accumulate_usage(fj_verify_usage)

    if fj_verified_response:
        meta_data.final_jep.correct_response = fj_verified_response

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

            # ASR reconstruction from WhisperX segments (used for timestamps
            # and as fallback if LLM-extracted clue_text is missing)
            asr_clue_text = " ".join([seg.get("text", "").strip() for seg in segments[s_idx : e_idx + 1]])

            # ── Dual-source clue_text resolution ──────────────────
            # Primary: LLM-extracted text (verified by Stage 3.5).
            # The LLM understands semantic boundaries (where the
            # selection announcement ends and the actual clue begins)
            # far better than segment splicing.
            # Fallback: ASR-reconstructed text from segment Line IDs.
            llm_clue_text = getattr(ext_clue, "clue_text", "").strip()
            if llm_clue_text and len(llm_clue_text) >= 10:
                clue_text = llm_clue_text
            else:
                # LLM text missing or too short — fall back to ASR
                clue_text = _strip_selection_preamble(asr_clue_text, ext_clue.category)

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

            # ── Board row inference from dollar value ──────────────
            # Instead of trusting the LLM's guess (it can't see the board),
            # parse the dollar value from the selection text in the transcript
            # and deterministically map it to a row using the known value grid.
            from trebek.llm.board import infer_board_row, _parse_dollar_value

            # Look for dollar value in the 2 segments before the clue start.
            # The contestant's category+value selection is typically in the
            # 1-2 lines immediately preceding the host's clue read.
            # A wider window (3+) catches contestant scores and wagers.
            llm_row_guess = max(1, min(ext_clue.board_row, 5))
            selection_context = ""
            if s_idx > 0:
                for ctx_idx in range(max(0, s_idx - 2), s_idx + 1):
                    if ctx_idx < len(segments):
                        selection_context += " " + segments[ctx_idx].get("text", "")

            parsed_value = _parse_dollar_value(selection_context)
            if parsed_value is not None:
                inferred_row = infer_board_row(
                    parsed_value,
                    ext_clue.round,
                    board_format,
                    llm_fallback_row=llm_row_guess,
                )
            else:
                # No valid clue value found — use LLM's guess, clamped
                inferred_row = llm_row_guess
            clamped_col = max(1, min(ext_clue.board_col, 6))

            all_clues.append(
                Clue(
                    round=ext_clue.round,
                    category=ext_clue.category,
                    board_row=inferred_row,
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
                    is_verified=ext_clue.is_verified,
                    original_response=ext_clue.original_response,
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

    from trebek.llm.validation import resolve_duplicate_board_rows

    resolve_duplicate_board_rows(unique_clues)

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

    # ── Stage 4c: Response Format Normalization ─────────────────────
    # Ensure all correct_response values are in J! question format
    # (e.g., "What is Paris?" not just "Paris")
    _response_fixes = normalize_response_format(sorted_clues)

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
        "Manifest-Verify-Fill pipeline complete",
        total_clues_extracted=len(sorted_clues),
        max_retries_used=max_attempt_reached,
        model=model,
    )

    # ── Episode Quality Score ────────────────────────────────────────
    # Severity-weighted: board_row duplicates (cosmetic, 0.5 weight)
    # vs structural issues (1.0 weight). Consistent with the quality
    # gate in the state machine worker.
    weighted_score = sum(0.5 if "duplicate board_row" in w else 1.0 for w in integrity_warnings)
    quality_score = "PASS" if not integrity_warnings else ("DEGRADED" if weighted_score <= 3.0 else "FAIL")

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
