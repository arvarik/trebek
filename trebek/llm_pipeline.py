import ast
import json
import structlog
import asyncio
from typing import Any, Dict, Optional, TypeVar
from pydantic import ValidationError, BaseModel, Field
from trebek.schemas import Episode, Contestant, FinalJeopardy, ScoreAdjustment, Clue

_T = TypeVar("_T", bound=BaseModel)

logger = structlog.get_logger()

_client: Optional["GeminiClient"] = None

# Concurrency guard — prevents rate-limit cascades when firing parallel chunk extractions.
# Gemini 3.1 Pro allows 5 RPM on free tier, 1000 on paid; 3 is a safe concurrent ceiling.
GEMINI_CONCURRENCY = 3


def _get_client() -> "GeminiClient":
    """Lazy-initializes the Gemini client on first use, not at import time."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


class GeminiClient:
    def __init__(self) -> None:
        import os
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required but not set. "
                "Set it in your .env file or export it before running the pipeline."
            )
        self.client = genai.Client(api_key=api_key)

    async def generate_content(
        self,
        model: str,
        prompt: str,
        system_instruction: str,
        response_schema: Optional[type[BaseModel]] = None,
        max_output_tokens: int = 65536,
        cached_content_name: Optional[str] = None,
    ) -> "tuple[str, dict[str, float]]":
        import re
        import random
        import time

        from google.genai import types

        kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_output_tokens": max_output_tokens,
        }

        # Context caching constraint: system_instruction cannot be set per-call
        # when using cached_content — it must be baked into the cache.
        if not cached_content_name:
            kwargs["system_instruction"] = system_instruction

        if response_schema:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema
        else:
            kwargs["response_mime_type"] = "application/json"

        if cached_content_name:
            kwargs["cached_content"] = cached_content_name

        config = types.GenerateContentConfig(**kwargs)

        max_retries = 10
        base_delay = 5.0  # seconds

        for attempt in range(max_retries + 1):
            try:
                start_t = time.perf_counter()
                response = await self.client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                latency_ms = (time.perf_counter() - start_t) * 1000

                usage: dict[str, float] = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = {
                        "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                        "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                        "cached_tokens": getattr(response.usage_metadata, "cached_content_token_count", 0),
                        "latency_ms": latency_ms,
                    }
                else:
                    usage = {"input_tokens": 0.0, "output_tokens": 0.0, "cached_tokens": 0.0, "latency_ms": latency_ms}

                return str(response.text), usage

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_server_error = "500" in err_str or "503" in err_str or "UNAVAILABLE" in err_str

                if not (is_rate_limit or is_server_error) or attempt == max_retries:
                    raise

                # Parse server-suggested retry delay if available
                retry_match = re.search(r"retry.*?(\d+\.?\d*)\s*s", err_str, re.IGNORECASE)
                if retry_match:
                    delay = float(retry_match.group(1)) + random.uniform(1.0, 3.0)
                else:
                    delay = min(base_delay * (2**attempt), 120.0) + random.uniform(0.5, 2.0)

                logger.warning(
                    "Rate limited, retrying",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay_s=round(delay, 1),
                    model=model,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("Unreachable")


async def execute_pass_1_speaker_anchoring(host_interview_segment: str) -> "tuple[Dict[str, str], dict[str, float]]":
    """
    Pass 1: Fast Gemini 3.1 Flash-Lite extraction isolating the host interview.
    Generates a rigid speaker mapping dictionary to prevent hallucinations.
    """
    system_prompt = (
        "You are a strict data extractor. Analyze the Jeopardy host interview segment. "
        "Return a pure JSON dictionary mapping Diarization Speaker IDs to Contestant/Host Names. "
        'Format: {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt Amodio"}. '
        "Use double quotes only. Return ONLY the JSON object, no markdown."
    )

    try:
        client = _get_client()
        response_text, usage = await client.generate_content(
            model="gemini-3.1-flash-lite-preview",
            prompt=f"Segment:\n{host_interview_segment}",
            system_instruction=system_prompt,
            max_output_tokens=2048,  # Speaker mapping is tiny
        )
        # Clean markdown fences if present
        clean = response_text.replace("```json", "").replace("```", "").strip()
        if not clean:
            logger.warning("Pass 1 returned empty response, using empty mapping")
            return {}, usage
        # Try json.loads first, fall back to ast.literal_eval for single-quoted dicts
        try:
            mapping: Dict[str, str] = json.loads(clean)
        except json.JSONDecodeError:
            mapping = dict(ast.literal_eval(clean))
        logger.info("Pass 1 Speaker Anchor resolved", mapping=mapping)
        return mapping, usage
    except Exception as e:
        logger.error("Failed to generate speaker anchor", error=str(e))
        # Fallback or default mapping logic here
        return {}, {"input_tokens": 0.0, "output_tokens": 0.0, "cached_tokens": 0.0, "latency_ms": 0.0}


# ─────────────────────────────────────────────────────────────────────
#  Partial schemas — keep extraction surfaces as flat as possible
#  to reduce schema complexity and improve constrained decoding.
# ─────────────────────────────────────────────────────────────────────


class PartialEpisodeMeta(BaseModel):
    episode_date: str
    host_name: str
    is_tournament: bool
    contestants: list[Contestant]
    final_jeopardy: FinalJeopardy
    score_adjustments: list[ScoreAdjustment]


class PartialClues(BaseModel):
    clues: list[Clue]


class EpisodeSkeleton(BaseModel):
    """Lightweight structural contract extracted before full clue extraction."""

    jeopardy_categories: list[str] = Field(description="The 6 Jeopardy round category names, left to right.")
    double_jeopardy_categories: list[str] = Field(
        description="The 6 Double Jeopardy round category names, left to right."
    )
    total_jeopardy_clues_played: int = Field(description="How many Jeopardy round clues were actually played (max 30).")
    total_double_jeopardy_clues_played: int = Field(
        description="How many Double Jeopardy round clues were actually played (max 30)."
    )
    daily_double_count: int = Field(description="Total number of Daily Doubles in this episode (typically 3).")


def _estimate_output_tokens(num_input_lines: int, extraction_type: str) -> int:
    """
    Conservative output token budget estimation.
    Prevents both truncation (too low) and wasted quota (too high).
    """
    if extraction_type == "meta":
        return 4096  # Metadata is compact
    elif extraction_type == "skeleton":
        return 2048  # Just category names and counts
    elif extraction_type == "clues":
        # ~250 tokens per clue (including nested BuzzAttempts),
        # assume roughly 1 clue per 4 transcript lines
        estimated_clues = max(1, num_input_lines // 4)
        return min(max(estimated_clues * 300, 8192), 65536)
    return 16384


async def _extract_part(
    prompt: str,
    system_prompt: str,
    schema_cls: type[_T],
    max_retries: int,
    max_output_tokens: int = 65536,
    cached_content_name: Optional[str] = None,
) -> "tuple[_T, dict[str, float], int]":
    """
    Core extraction primitive with self-healing retry loop.

    On validation failure, feeds the Pydantic error back into the prompt
    so the model can self-correct. Uses a FRESH prompt (not appending to
    the growing context) to avoid prompt inflation.
    """
    total_usage: dict[str, float] = {"input_tokens": 0.0, "output_tokens": 0.0, "cached_tokens": 0.0, "latency_ms": 0.0}
    last_error_feedback: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            # Build the prompt — on retries, replace (not append) error context
            # to avoid inflating the prompt on each retry
            if last_error_feedback and attempt > 0:
                current_prompt = (
                    prompt + "\n\n--- CRITICAL: FIX THESE ERRORS ---\n"
                    "Your previous JSON output failed validation. "
                    "Return corrected JSON matching the schema exactly.\n"
                    f"ERRORS:\n{last_error_feedback}"
                )
            else:
                current_prompt = prompt

            client = _get_client()
            response_text, usage = await client.generate_content(
                model="gemini-3.1-pro-preview",
                prompt=current_prompt,
                system_instruction=system_prompt,
                response_schema=schema_cls,
                max_output_tokens=max_output_tokens,
                cached_content_name=cached_content_name,
            )

            for k in total_usage:
                total_usage[k] += usage.get(k) or 0.0

            # Schema-constrained generation should produce clean JSON,
            # but strip markdown fences defensively
            clean_json = response_text.replace("```json", "").replace("```", "").strip()

            validated_data = await asyncio.to_thread(schema_cls.model_validate_json, clean_json)
            return validated_data, total_usage, attempt

        except ValidationError as validation_error:
            logger.warning(
                "Pydantic Validation failed for part",
                attempt=attempt + 1,
                error_count=validation_error.error_count(),
                schema=schema_cls.__name__,
            )
            if attempt == max_retries:
                raise validation_error

            # Store error for next attempt — keep it concise to avoid prompt bloat
            errors = validation_error.errors()
            last_error_feedback = "\n".join(
                f"- Field '{'.'.join(str(loc) for loc in e['loc'])}': {e['msg']}"
                for e in errors[:10]  # Cap at 10 errors to prevent prompt inflation
            )

    raise RuntimeError("Unexpected escape from _extract_part self-healing loop.")


# ─────────────────────────────────────────────────────────────────────
#  Semantic Chunking — split by Jeopardy round boundaries, not
#  arbitrary line counts.  Falls back to sized chunks for safety.
# ─────────────────────────────────────────────────────────────────────

_ROUND_MARKERS = [
    "double jeopardy",
    "final jeopardy",
    "tiebreaker",
]


def _chunk_by_semantic_boundaries(transcript_lines: list[str], max_chunk_lines: int = 400) -> list[str]:
    """
    Splits transcript at Jeopardy round transitions.
    If any resulting chunk exceeds max_chunk_lines, subdivides it with
    overlap to prevent cutting a clue mid-read.
    """
    raw_chunks: list[list[str]] = []
    current_chunk: list[str] = []

    for line in transcript_lines:
        current_chunk.append(line)
        # Detect round boundaries in a case-insensitive manner
        lower_line = line.lower()
        if any(marker in lower_line for marker in _ROUND_MARKERS) and len(current_chunk) > 10:
            raw_chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        raw_chunks.append(current_chunk)

    # Subdivide any oversized chunks with overlap
    final_chunks: list[str] = []
    overlap = 40

    for chunk_lines in raw_chunks:
        if len(chunk_lines) <= max_chunk_lines:
            final_chunks.append("\n".join(chunk_lines))
        else:
            # Subdivide with overlap
            i = 0
            while i < len(chunk_lines):
                end = min(i + max_chunk_lines, len(chunk_lines))
                final_chunks.append("\n".join(chunk_lines[i:end]))
                i += max_chunk_lines - overlap
                if i >= len(chunk_lines):
                    break

    # Fallback: if semantic splitting produced only 1 giant chunk (no markers found),
    # fall back to sized chunking with overlap
    if len(final_chunks) == 1 and len(transcript_lines) > max_chunk_lines:
        logger.warning("No round markers found, falling back to sized chunking")
        final_chunks = []
        i = 0
        while i < len(transcript_lines):
            end = min(i + max_chunk_lines, len(transcript_lines))
            final_chunks.append("\n".join(transcript_lines[i:end]))
            i += max_chunk_lines - overlap
            if i >= len(transcript_lines):
                break

    return final_chunks


# ─────────────────────────────────────────────────────────────────────
#  Output Integrity Validation — deterministic domain-knowledge checks
#  that catch LLM hallucinations without another LLM call.
# ─────────────────────────────────────────────────────────────────────


def _validate_extraction_integrity(episode: Episode) -> list[str]:
    """
    Post-extraction deterministic sanity checks encoding Jeopardy domain rules.
    Returns a list of warning strings (empty = clean extraction).
    """
    warnings: list[str] = []

    # 1. Clue count bounds (standard episode: up to 61 clues including FJ)
    j_clues = [c for c in episode.clues if c.round == "Jeopardy"]
    dj_clues = [c for c in episode.clues if c.round == "Double Jeopardy"]

    if len(j_clues) > 30:
        warnings.append(f"Jeopardy round has {len(j_clues)} clues (max 30)")
    if len(dj_clues) > 30:
        warnings.append(f"Double Jeopardy round has {len(dj_clues)} clues (max 30)")
    if len(j_clues) == 0:
        warnings.append("No Jeopardy round clues extracted")
    if len(dj_clues) == 0:
        warnings.append("No Double Jeopardy round clues extracted")

    # 2. Timestamp monotonicity within each round
    for round_name, round_clues in [("Jeopardy", j_clues), ("Double Jeopardy", dj_clues)]:
        sorted_round = sorted(round_clues, key=lambda c: c.host_start_timestamp_ms)
        for i in range(1, len(sorted_round)):
            if sorted_round[i].host_start_timestamp_ms < sorted_round[i - 1].host_finish_timestamp_ms:
                warnings.append(
                    f"{round_name} clue {i + 1} overlaps with clue {i}: "
                    f"start {sorted_round[i].host_start_timestamp_ms} < "
                    f"prev finish {sorted_round[i - 1].host_finish_timestamp_ms}"
                )
                break  # One overlap warning per round is enough

    # 3. Daily Double count (standard: 1 in J, 2 in DJ)
    dd_j = sum(1 for c in j_clues if c.is_daily_double)
    dd_dj = sum(1 for c in dj_clues if c.is_daily_double)
    total_dd = dd_j + dd_dj

    if total_dd > 3:
        warnings.append(f"Found {total_dd} Daily Doubles (expected max 3)")
    if dd_j > 1:
        warnings.append(f"Found {dd_j} Daily Doubles in Jeopardy round (expected 1)")
    if dd_dj > 2:
        warnings.append(f"Found {dd_dj} Daily Doubles in Double Jeopardy (expected 2)")

    # 4. Contestant name consistency — names in attempts should match roster
    contestant_names = {c.name.lower().strip() for c in episode.contestants}
    unknown_buzzers = set()
    for clue in episode.clues:
        for attempt in clue.attempts:
            if attempt.speaker.lower().strip() not in contestant_names:
                unknown_buzzers.add(attempt.speaker)

    if unknown_buzzers:
        warnings.append(f"Unknown contestants in buzz attempts: {unknown_buzzers}")

    # 5. Board position validity
    for clue in episode.clues:
        if clue.round in ("Jeopardy", "Double Jeopardy"):
            if not (1 <= clue.board_row <= 5):
                warnings.append(f"Clue '{clue.clue_text[:30]}' has invalid board_row: {clue.board_row}")
            if not (1 <= clue.board_col <= 6):
                warnings.append(f"Clue '{clue.clue_text[:30]}' has invalid board_col: {clue.board_col}")

    return warnings


# ─────────────────────────────────────────────────────────────────────
#  Deduplication — merge clues from overlapping chunks
# ─────────────────────────────────────────────────────────────────────


def _deduplicate_clues(all_clues: list[Clue]) -> list[Clue]:
    """
    Deduplicates clues from overlapping chunks using a composite key of:
    - Rounded timestamp (100ms granularity)
    - Category name
    - Board position (row, col)

    When duplicates collide, keeps the version with more buzz attempts
    (indicating a more complete extraction).
    """
    unique_clues: dict[str, Clue] = {}

    for clue in all_clues:
        # Composite dedup key: timestamp bucket + board position
        time_bucket = round(clue.host_start_timestamp_ms / 100.0) * 100
        key = f"{time_bucket}_{clue.round}_{clue.board_row}_{clue.board_col}"

        if key not in unique_clues:
            unique_clues[key] = clue
        else:
            # Keep the more complete extraction (more attempts = more data preserved)
            existing = unique_clues[key]
            if len(clue.attempts) > len(existing.attempts):
                unique_clues[key] = clue
            elif len(clue.attempts) == len(existing.attempts) and len(clue.clue_text) > len(existing.clue_text):
                # Same attempt count — prefer longer clue text (less truncated)
                unique_clues[key] = clue

    return list(unique_clues.values())


# ─────────────────────────────────────────────────────────────────────
#  Pass 2: Map-Reduce Data Extraction Pipeline
# ─────────────────────────────────────────────────────────────────────


async def execute_pass_2_data_extraction(
    full_transcript: str, speaker_mapping: Dict[str, str], max_retries: int = 2
) -> "tuple[Episode, dict[str, float], int]":
    """
    Pass 2: Production-grade map-reduce extraction (Gemini 3.1 Pro).

    Architecture:
    1. Extract Episode Meta (contestants, FJ, score adjustments) — full transcript
    2. Semantic chunking by round boundaries (not arbitrary line counts)
    3. Semaphore-bounded concurrent clue extraction per chunk
    4. Composite-key deduplication for overlapping chunk regions
    5. Deterministic integrity validation encoding Jeopardy domain rules
    """
    logger.info("Starting Map-Reduce extraction pipeline...")

    base_system = (
        "You are Trebek, an expert data extraction pipeline for Jeopardy game transcripts. "
        "Extract game events into the provided JSON schema with surgical precision. "
        f"CRITICAL CONSTRAINT: Map speakers using this exact dictionary: {json.dumps(speaker_mapping)}. "
        "Do NOT hallucinate names outside this mapping. "
        "Do NOT perform any running score math — extract only observable facts. "
        "Every timestamp must be copied verbatim from the transcript's WhisperX alignment data."
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
        max_output_tokens=_estimate_output_tokens(0, "meta"),
    )
    _accumulate_usage(meta_usage)
    max_attempt_reached = max(max_attempt_reached, meta_att)

    # ── Stage 2: Semantic Chunking ──────────────────────────────────
    lines = full_transcript.split("\n")
    chunks = _chunk_by_semantic_boundaries(lines)
    logger.info(
        "Chunked transcript for concurrent clue extraction",
        num_chunks=len(chunks),
        strategy="semantic_round_boundaries",
    )

    # ── Stage 3: Semaphore-Bounded Concurrent Clue Extraction ──────
    semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)

    async def extract_chunk(chunk_idx: int, chunk_text: str) -> "tuple[PartialClues, dict[str, float], int]":
        async with semaphore:
            chunk_lines = chunk_text.count("\n") + 1
            logger.info(f"Extracting clues from chunk {chunk_idx + 1}/{len(chunks)}", lines=chunk_lines)
            prompt = (
                f"Transcript Chunk ({chunk_idx + 1} of {len(chunks)}):\n{chunk_text}\n\n"
                "Extract ALL Jeopardy and Double Jeopardy clues found in this chunk. "
                "If a clue appears cut off at the start or end of this chunk, SKIP IT completely "
                "(overlapping chunks will capture it whole). "
                "If no clues are found in this chunk, return an empty array for clues. "
                "Ensure you extract ALL buzz attempts and copy ALL timestamps verbatim."
            )
            output_budget = _estimate_output_tokens(chunk_lines, "clues")
            return await _extract_part(
                prompt,
                base_system,
                PartialClues,
                max_retries,
                max_output_tokens=output_budget,
            )

    chunk_tasks = [extract_chunk(idx, chunk) for idx, chunk in enumerate(chunks)]
    chunk_results = await asyncio.gather(*chunk_tasks)

    # ── Stage 4: Merge & Deduplicate ────────────────────────────────
    all_clues: list[Clue] = []
    for chunk_data, chunk_usage, chunk_att in chunk_results:
        all_clues.extend(chunk_data.clues)
        _accumulate_usage(chunk_usage)
        max_attempt_reached = max(max_attempt_reached, chunk_att)

    logger.info("Raw clues before deduplication", count=len(all_clues))
    unique_clues = _deduplicate_clues(all_clues)

    # Sort chronologically and re-index selection_order
    sorted_clues = sorted(unique_clues, key=lambda c: c.host_start_timestamp_ms)
    for i, clue in enumerate(sorted_clues):
        clue.selection_order = i + 1

    logger.info("Clues after deduplication", count=len(sorted_clues))

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
    )
    return episode, total_usage, max_attempt_reached
