"""
Stage 3.5: Post-extraction verification and correction pass.

Cross-validates extracted clue data against the surrounding transcript
context using cheap Flash calls. Catches three categories of errors:

1. **Incorrect correct_response**: The LLM's extracted answer doesn't match
   what the host actually reveals on air (especially for triple stumpers).
2. **Selection preamble in clue_text**: The LLM-extracted clue text includes
   the category/value announcement instead of just the clue content.
3. **ASR transcription artifacts**: WhisperX mishears words that the LLM
   can correct using world knowledge (e.g., "Kemu" → "Camus").

Architecture:
- Batches clues in groups of 12 for efficient Flash utilization
- Runs all batches concurrently with a semaphore
- Each batch gets the surrounding transcript context for verification
- Corrections are logged with severity classification (typo vs major)
- FJ verification is included for consistency
"""

import asyncio
import structlog
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field

from trebek.config import MODEL_FLASH
from trebek.llm.utils import _extract_part

logger = structlog.get_logger()

# ── Verification batch size ─────────────────────────────────────────
# 12 clues per batch = 5 batches for a standard 60-clue episode.
# Balances Flash output token budget against call overhead.
VERIFY_BATCH_SIZE = 12

# Concurrency limit for verification calls
VERIFY_CONCURRENCY = 5

# Lines of transcript context before/after each clue for verification
CONTEXT_LINES_BEFORE = 3
CONTEXT_LINES_AFTER = 8


# ── Verification Schemas ────────────────────────────────────────────


class SingleClueVerification(BaseModel):
    clue_index: int = Field(description="The 0-based index of the clue being verified within this batch.")
    verified_clue_text: str = Field(
        description=(
            "The actual clue text as read by the host. "
            "Must NOT include the category selection announcement "
            "(e.g., remove 'Songwriters for 800. '). "
            "Correct any obvious ASR transcription errors using context."
        )
    )
    verified_correct_response: str = Field(
        description=(
            "The correct response verified against the host's on-air reveal. "
            "MUST be in J! question form (e.g., 'What is Paris?'). "
            "If the host reveals the answer after contestants fail, use that exact reveal. "
            "If a contestant answered correctly, use the semantically correct version "
            "(fix ASR misspellings like 'Kemu' → 'Camus')."
        )
    )
    confidence: Literal["verified", "corrected", "unverifiable"] = Field(
        description=(
            "'verified' = extraction matches transcript. "
            "'corrected' = extraction had errors that were fixed. "
            "'unverifiable' = cannot confirm from transcript context alone."
        )
    )
    correction_type: str = Field(
        default="",
        description=(
            "If confidence is 'corrected', classify as: "
            "'typo' (ASR misspelling), "
            "'preamble_stripped' (selection text removed from clue), "
            "'response_fixed' (wrong answer corrected from host reveal), "
            "'major' (significant content change). "
            "Empty string if not corrected."
        ),
    )
    correction_detail: str = Field(default="", description="Brief explanation of what was corrected, if anything.")


class BatchVerificationResult(BaseModel):
    verifications: list[SingleClueVerification]


class FJVerification(BaseModel):
    verified_correct_response: str = Field(
        description=(
            "The correct Final J! response verified against the host's on-air reveal. "
            "The host ALWAYS reveals the correct answer after showing contestant responses. "
            "MUST be in J! question form."
        )
    )
    confidence: Literal["verified", "corrected", "unverifiable"]
    correction_type: str = Field(default="")
    correction_detail: str = Field(default="")


# ── Transcript Context Builder ──────────────────────────────────────


def _build_clue_context(
    clue: Any,
    segments: list[dict[str, Any]],
    context_before: int = CONTEXT_LINES_BEFORE,
    context_after: int = CONTEXT_LINES_AFTER,
) -> str:
    """Build transcript context window around a clue for verification.

    Returns the transcript lines surrounding the clue, with markers
    showing where the clue starts and ends.
    """

    # Parse Line IDs
    start_raw = clue.host_read_start_line_id.replace("L", "").replace("[", "").replace("]", "").strip()
    end_raw = clue.host_read_end_line_id.replace("L", "").replace("[", "").replace("]", "").strip()

    try:
        s_idx = int(start_raw)
        e_idx = int(end_raw)
    except ValueError:
        return "(Unable to resolve clue position in transcript)"

    s_idx = max(0, min(s_idx, len(segments) - 1))
    e_idx = max(0, min(e_idx, len(segments) - 1))

    # Build context window
    ctx_start = max(0, s_idx - context_before)
    ctx_end = min(len(segments) - 1, e_idx + context_after)

    lines = []
    for i in range(ctx_start, ctx_end + 1):
        seg = segments[i]
        speaker = seg.get("speaker", "?")
        text = seg.get("text", "").strip()
        marker = ""
        if i == s_idx:
            marker = " ◄ CLUE START"
        if i == e_idx:
            marker += " ◄ CLUE END"
        lines.append(f"L{i} [{speaker}]: {text}{marker}")

    return "\n".join(lines)


def _build_fj_context(
    segments: list[dict[str, Any]],
) -> str:
    """Build transcript context for the Final Jeopardy section.

    Uses the last ~30 segments of the transcript, which typically
    contains the FJ category reveal, clue, think music, contestant
    responses, and the host's correct answer reveal.
    """
    fj_start = max(0, len(segments) - 40)
    lines = []
    for i in range(fj_start, len(segments)):
        seg = segments[i]
        speaker = seg.get("speaker", "?")
        text = seg.get("text", "").strip()
        lines.append(f"L{i} [{speaker}]: {text}")
    return "\n".join(lines)


# ── Core Verification Logic ─────────────────────────────────────────


async def verify_and_correct_clues(
    extracted_clues: list[Any],
    segments: list[dict[str, Any]],
    contestant_names: list[str],
    model: str = MODEL_FLASH,
    max_retries: int = 2,
) -> "tuple[list[dict[str, Any]], dict[str, float]]":
    """Stage 3.5: Post-extraction verification pass.

    For each extracted clue, sends the surrounding transcript context to
    Flash and asks it to verify/correct the clue_text and correct_response.

    Batches clues in groups of 12 for efficient Flash utilization.
    All batches run concurrently with a semaphore.

    Returns:
        - List of correction dicts: [{clue_index, field, old, new, type, detail}]
        - Accumulated usage stats for the verification pass
    """
    total_usage: dict[str, float] = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "thinking_tokens": 0.0,
        "cached_tokens": 0.0,
        "total_tokens": 0.0,
        "cost_usd": 0.0,
        "latency_ms": 0.0,
    }

    corrections: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(VERIFY_CONCURRENCY)

    # Build batches of 12
    batches: list[list[tuple[int, Any]]] = []
    current_batch: list[tuple[int, Any]] = []
    for i, clue in enumerate(extracted_clues):
        current_batch.append((i, clue))
        if len(current_batch) >= VERIFY_BATCH_SIZE:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    logger.info(
        "Stage 3.5: Starting verification pass",
        total_clues=len(extracted_clues),
        batches=len(batches),
        batch_size=VERIFY_BATCH_SIZE,
        model=model,
    )

    async def _verify_batch(
        batch: list[tuple[int, Any]],
        batch_num: int,
    ) -> "tuple[list[dict[str, Any]], dict[str, float]]":
        """Verify a single batch of clues."""
        async with semaphore:
            # Build the verification prompt
            clue_blocks = []
            for batch_idx, (global_idx, clue) in enumerate(batch):
                context = _build_clue_context(clue, segments)

                # Get the correct contestant response if any
                correct_attempts = [a for a in clue.attempts if a.is_correct]
                contestant_response = (
                    correct_attempts[0].response_given if correct_attempts else "N/A (triple stumper or all wrong)"
                )

                clue_blocks.append(
                    f"--- Clue {batch_idx} (#{global_idx + 1}) ---\n"
                    f"Category: {clue.category}\n"
                    f"Extracted clue_text: {clue.clue_text}\n"
                    f"Extracted correct_response: {clue.correct_response}\n"
                    f"Contestant response (marked correct): {contestant_response}\n"
                    f"Transcript context:\n{context}\n"
                )

            prompt = (
                "VERIFY the following extracted J! clue data against the transcript context.\n\n"
                "For each clue:\n"
                "1. CHECK if clue_text matches what the host actually reads (no selection preamble).\n"
                "2. CHECK if correct_response matches what the host reveals as the answer.\n"
                "   - For clues answered correctly: verify against the contestant's response.\n"
                "   - For triple stumpers/wrong answers: find the host's reveal AFTER the last buzz.\n"
                "3. FIX any ASR transcription errors (e.g., 'Kemu' → 'Camus').\n"
                "4. PRESERVE the J! question format for correct_response.\n\n" + "\n".join(clue_blocks)
            )

            system = (
                "You are a J! data verification agent. Your job is to cross-check "
                "extracted game data against the raw transcript. "
                "Fix errors but do NOT hallucinate information not present in the transcript. "
                "If you cannot verify a field from the transcript, set confidence to 'unverifiable'. "
                f"Contestants in this episode: {', '.join(contestant_names)}."
            )

            try:
                result, usage, attempt = await _extract_part(
                    prompt,
                    system,
                    BatchVerificationResult,
                    max_retries=max_retries,
                    model=model,
                    invocation_context=f"Stage 3.5 Verify batch {batch_num + 1}/{len(batches)}",
                    thinking_level="low",
                )

                batch_corrections: list[dict[str, Any]] = []

                for v in result.verifications:
                    if v.clue_index < 0 or v.clue_index >= len(batch):
                        logger.warning(
                            "Verification returned out-of-range clue_index",
                            clue_index=v.clue_index,
                            batch_size=len(batch),
                            batch_num=batch_num,
                        )
                        continue

                    global_idx, original_clue = batch[v.clue_index]

                    # Check for clue_text corrections
                    if v.confidence == "corrected" and v.verified_clue_text != original_clue.clue_text:
                        correction = {
                            "clue_index": global_idx,
                            "field": "clue_text",
                            "old": original_clue.clue_text,
                            "new": v.verified_clue_text,
                            "correction_type": v.correction_type,
                            "correction_detail": v.correction_detail,
                        }
                        batch_corrections.append(correction)
                        # Apply the correction to the clue object
                        original_clue.clue_text = v.verified_clue_text
                        original_clue.is_verified = True

                    # Check for correct_response corrections
                    if v.confidence == "corrected" and v.verified_correct_response != original_clue.correct_response:
                        correction = {
                            "clue_index": global_idx,
                            "field": "correct_response",
                            "old": original_clue.correct_response,
                            "new": v.verified_correct_response,
                            "correction_type": v.correction_type,
                            "correction_detail": v.correction_detail,
                        }
                        batch_corrections.append(correction)
                        # Apply the correction
                        original_clue.original_response = original_clue.correct_response
                        original_clue.correct_response = v.verified_correct_response
                        original_clue.is_verified = True

                    # Even for "verified" clues, adopt the verified text if it's
                    # meaningfully different (the verifier may have cleaned up
                    # without flagging as "corrected")
                    if v.confidence == "verified":
                        original_clue.is_verified = True
                        # Adopt verified clue_text if it's cleaner (no preamble)
                        if (
                            v.verified_clue_text
                            and len(v.verified_clue_text) >= 10
                            and v.verified_clue_text != original_clue.clue_text
                            and len(v.verified_clue_text) < len(original_clue.clue_text)
                        ):
                            original_clue.clue_text = v.verified_clue_text

                logger.info(
                    "Verification batch complete",
                    batch=batch_num + 1,
                    clues_verified=len(result.verifications),
                    corrections=len(batch_corrections),
                    attempt=attempt + 1,
                )

                return batch_corrections, usage

            except Exception as e:
                logger.error(
                    "Verification batch failed — using unverified extraction",
                    batch=batch_num + 1,
                    error=str(e)[:200],
                )
                return [], total_usage

    # Run all batches concurrently
    batch_tasks = [_verify_batch(batch, i) for i, batch in enumerate(batches)]
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

    failed_batches = 0

    for result in batch_results:
        if isinstance(result, BaseException):
            logger.error(
                "Verification batch raised exception",
                error=str(result)[:200],
            )
            failed_batches += 1
            continue

        batch_corrections, batch_usage = result
        corrections.extend(batch_corrections)
        for k in total_usage:
            total_usage[k] += batch_usage.get(k, 0.0)

    # Log all corrections with severity
    for c in corrections:
        log_fn = logger.warning if c["correction_type"] in ("response_fixed", "major") else logger.info
        log_fn(
            "Stage 3.5 correction applied",
            clue_index=c["clue_index"],
            field=c["field"],
            correction_type=c["correction_type"],
            detail=c["correction_detail"],
            old_value=c["old"][:80] if c["old"] else "",
            new_value=c["new"][:80] if c["new"] else "",
        )

    # Summary
    clue_text_corrections = sum(1 for c in corrections if c["field"] == "clue_text")
    response_corrections = sum(1 for c in corrections if c["field"] == "correct_response")
    typo_fixes = sum(1 for c in corrections if c["correction_type"] == "typo")
    preamble_fixes = sum(1 for c in corrections if c["correction_type"] == "preamble_stripped")
    response_fixes = sum(1 for c in corrections if c["correction_type"] == "response_fixed")
    major_fixes = sum(1 for c in corrections if c["correction_type"] == "major")

    logger.info(
        "Stage 3.5 verification complete",
        total_clues=len(extracted_clues),
        total_corrections=len(corrections),
        clue_text_corrections=clue_text_corrections,
        response_corrections=response_corrections,
        typo_fixes=typo_fixes,
        preamble_stripped=preamble_fixes,
        response_fixed=response_fixes,
        major_changes=major_fixes,
        failed_batches=failed_batches,
        verification_cost_usd=round(total_usage.get("cost_usd", 0.0), 6),
    )

    return corrections, total_usage


# ── Final Jeopardy Verification ─────────────────────────────────────


async def verify_final_jeopardy(
    fj_data: Any,
    segments: list[dict[str, Any]],
    contestant_names: list[str],
    model: str = MODEL_FLASH,
    max_retries: int = 2,
) -> "tuple[Optional[str], dict[str, float]]":
    """Verify and extract the Final Jeopardy correct_response.

    The host ALWAYS reveals the FJ answer on air after showing contestant
    responses. This function finds that reveal in the transcript and
    returns the verified correct_response.

    Returns:
        - The verified correct_response string (or None if verification fails)
        - Usage stats
    """
    fj_context = _build_fj_context(segments)

    # Build contestant response summary
    contestant_responses = []
    for w in fj_data.wagers_and_responses:
        mark = "✅" if w.is_correct else "❌"
        contestant_responses.append(f'  {w.contestant}: "{w.response}" {mark}')
    responses_text = "\n".join(contestant_responses)

    prompt = (
        f"VERIFY the Final Jeopardy correct response.\n\n"
        f"Category: {fj_data.category}\n"
        f"Clue: {fj_data.clue_text}\n"
        f'Current extracted correct_response: "{fj_data.correct_response}"\n\n'
        f"Contestant responses:\n{responses_text}\n\n"
        f"Transcript (end of episode, containing FJ):\n{fj_context}\n\n"
        f"Find the host's on-air reveal of the correct answer. "
        f"The host always says the correct response after showing what contestants wrote. "
        f"Return the verified correct response in J! question form."
    )

    system = (
        "You are a J! data verification agent. Extract the correct Final Jeopardy "
        "response from the host's on-air reveal in the transcript. "
        "Do NOT guess — only use what the host actually says."
    )

    try:
        result, usage, attempt = await _extract_part(
            prompt,
            system,
            FJVerification,
            max_retries=max_retries,
            model=model,
            invocation_context="Stage 3.5 FJ Verify",
            thinking_level="low",
        )

        if result.confidence == "corrected":
            logger.warning(
                "Stage 3.5 FJ correction applied",
                correction_type=result.correction_type,
                detail=result.correction_detail,
                old_response=fj_data.correct_response[:80] if fj_data.correct_response else "",
                new_response=result.verified_correct_response[:80],
            )
        elif result.confidence == "verified":
            logger.info(
                "Stage 3.5 FJ verified",
                correct_response=result.verified_correct_response[:80],
            )
        else:
            logger.warning(
                "Stage 3.5 FJ unverifiable — keeping original",
                original=fj_data.correct_response[:80] if fj_data.correct_response else "N/A",
            )

        return result.verified_correct_response, usage

    except Exception as e:
        logger.error(
            "Stage 3.5 FJ verification failed — keeping original",
            error=str(e)[:200],
        )
        return None, {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0,
            "latency_ms": 0,
        }
