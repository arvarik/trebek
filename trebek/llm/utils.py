import asyncio
from typing import Optional, TypeVar
from pydantic import BaseModel, ValidationError

from trebek.llm.client import _get_client
from trebek.config import MODEL_PRO, MODEL_FLASH

_T = TypeVar("_T", bound=BaseModel)

# Gemini 3.1 Pro max output: 65,536 tokens.
# You only pay for tokens actually generated, NOT the budget ceiling.
# There is zero reason to cap below the maximum.
GEMINI_MAX_OUTPUT_TOKENS = 65536


def _try_close_truncated_json(json_str: str) -> str:
    """
    Attempt to recover truncated JSON by rolling back to the last complete
    element and closing unmatched brackets/braces.

    When the model hits MAX_TOKENS, the JSON output is cut mid-generation.

    Strategy:
    1. If we can find a last complete `}` or `]`, truncate to that point.
       This drops the partial trailing element cleanly.
    2. If no complete sub-elements exist (e.g., truncation inside the first
       string value), fall back to closing the string, stripping any partial
       key-value, and closing all open structures.

    The resulting data will be missing trailing elements, but the ones that
    were fully serialized before truncation will be recoverable.
    """
    import structlog
    import re

    logger = structlog.get_logger()
    stripped = json_str.rstrip()

    # Already looks complete
    if stripped.endswith("}"):
        return json_str

    logger.info("Attempting truncated JSON recovery", original_length=len(json_str))

    # Pass 1: Walk the string tracking state
    in_string = False
    escaped = False
    open_braces = 0
    open_brackets = 0

    # Positions right after each complete } or ] (safe rollback points)
    complete_element_positions: list[int] = []

    for i, ch in enumerate(stripped):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if in_string:
            if ch == '"':
                in_string = False
            continue
        # Outside string
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            open_braces += 1
        elif ch == "}":
            open_braces -= 1
            complete_element_positions.append(i + 1)
        elif ch == "[":
            open_brackets += 1
        elif ch == "]":
            open_brackets -= 1
            complete_element_positions.append(i + 1)

    # Decide recovery strategy
    if complete_element_positions:
        cut_pos = complete_element_positions[-1]
        if cut_pos < len(stripped):
            # Strategy 1: Roll back to last complete element
            truncated = stripped[:cut_pos]
            strategy = "rollback"
        else:
            # Last complete element IS the end — nothing to truncate
            truncated = stripped
            strategy = "no_truncation"
    else:
        # Strategy 2: No complete sub-elements — close string + strip partials
        truncated = stripped
        if in_string:
            truncated += '"'
        strategy = "close_and_strip"

    # Strip trailing partial key-values and commas
    truncated = re.sub(r',\s*"[^"]*"\s*:\s*$', "", truncated)
    truncated = re.sub(r",\s*$", "", truncated)

    # Recount open structures after truncation using a stack (preserves LIFO order)
    in_str = False
    esc = False
    stack: list[str] = []
    for ch in truncated:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            stack.append("{")
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "[":
            stack.append("[")
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    # Close unmatched structures in reverse order (LIFO)
    closing_chars = []
    for opener in reversed(stack):
        if opener == "{":
            closing_chars.append("}")
        elif opener == "[":
            closing_chars.append("]")
    truncated += "".join(closing_chars)

    brackets_closed = closing_chars.count("]")
    braces_closed = closing_chars.count("}")

    logger.info(
        "Truncated JSON recovery complete",
        strategy=strategy,
        recovered_length=len(truncated),
        was_in_string=in_string,
        brackets_closed=brackets_closed,
        braces_closed=braces_closed,
    )
    return truncated


async def _attempt_flash_repair(
    broken_json: str,
    error_msg: str,
    schema_cls: type[_T],
    invocation_context: str,
) -> "Optional[tuple[_T, dict[str, float]]]":
    """
    Attempts to repair broken JSON using the cheap Flash model.

    Sends the broken JSON string and the validation error to gemini-3.1-flash-lite-preview
    with a targeted repair prompt. If the repaired output passes Pydantic validation,
    returns the validated object and usage stats. Returns None on any failure,
    allowing the caller to fall through to a full retry.

    Cost: ~100x cheaper than re-running the full extraction prompt on Pro.
    """
    import structlog

    logger = structlog.get_logger()
    client = _get_client()

    repair_prompt = (
        "The following JSON failed schema validation. Fix it to produce valid JSON "
        "that resolves the error below. Output ONLY the corrected JSON, nothing else.\n\n"
        f"Error: {error_msg[:500]}\n\n"
        f"Broken JSON:\n{broken_json}"
    )

    try:
        response, usage = await client.generate_content(
            model=MODEL_FLASH,
            prompt=repair_prompt,
            system_instruction="You are a JSON repair tool. Output ONLY valid, corrected JSON.",
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            invocation_context=f"{invocation_context} (Flash repair)",
        )

        response_text = str(response.text)
        clean_json = response_text.replace("```json", "").replace("```", "").strip()
        repaired = await asyncio.to_thread(schema_cls.model_validate_json, clean_json)

        logger.info(
            "Flash repair succeeded",
            context=invocation_context,
            schema=schema_cls.__name__,
            repair_input_tokens=int(usage.get("input_tokens", 0)),
            repair_output_tokens=int(usage.get("output_tokens", 0)),
            repair_thinking_tokens=int(usage.get("thinking_tokens", 0)),
            repair_cost_usd=round(usage.get("cost_usd", 0.0), 6),
        )
        return repaired, usage
    except Exception as repair_err:
        logger.warning(
            "Flash repair failed, falling through to full retry",
            context=invocation_context,
            schema=schema_cls.__name__,
            repair_error=str(repair_err)[:200],
        )
        return None


async def _extract_part(
    prompt: str,
    system_prompt: str,
    schema_cls: type[_T],
    max_retries: int,
    model: str = MODEL_PRO,
    max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
    cached_content_name: Optional[str] = None,
    invocation_context: str = "",
    thinking_level: Optional[str] = None,
) -> "tuple[_T, dict[str, float], int]":
    """
    Core extraction primitive using Native Structured Outputs.
    Relies on Gemini API to enforce JSON schema natively, with a retry loop for
    validation failures. Always uses the full 65536 output token budget.

    On validation failures, attempts a cheap Flash repair before burning a full
    retry cycle. The repair does not count as a retry attempt.

    Args:
        thinking_level: Override the default thinking level ("low") for structured
            output. Use "medium" or "high" for tasks requiring genuine reasoning
            (e.g., episode meta extraction).
    """
    import structlog

    logger = structlog.get_logger()
    client = _get_client()

    ctx = invocation_context or schema_cls.__name__

    for attempt in range(max_retries + 1):
        try:
            response, usage = await client.generate_content(
                model=model,
                prompt=prompt,
                system_instruction=system_prompt,
                response_schema=schema_cls,
                max_output_tokens=max_output_tokens,
                cached_content_name=cached_content_name,
                invocation_context=f"{ctx} (schema validation attempt {attempt + 1}/{max_retries + 1})",
                thinking_level=thinking_level,
            )

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                output_tokens_used = int(getattr(response.usage_metadata, "candidates_token_count", 0) or 0)
                thinking_tokens_used = int(getattr(response.usage_metadata, "thoughts_token_count", 0) or 0)
            else:
                output_tokens_used = 0
                thinking_tokens_used = 0
            total_gen_tokens = output_tokens_used + thinking_tokens_used
            logger.info(
                "LLM extraction completed",
                context=ctx,
                schema=schema_cls.__name__,
                attempt=attempt + 1,
                budget=max_output_tokens,
                output_tokens_used=output_tokens_used,
                thinking_tokens_used=thinking_tokens_used,
                true_utilization_pct=round(total_gen_tokens / max_output_tokens * 100, 1)
                if max_output_tokens > 0
                else 0,
                latency_ms=round(usage.get("latency_ms", 0), 0),
            )

            response_text = response.text
            if response_text is None:
                raise ValueError(
                    f"Empty response from API (model may have hit MAX_TOKENS). Context: {ctx}, attempt: {attempt + 1}"
                )
            clean_json = str(response_text).replace("```json", "").replace("```", "").strip()

            # Safety net: if the JSON is truncated (MAX_TOKENS), attempt recovery
            if clean_json and not clean_json.rstrip().endswith("}"):
                clean_json = _try_close_truncated_json(clean_json)

            validated_data = await asyncio.to_thread(schema_cls.model_validate_json, clean_json)
            return validated_data, usage, attempt
        except (ValidationError, ValueError) as e:
            # Attempt cheap Flash repair before burning a full retry
            if attempt < max_retries:
                broken_text = ""
                try:
                    broken_text = str(response.text).replace("```json", "").replace("```", "").strip()
                except Exception:
                    pass

                if broken_text:
                    logger.warning(
                        "Schema validation failed, attempting Flash repair",
                        context=ctx,
                        schema=schema_cls.__name__,
                        response_length=len(broken_text),
                        finish_reason=str(
                            getattr(getattr(response, "candidates", [None])[0], "finish_reason", "UNKNOWN")
                        )
                        if hasattr(response, "candidates") and response.candidates
                        else "UNKNOWN",
                        error=str(e)[:500],
                    )
                    repair_result = await _attempt_flash_repair(broken_text, str(e), schema_cls, ctx)
                    if repair_result is not None:
                        repaired_data, repair_usage = repair_result
                        # Merge repair usage into the original usage
                        for k in usage:
                            usage[k] = usage.get(k, 0.0) + repair_usage.get(k, 0.0)
                        return repaired_data, usage, attempt

            # Flash repair failed or no broken text — fall through to full retry
            if attempt == max_retries:
                logger.error(
                    "LLM extraction failed after all retries",
                    context=ctx,
                    schema=schema_cls.__name__,
                    attempts_exhausted=max_retries + 1,
                    error=str(e)[:200],
                )
                raise
            backoff_delay = 2.0 * (2**attempt)  # 2s, 4s, 8s, 16s
            logger.warning(
                "Extraction validation failed, retrying",
                context=ctx,
                schema=schema_cls.__name__,
                attempt=attempt + 1,
                error=str(e)[:200],
                backoff_s=backoff_delay,
            )
            await asyncio.sleep(backoff_delay)
        except Exception as e:
            if attempt == max_retries:
                logger.error(
                    "LLM extraction failed after all retries",
                    context=ctx,
                    schema=schema_cls.__name__,
                    attempts_exhausted=max_retries + 1,
                    error=str(e)[:200],
                )
                raise
            backoff_delay = 2.0 * (2**attempt)  # 2s, 4s, 8s, 16s
            logger.warning(
                "Extraction validation failed, retrying",
                context=ctx,
                schema=schema_cls.__name__,
                attempt=attempt + 1,
                error=str(e)[:200],
                backoff_s=backoff_delay,
            )
            await asyncio.sleep(backoff_delay)

    raise RuntimeError("Unreachable")
