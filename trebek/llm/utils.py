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
) -> "tuple[_T, dict[str, float], int]":
    """
    Core extraction primitive using Native Structured Outputs with CoT scratchpad.
    Relies on Gemini API to enforce JSON schema natively, with a retry loop for
    validation failures. Always uses the full 65536 output token budget.

    On validation failures, attempts a cheap Flash repair before burning a full
    retry cycle. The repair does not count as a retry attempt.
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
            )

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                output_tokens_used = int(
                    getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                )
            else:
                output_tokens_used = 0
            logger.info(
                "LLM extraction completed",
                context=ctx,
                schema=schema_cls.__name__,
                attempt=attempt + 1,
                budget=max_output_tokens,
                output_tokens_used=output_tokens_used,
                budget_utilization_pct=round(output_tokens_used / max_output_tokens * 100, 1)
                if max_output_tokens > 0
                else 0,
                latency_ms=round(usage.get("latency_ms", 0), 0),
            )

            response_text = str(response.text)
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
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
