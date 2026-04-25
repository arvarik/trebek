import asyncio
from typing import Optional, TypeVar
from pydantic import BaseModel

from trebek.llm.client import _get_client

_T = TypeVar("_T", bound=BaseModel)


def _estimate_output_tokens(num_input_lines: int, extraction_type: str) -> int:
    """
    Output token budget. Generous ceilings are free — you only pay for tokens
    actually generated. Being too conservative causes truncation, which triggers
    retries that cost MORE in the end.

    Gemini 3.1 Pro max: 65,536 output tokens.
    """
    if extraction_type == "meta":
        return 16384
    elif extraction_type == "skeleton":
        return 4096
    elif extraction_type == "clues":
        # Each transcript segment is 20-30s of dense, multi-speaker dialogue.
        # A chunk of 20 segments can contain 20-30 clues, each needing ~400 tokens,
        # plus the reasoning_scratchpad at ~3000-5000 tokens.
        # Use 32768 as floor — it's free headroom.
        return 32768
    return 32768


async def _extract_part(
    prompt: str,
    system_prompt: str,
    schema_cls: type[_T],
    max_retries: int,
    max_output_tokens: int = 65536,
    cached_content_name: Optional[str] = None,
) -> "tuple[_T, dict[str, float], int]":
    """
    Core extraction primitive using Native Structured Outputs with CoT scratchpad.
    Relies on Gemini API to enforce JSON schema natively, with a retry loop for truncation.

    Retry strategy: each retry increases output token budget by 50% and uses
    exponential backoff to handle both truncation and rate limiting.
    """
    import structlog
    logger = structlog.get_logger()
    client = _get_client()

    current_budget = max_output_tokens
    for attempt in range(max_retries + 1):
        try:
            response, usage = await client.generate_content(
                model="gemini-3.1-pro-preview",
                prompt=prompt,
                system_instruction=system_prompt,
                response_schema=schema_cls,
                max_output_tokens=current_budget,
                cached_content_name=cached_content_name,
            )

            output_tokens_used = int(usage.get("output_tokens", 0))
            logger.info(
                "LLM call completed",
                schema=schema_cls.__name__,
                attempt=attempt + 1,
                budget=current_budget,
                output_tokens_used=output_tokens_used,
                budget_utilization_pct=round(output_tokens_used / current_budget * 100, 1) if current_budget > 0 else 0,
                latency_ms=round(usage.get("latency_ms", 0), 0),
            )

            if hasattr(response, "parsed") and response.parsed:
                return response.parsed, usage, attempt

            response_text = str(response.text)
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            validated_data = await asyncio.to_thread(schema_cls.model_validate_json, clean_json)
            return validated_data, usage, attempt
        except Exception as e:
            if attempt == max_retries:
                raise
            # Escalate output token budget by 50% on each retry (capped at 65536)
            # to give the model more room if truncation is the root cause
            current_budget = min(int(current_budget * 1.5), 65536)
            backoff_delay = 2.0 * (2 ** attempt)  # 2s, 4s, 8s, 16s
            logger.warning(
                "Extraction validation failed, retrying",
                schema=schema_cls.__name__,
                attempt=attempt + 1,
                error=str(e)[:200],
                next_budget=current_budget,
                backoff_s=backoff_delay,
            )
            await asyncio.sleep(backoff_delay)

    raise RuntimeError("Unreachable")
