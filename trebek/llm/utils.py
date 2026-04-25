import asyncio
from typing import Optional, TypeVar
from pydantic import BaseModel

from trebek.llm.client import _get_client

_T = TypeVar("_T", bound=BaseModel)

# Gemini 3.1 Pro max output: 65,536 tokens.
# You only pay for tokens actually generated, NOT the budget ceiling.
# There is zero reason to cap below the maximum.
GEMINI_MAX_OUTPUT_TOKENS = 65536


async def _extract_part(
    prompt: str,
    system_prompt: str,
    schema_cls: type[_T],
    max_retries: int,
    max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
    cached_content_name: Optional[str] = None,
) -> "tuple[_T, dict[str, float], int]":
    """
    Core extraction primitive using Native Structured Outputs with CoT scratchpad.
    Relies on Gemini API to enforce JSON schema natively, with a retry loop for
    validation failures. Always uses the full 65536 output token budget.
    """
    import structlog
    logger = structlog.get_logger()
    client = _get_client()

    for attempt in range(max_retries + 1):
        try:
            response, usage = await client.generate_content(
                model="gemini-3.1-pro-preview",
                prompt=prompt,
                system_instruction=system_prompt,
                response_schema=schema_cls,
                max_output_tokens=max_output_tokens,
                cached_content_name=cached_content_name,
            )

            output_tokens_used = int(usage.get("output_tokens", 0))
            logger.info(
                "LLM call completed",
                schema=schema_cls.__name__,
                attempt=attempt + 1,
                budget=max_output_tokens,
                output_tokens_used=output_tokens_used,
                budget_utilization_pct=round(output_tokens_used / max_output_tokens * 100, 1) if max_output_tokens > 0 else 0,
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
            backoff_delay = 2.0 * (2 ** attempt)  # 2s, 4s, 8s, 16s
            logger.warning(
                "Extraction validation failed, retrying",
                schema=schema_cls.__name__,
                attempt=attempt + 1,
                error=str(e)[:200],
                backoff_s=backoff_delay,
            )
            await asyncio.sleep(backoff_delay)

    raise RuntimeError("Unreachable")
