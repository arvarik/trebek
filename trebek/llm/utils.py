import asyncio
from typing import Optional, TypeVar
from pydantic import BaseModel

from trebek.llm.client import _get_client

_T = TypeVar("_T", bound=BaseModel)


def _estimate_output_tokens(num_input_lines: int, extraction_type: str) -> int:
    """
    Conservative output token budget estimation.
    Prevents both truncation (too low) and wasted quota (too high).
    """
    if extraction_type == "meta":
        return 8192  # Metadata is compact
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
    Core extraction primitive using Native Structured Outputs with CoT scratchpad.
    Relies on Gemini API to enforce JSON schema natively, with a retry loop for truncation.
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

            if hasattr(response, "parsed") and response.parsed:
                return response.parsed, usage, attempt

            response_text = str(response.text)
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            validated_data = await asyncio.to_thread(schema_cls.model_validate_json, clean_json)
            return validated_data, usage, attempt
        except Exception as e:
            if attempt == max_retries:
                raise
            logger.warning("Extraction validation failed, retrying", attempt=attempt + 1, error=str(e))
            await asyncio.sleep(2.0)
            
    raise RuntimeError("Unreachable")
