import structlog
import asyncio
from typing import Any, Optional
from pydantic import BaseModel

logger = structlog.get_logger()

_client: Optional["GeminiClient"] = None


def _get_client() -> "GeminiClient":
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
            raise RuntimeError("GEMINI_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)

    async def upload_file(self, file_path: str) -> Any:
        return await asyncio.to_thread(self.client.files.upload, file=file_path)

    async def delete_file(self, file_name: str) -> None:
        await asyncio.to_thread(self.client.files.delete, name=file_name)

    # ── Context Caching Lifecycle ────────────────────────────────────

    async def create_cache(
        self,
        model: str,
        system_instruction: str,
        contents: list[Any],
        ttl: str = "1800s",
    ) -> Optional[str]:
        """
        Creates a Gemini cached content object for the given model, system prompt,
        and contents. Returns the cache name on success, or None if caching cannot
        be used (e.g. content below minimum token threshold).

        TTL defaults to 30 minutes — long enough for a full episode extraction.
        """
        from google.genai import types

        try:
            logger.info(
                "Gemini cache: creating",
                model=model,
                ttl=ttl,
                content_parts=len(contents),
            )
            cache = await asyncio.to_thread(
                self.client.caches.create,
                model=model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    contents=contents,
                    ttl=ttl,
                ),
            )
            cached_token_count = getattr(getattr(cache, "usage_metadata", None), "total_token_count", "unknown")
            logger.info(
                "Gemini cache: created successfully",
                cache_name=cache.name,
                cached_tokens=cached_token_count,
            )
            return str(cache.name)
        except Exception as e:
            err_str = str(e)
            if "too few tokens" in err_str.lower() or "minimum" in err_str.lower():
                logger.info(
                    "Gemini cache: content below minimum token threshold, falling back to non-cached mode",
                    error=err_str[:200],
                )
            else:
                logger.warning(
                    "Gemini cache: creation failed, falling back to non-cached mode",
                    error=err_str[:200],
                )
            return None

    async def delete_cache(self, cache_name: str) -> None:
        """Deletes a previously created cached content object. Logs but does not raise on failure."""
        try:
            await asyncio.to_thread(self.client.caches.delete, name=cache_name)
            logger.info("Gemini cache: deleted", cache_name=cache_name)
        except Exception as e:
            logger.warning("Gemini cache: deletion failed", cache_name=cache_name, error=str(e)[:200])

    # ── Content Generation ───────────────────────────────────────────

    async def generate_content(
        self,
        model: str,
        prompt: Any,
        system_instruction: str,
        response_schema: Optional[type[BaseModel]] = None,
        max_output_tokens: int = 65536,
        cached_content_name: Optional[str] = None,
        invocation_context: str = "",
        thinking_level: Optional[str] = None,
    ) -> "tuple[Any, dict[str, float]]":
        """
        Generates content via the Gemini API with structured invocation logging.

        Args:
            invocation_context: A human-readable label for this call (e.g. "Pass 2 Meta",
                "Pass 2 Chunk 3/5") used in structured logs for traceability.
        """
        import re
        import random
        import time
        from google.genai import types

        kwargs: dict[str, Any] = {"temperature": 0.0, "max_output_tokens": max_output_tokens}
        if not cached_content_name:
            kwargs["system_instruction"] = system_instruction
        _VALID_THINKING_LEVELS = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
        if response_schema:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = response_schema.model_json_schema()
            # Thinking burns output tokens from the same max_output_tokens budget,
            # causing JSON truncation when the model defaults to "high" thinking.
            # Default to LOW for structured output, but allow callers to override
            # for tasks that genuinely need reasoning (e.g., meta extraction).
            if thinking_level:
                level_upper = thinking_level.upper()
                if level_upper not in _VALID_THINKING_LEVELS:
                    raise ValueError(
                        f"Invalid thinking_level '{thinking_level}'. Valid values: {sorted(_VALID_THINKING_LEVELS)}"
                    )
                level = getattr(types.ThinkingLevel, level_upper)
            else:
                level = types.ThinkingLevel.LOW
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        else:
            kwargs["response_mime_type"] = "application/json"
            # For non-schema calls, apply thinking config if caller explicitly requests it
            if thinking_level:
                level_upper = thinking_level.upper()
                if level_upper not in _VALID_THINKING_LEVELS:
                    raise ValueError(
                        f"Invalid thinking_level '{thinking_level}'. Valid values: {sorted(_VALID_THINKING_LEVELS)}"
                    )
                level = getattr(types.ThinkingLevel, level_upper)
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        if cached_content_name:
            kwargs["cached_content"] = cached_content_name
        config = types.GenerateContentConfig(**kwargs)

        # Resolve effective thinking level for logging
        effective_thinking = thinking_level.upper() if thinking_level else ("LOW" if response_schema else "DEFAULT")

        ctx = invocation_context or "unnamed"
        max_retries = 10
        base_delay = 5.0
        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    "Gemini API: invoking",
                    context=ctx,
                    model=model,
                    attempt=attempt + 1,
                    max_output_tokens=max_output_tokens,
                    thinking_level=effective_thinking,
                    has_schema=response_schema is not None,
                    schema_name=response_schema.__name__ if response_schema else None,
                    cached=cached_content_name is not None,
                    prompt_length=len(str(prompt)) if isinstance(prompt, str) else "multimodal",
                )
                start_t = time.perf_counter()
                response = await self.client.aio.models.generate_content(model=model, contents=prompt, config=config)
                latency_ms = (time.perf_counter() - start_t) * 1000
                from trebek.config import MODEL_PRICING

                usage = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = {
                        "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0.0,
                        "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0.0,
                        "thinking_tokens": getattr(response.usage_metadata, "thoughts_token_count", 0) or 0.0,
                        "cached_tokens": getattr(response.usage_metadata, "cached_content_token_count", 0) or 0.0,
                        "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0.0,
                        "latency_ms": latency_ms,
                    }
                else:
                    usage = {
                        "input_tokens": 0.0,
                        "output_tokens": 0.0,
                        "thinking_tokens": 0.0,
                        "cached_tokens": 0.0,
                        "total_tokens": 0.0,
                        "latency_ms": latency_ms,
                    }

                cost = 0.0
                pricing = MODEL_PRICING.get(model)
                if pricing:
                    # Thinking tokens are billed at the output token rate
                    billable_output = usage["output_tokens"] + usage["thinking_tokens"]
                    cost = (usage["input_tokens"] * pricing["input"] + billable_output * pricing["output"]) / 1_000_000
                usage["cost_usd"] = cost

                # Detect output truncation before returning
                finish_reason = None
                if response.candidates:
                    finish_reason = getattr(response.candidates[0], "finish_reason", None)
                if finish_reason and "MAX_TOKENS" in str(finish_reason):
                    logger.warning(
                        "Gemini API: response truncated (MAX_TOKENS)",
                        context=ctx,
                        model=model,
                        thinking_level=effective_thinking,
                        thinking_tokens=int(usage["thinking_tokens"]),
                        output_tokens=int(usage["output_tokens"]),
                        max_output_tokens=max_output_tokens,
                        total_tokens=int(usage["total_tokens"]),
                    )

                logger.info(
                    "Gemini API: success",
                    context=ctx,
                    model=model,
                    attempt=attempt + 1,
                    finish_reason=str(finish_reason) if finish_reason else "UNKNOWN",
                    input_tokens=int(usage["input_tokens"]),
                    output_tokens=int(usage["output_tokens"]),
                    thinking_tokens=int(usage["thinking_tokens"]),
                    cached_tokens=int(usage["cached_tokens"]),
                    cost_usd=round(usage["cost_usd"], 6),
                    latency_ms=round(latency_ms, 0),
                )
                return response, usage
            except Exception as e:
                err_str = str(e)
                is_retryable = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "500" in err_str
                    or "503" in err_str
                    or "UNAVAILABLE" in err_str
                )
                if not is_retryable or attempt == max_retries:
                    logger.error(
                        "Gemini API: failed (non-retryable or exhausted retries)",
                        context=ctx,
                        model=model,
                        attempt=attempt + 1,
                        error=err_str[:300],
                    )
                    raise
                retry_match = re.search(r"retry.*?(\d+\.?\d*)\s*s", err_str, re.IGNORECASE)
                if retry_match:
                    delay = float(retry_match.group(1)) + random.uniform(1.0, 3.0)
                else:
                    delay = min(base_delay * (2**attempt), 120.0) + random.uniform(0.5, 2.0)
                logger.warning(
                    "Gemini API: retryable error",
                    context=ctx,
                    model=model,
                    attempt=attempt + 1,
                    delay_s=round(delay, 1),
                    error=err_str[:200],
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable")
