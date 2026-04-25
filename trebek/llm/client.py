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

    async def generate_content(
        self,
        model: str,
        prompt: Any,
        system_instruction: str,
        response_schema: Optional[type[BaseModel]] = None,
        max_output_tokens: int = 65536,
        cached_content_name: Optional[str] = None,
    ) -> "tuple[Any, dict[str, float]]":
        import re
        import random
        import time
        from google.genai import types

        kwargs: dict[str, Any] = {"temperature": 0.0, "max_output_tokens": max_output_tokens}
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
        base_delay = 5.0
        for attempt in range(max_retries + 1):
            try:
                start_t = time.perf_counter()
                response = await self.client.aio.models.generate_content(model=model, contents=prompt, config=config)
                latency_ms = (time.perf_counter() - start_t) * 1000
                usage = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = {
                        "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0.0,
                        "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0.0,
                        "cached_tokens": getattr(response.usage_metadata, "cached_content_token_count", 0) or 0.0,
                        "latency_ms": latency_ms,
                    }
                else:
                    usage = {"input_tokens": 0.0, "output_tokens": 0.0, "cached_tokens": 0.0, "latency_ms": latency_ms}
                return response, usage
            except Exception as e:
                err_str = str(e)
                if (
                    not (
                        "429" in err_str
                        or "RESOURCE_EXHAUSTED" in err_str
                        or "500" in err_str
                        or "503" in err_str
                        or "UNAVAILABLE" in err_str
                    )
                    or attempt == max_retries
                ):
                    raise
                retry_match = re.search(r"retry.*?(\d+\.?\d*)\s*s", err_str, re.IGNORECASE)
                if retry_match:
                    delay = float(retry_match.group(1)) + random.uniform(1.0, 3.0)
                else:
                    delay = min(base_delay * (2**attempt), 120.0) + random.uniform(0.5, 2.0)
                logger.warning("Rate limited, retrying", attempt=attempt + 1, delay_s=round(delay, 1), model=model)
                import asyncio

                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable")
