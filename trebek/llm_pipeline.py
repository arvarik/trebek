import json
import structlog
import asyncio
from typing import Dict, Optional
from pydantic import ValidationError
from trebek.schemas import Episode

logger = structlog.get_logger()

_client: Optional["GeminiClient"] = None


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
        self, model: str, prompt: str, system_instruction: str
    ) -> "tuple[str, dict[str, float]]":
        from google.genai import types
        import time

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
        )
        start_t = time.perf_counter()
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        latency_ms = (time.perf_counter() - start_t) * 1000

        usage = {}
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


async def execute_pass_1_speaker_anchoring(host_interview_segment: str) -> "tuple[Dict[str, str], dict[str, float]]":
    """
    Pass 1: Fast Gemini 3.1 Flash-Lite extraction isolating the host interview.
    Generates a rigid speaker mapping dictionary to prevent hallucinations.
    """
    system_prompt = (
        "You are a strict data extractor. Analyze the Jeopardy host interview segment. "
        "Return a pure JSON dictionary mapping Diarization Speaker IDs to Contestant/Host Names. "
        "Format: {'SPEAKER_00': 'Ken Jennings', 'SPEAKER_01': 'Matt Amodio'}."
    )

    try:
        client = _get_client()
        response_text, usage = await client.generate_content(
            model="gemini-3.1-flash-lite",
            prompt=f"Segment:\n{host_interview_segment}",
            system_instruction=system_prompt,
        )
        mapping: Dict[str, str] = json.loads(response_text)
        logger.info("Pass 1 Speaker Anchor resolved", mapping=mapping)
        return mapping, usage
    except Exception as e:
        logger.error("Failed to generate speaker anchor", error=str(e))
        # Fallback or default mapping logic here
        return {}, {"input_tokens": 0.0, "output_tokens": 0.0, "cached_tokens": 0.0, "latency_ms": 0.0}


async def execute_pass_2_data_extraction(
    full_transcript: str, speaker_mapping: Dict[str, str], max_retries: int = 2
) -> "tuple[Episode, dict[str, float], int]":
    """
    Pass 2: Massive structured extraction (Gemini 3.1 Pro).
    Injects the locked speaker mapping and implements a Pydantic self-healing retry loop.
    """
    system_prompt = (
        "You are Trebek, an expert data extraction pipeline. Extract the game events into the provided JSON schema. "
        f"CRITICAL CONSTRAINT: You MUST map speakers using this exact reference dictionary: {json.dumps(speaker_mapping)}. "
        "Do not hallucinate names outside of this mapping. "
        "Do NOT perform any running score math. Just extract the facts."
    )

    base_prompt = f"Transcript Data:\n{full_transcript}\n\nOutput strict JSON matching the Episode schema."
    current_prompt = base_prompt

    total_usage: dict[str, float] = {"input_tokens": 0.0, "output_tokens": 0.0, "cached_tokens": 0.0, "latency_ms": 0.0}

    for attempt in range(max_retries + 1):
        logger.info("Stage 5 Extraction Attempt", attempt=attempt + 1, max_attempts=max_retries + 1)

        try:
            client = _get_client()
            response_text, usage = await client.generate_content(
                model="gemini-3.1-pro", prompt=current_prompt, system_instruction=system_prompt
            )

            total_usage["input_tokens"] += usage.get("input_tokens", 0.0)
            total_usage["output_tokens"] += usage.get("output_tokens", 0.0)
            total_usage["cached_tokens"] += usage.get("cached_tokens", 0.0)
            total_usage["latency_ms"] += usage.get("latency_ms", 0.0)

            clean_json = response_text.replace("```json", "").replace("```", "").strip()

            # Offloads CPU-bound JSON schema validation
            validated_data = await asyncio.to_thread(Episode.model_validate_json, clean_json)
            logger.info("Stage 5 schema validation successful.")
            return validated_data, total_usage, attempt

        except ValidationError as validation_error:
            logger.warning(
                "Pydantic Validation failed",
                attempt=attempt + 1,
                error_count=validation_error.error_count(),
            )

            if attempt == max_retries:
                logger.error("Max LLM self-healing retries exhausted. Dropping payload.")
                raise validation_error

            error_details = str(validation_error)
            current_prompt = (
                base_prompt + "\n\n--- CRITICAL ERROR IN PREVIOUS RESPONSE ---\n"
                "Your previous JSON output failed system validation. "
                "You must fix the exact errors listed below and return the corrected JSON.\n"
                f"VALIDATION ERRORS:\n{error_details}"
            )

    raise RuntimeError("Unexpected escape from Stage 5 self-healing loop.")
