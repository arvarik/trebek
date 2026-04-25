import json
import structlog
import asyncio
from typing import Dict, Optional
from pydantic import ValidationError
from schemas import Episode

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

    async def generate_content(self, model: str, prompt: str, system_instruction: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
        )
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return str(response.text)


async def execute_pass_1_speaker_anchoring(host_interview_segment: str) -> Dict[str, str]:
    """
    Pass 1: Fast Gemini Flash extraction isolating the host interview.
    Generates a rigid speaker mapping dictionary to prevent hallucinations.
    """
    system_prompt = (
        "You are a strict data extractor. Analyze the Jeopardy host interview segment. "
        "Return a pure JSON dictionary mapping Diarization Speaker IDs to Contestant/Host Names. "
        "Format: {'SPEAKER_00': 'Ken Jennings', 'SPEAKER_01': 'Matt Amodio'}."
    )

    try:
        client = _get_client()
        response_text = await client.generate_content(
            model="gemini-1.5-flash",
            prompt=f"Segment:\n{host_interview_segment}",
            system_instruction=system_prompt,
        )
        mapping: Dict[str, str] = json.loads(response_text)
        logger.info("Pass 1 Speaker Anchor resolved", mapping=mapping)
        return mapping
    except Exception as e:
        logger.error("Failed to generate speaker anchor", error=str(e))
        # Fallback or default mapping logic here
        return {}


async def execute_pass_2_data_extraction(
    full_transcript: str, speaker_mapping: Dict[str, str], max_retries: int = 2
) -> Episode:
    """
    Pass 2: Massive structured extraction (Gemini 1.5 Pro).
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

    for attempt in range(max_retries + 1):
        logger.info("Stage 5 Extraction Attempt", attempt=attempt + 1, max_attempts=max_retries + 1)

        try:
            client = _get_client()
            response_text = await client.generate_content(
                model="gemini-1.5-pro", prompt=current_prompt, system_instruction=system_prompt
            )

            clean_json = response_text.replace("```json", "").replace("```", "").strip()

            # Offloads CPU-bound JSON schema validation
            validated_data = await asyncio.to_thread(Episode.model_validate_json, clean_json)
            logger.info("Stage 5 schema validation successful.")
            return validated_data

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
