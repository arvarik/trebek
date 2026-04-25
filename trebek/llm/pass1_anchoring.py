import ast
import json
import structlog
from typing import Dict

from trebek.llm.client import _get_client

logger = structlog.get_logger()


async def execute_pass_1_speaker_anchoring(audio_file_path: str) -> "tuple[Dict[str, str], dict[str, float]]":
    """
    Pass 1: Fast Gemini 3.1 Flash extraction isolating the host interview.
    Generates a rigid speaker mapping dictionary via Native Audio Anchoring.
    """

    system_prompt = (
        "You are a strict data extractor. Analyze the audio slice of the Jeopardy host interview segment. "
        "Listen to the host interview the contestants. Map their distinct vocal timbres to Diarization Speaker IDs (like SPEAKER_00). "
        "Return a pure JSON dictionary mapping Diarization Speaker IDs to Contestant/Host Names. "
        'Format: {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt Amodio"}. '
        "Use double quotes only. Return ONLY the JSON object, no markdown."
    )

    try:
        client = _get_client()
        uploaded_file = await client.upload_file(audio_file_path)

        prompt = [
            "Listen to the host interview the contestants. Map their distinct vocal timbres to the generic SPEAKER_XX IDs.",
            uploaded_file,
        ]

        response, usage = await client.generate_content(
            model="gemini-3.1-flash-preview",
            prompt=prompt,
            system_instruction=system_prompt,
            max_output_tokens=2048,
        )

        await client.delete_file(uploaded_file.name)

        # Clean markdown fences if present
        response_text = str(response.text)
        clean = response_text.replace("```json", "").replace("```", "").strip()
        if not clean:
            logger.warning("Pass 1 returned empty response, using empty mapping")
            return {}, usage
        # Try json.loads first, fall back to ast.literal_eval for single-quoted dicts
        try:
            mapping: Dict[str, str] = json.loads(clean)
        except json.JSONDecodeError:
            mapping = dict(ast.literal_eval(clean))
        logger.info("Pass 1 Speaker Anchor resolved", mapping=mapping)
        return mapping, usage
    except Exception as e:
        logger.error("Failed to generate speaker anchor", error=str(e))
        # Fallback or default mapping logic here
        return {}, {"input_tokens": 0.0, "output_tokens": 0.0, "cached_tokens": 0.0, "latency_ms": 0.0}
