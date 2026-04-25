import math
import structlog
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

logger = structlog.get_logger()

_vision_client: Optional["VisionClient"] = None


def _get_vision_client() -> "VisionClient":
    """Lazy-initializes the Vision client on first use, not at import time."""
    global _vision_client
    if _vision_client is None:
        _vision_client = VisionClient()
    return _vision_client


class VisionClient:
    """Interface for Gemini 3.1 Pro via the Files API."""

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

    async def analyze_video(self, video_path: str, prompt: str) -> float:
        """
        Uploads the video via the Gemini Files API, then analyzes it with the
        specified prompt. Returns a float timestamp extracted from the response.

        NOTE: The Gemini Files API requires uploading the video first, then passing
        the file reference to the model. For very large files, consider chunking or
        pre-extracting the relevant video segment with FFmpeg before upload.
        """
        from google.genai import types

        # Upload the video file via the Files API
        # This is a synchronous upload; for production, consider async wrapper
        uploaded_file = self.client.files.upload(file=video_path)
        logger.info("Uploaded video for Vision analysis", video_path=video_path, file_name=uploaded_file.name)

        config = types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0.0,
        )
        response = await self.client.aio.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=uploaded_file,
            config=config,
        )
        return float(str(response.text).strip())


async def extract_podium_illumination_timestamp(video_path: str, host_finish_timestamp: float) -> float:
    """
    Stage 6: Analyzes video frames immediately following the host's completion
    to determine the exact frame the podium lockout system disengages.
    """
    system_prompt = (
        f"Analyze the video starting precisely at {host_finish_timestamp} seconds. "
        "Find the exact timestamp (in seconds) where the contestant podium indicator "
        "lights illuminate, signaling the lockout system has disengaged. Return ONLY a float."
    )
    try:
        client = _get_vision_client()
        response = await client.analyze_video(video_path, prompt=system_prompt)
        return float(response)
    except Exception as e:
        logger.error("Vision model failed", error=str(e))
        return host_finish_timestamp + 0.1  # Fallback estimation


def calculate_true_buzzer_latency(buzz_timestamp: float, podium_light_timestamp: float) -> float:
    """
    Calculates the true physical reaction time, removing the host's cadence variance.
    """
    return round(buzz_timestamp - podium_light_timestamp, 3)


class WhisperXWordSegment(BaseModel):
    word: str
    start: float
    end: float
    prob: float


def calculate_true_acoustic_metrics(
    buzz_start_time: float, buzz_end_time: float, whisper_words: List[WhisperXWordSegment]
) -> Dict[str, Any]:
    """
    Cross-references LLM semantic boundaries with raw WhisperX word-level `.prob`
    logprobs to calculate true acoustic confidence and deterministic disfluency counts.
    """
    relevant_probs: List[float] = []
    disfluency_count = 0
    disfluency_markers = {"um", "uh", "er", "ah", "hmm"}

    for segment in whisper_words:
        if segment.start >= buzz_start_time and segment.end <= buzz_end_time:
            relevant_probs.append(segment.prob)
            cleaned_word = segment.word.lower().strip(",.?!")
            if cleaned_word in disfluency_markers:
                disfluency_count += 1

    if not relevant_probs:
        return {"true_acoustic_confidence_score": 0.0, "disfluency_count": 0}

    avg_confidence = sum(relevant_probs) / len(relevant_probs)
    return {"true_acoustic_confidence_score": round(avg_confidence, 4), "disfluency_count": disfluency_count}


def cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculates the cosine distance between two floating-point vectors.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Embeddings must have the same dimensionality.")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0  # Max distance if one vector is empty

    similarity = dot_product / (norm_a * norm_b)
    return max(0.0, min(1.0, 1.0 - similarity))


def process_semantic_lateral_distance(clue_embedding: List[float], response_embedding: List[float]) -> float:
    """
    Calculates the lateral semantic distance between a clue and the correct response.
    High distance = wordplay/lateral thinking. Low distance = direct factual recall.
    """
    distance = cosine_distance(clue_embedding, response_embedding)
    logger.info("Calculated Semantic Lateral Distance", distance=round(distance, 4))
    return distance
