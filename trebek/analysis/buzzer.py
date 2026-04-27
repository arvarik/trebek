"""
Buzzer physics and acoustic confidence metrics.

Calculates true physical reaction times by subtracting host cadence variance,
and cross-references WhisperX word-level logprobs for acoustic confidence
scoring and disfluency detection.
"""

import structlog
from typing import Any, Dict, List

from pydantic import BaseModel

logger = structlog.get_logger()


class WhisperXWordSegment(BaseModel):
    """A single word-level segment from WhisperX output with timing and confidence."""
    word: str
    start: float
    end: float
    prob: float


def calculate_true_buzzer_latency(buzz_timestamp: float, podium_light_timestamp: float) -> float:
    """
    Calculates the true physical reaction time, removing the host's cadence variance.
    """
    return round(buzz_timestamp - podium_light_timestamp, 3)


def calculate_true_acoustic_metrics(
    buzz_start_time: float, buzz_end_time: float, whisper_words: List[WhisperXWordSegment]
) -> Dict[str, Any]:
    """
    Cross-references LLM semantic boundaries with raw WhisperX word-level ``.prob``
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
