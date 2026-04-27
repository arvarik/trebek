"""
Analysis package — post-extraction analytical features.

Contains modules for:
- **vision**: Gemini Vision API client and podium light detection
- **buzzer**: Reaction time physics and acoustic confidence metrics
- **embeddings**: Cosine distance and semantic lateral distance
"""

from trebek.analysis.vision import VisionClient, extract_podium_illumination_timestamp
from trebek.analysis.buzzer import (
    calculate_true_buzzer_latency,
    calculate_true_acoustic_metrics,
    WhisperXWordSegment,
)
from trebek.analysis.embeddings import cosine_distance, process_semantic_lateral_distance

__all__ = [
    "VisionClient",
    "extract_podium_illumination_timestamp",
    "calculate_true_buzzer_latency",
    "calculate_true_acoustic_metrics",
    "WhisperXWordSegment",
    "cosine_distance",
    "process_semantic_lateral_distance",
]
