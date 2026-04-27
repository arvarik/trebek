"""
LLM package — Gemini API-based extraction passes for Jeopardy! transcripts.

Three-pass architecture:
    - **Pass 1** (``pass1_anchoring``): Speaker diarization mapping via Flash
    - **Pass 2** (``pass2_extraction``): Full structured data extraction via Pro
    - **Pass 3** (``pass3_multimodal``): Vision-based temporal sniping via Pro

Supporting modules:
    - ``client`` — Gemini API wrapper with retry logic
    - ``chunking`` — Semantic round-boundary transcript splitter
    - ``schemas`` — Intermediate extraction Pydantic schemas
    - ``speaker_normalization`` — Post-extraction speaker name resolution
    - ``transcript`` — Transcript formatting and compression utilities
    - ``utils`` — Core extraction primitive with schema validation
    - ``validation`` — Integrity checks and deduplication
"""

from .pass1_anchoring import execute_pass_1_speaker_anchoring
from .pass2_extraction import execute_pass_2_data_extraction
from .pass3_multimodal import execute_pass_3_multimodal_augmentation

__all__ = [
    "execute_pass_1_speaker_anchoring",
    "execute_pass_2_data_extraction",
    "execute_pass_3_multimodal_augmentation",
]
