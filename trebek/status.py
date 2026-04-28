"""
Pipeline status enumeration — single source of truth for all pipeline state values.

Uses ``StrEnum`` so values serialize naturally to SQLite TEXT columns while providing
IDE autocomplete, exhaustive matching, and typo prevention at the type level.
"""

from enum import StrEnum


class PipelineStatus(StrEnum):
    """Valid states for a pipeline_state row."""

    PENDING = "PENDING"
    TRANSCRIBING = "TRANSCRIBING"
    TRANSCRIPT_READY = "TRANSCRIPT_READY"
    CLEANED = "CLEANED"
    SAVING = "SAVING"
    MULTIMODAL_PROCESSING = "MULTIMODAL_PROCESSING"
    MULTIMODAL_DONE = "MULTIMODAL_DONE"
    VECTORIZING = "VECTORIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
