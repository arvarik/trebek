"""
Typed exception hierarchy for the Trebek pipeline.

Provides structured error types instead of bare ``RuntimeError`` / ``ValueError``
so that callers can make typed retry-vs-permanent-fail decisions and telemetry
can categorize failures by exception class.
"""


class TrebekError(Exception):
    """Base exception for all Trebek errors."""


class ConfigurationError(TrebekError):
    """Invalid or missing configuration (API keys, env vars, paths)."""


class PipelineError(TrebekError):
    """Error during pipeline stage execution."""

    def __init__(self, message: str, episode_id: str | None = None, stage: str | None = None) -> None:
        self.episode_id = episode_id
        self.stage = stage
        super().__init__(message)


class ExtractionError(PipelineError):
    """LLM extraction failed after all retries."""


class TranscriptionError(PipelineError):
    """GPU transcription (WhisperX/FFmpeg) failed."""


class DatabaseError(TrebekError):
    """Database operation failed (write, transaction, migration)."""


class SchemaValidationError(TrebekError):
    """Pydantic schema validation failed during LLM output parsing."""


class RetryableError(PipelineError):
    """Error that should trigger a retry (e.g., transient API failure)."""


class PermanentError(PipelineError):
    """Error that should NOT be retried (e.g., missing video file)."""
