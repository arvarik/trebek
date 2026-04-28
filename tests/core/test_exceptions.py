"""
Tests for the typed exception hierarchy.
"""

import pytest

from trebek.exceptions import (
    TrebekError,
    ConfigurationError,
    PipelineError,
    ExtractionError,
    TranscriptionError,
    DatabaseError,
    SchemaValidationError,
    RetryableError,
    PermanentError,
)


class TestExceptionHierarchy:
    """Verify the inheritance chain enables typed except clauses."""

    def test_all_exceptions_inherit_from_trebek_error(self) -> None:
        for exc_cls in [
            ConfigurationError,
            PipelineError,
            ExtractionError,
            TranscriptionError,
            DatabaseError,
            SchemaValidationError,
            RetryableError,
            PermanentError,
        ]:
            assert issubclass(exc_cls, TrebekError)

    def test_pipeline_error_carries_context(self) -> None:
        err = PipelineError("test error", episode_id="ep123", stage="extract")
        assert err.episode_id == "ep123"
        assert err.stage == "extract"
        assert str(err) == "test error"

    def test_extraction_error_is_pipeline_error(self) -> None:
        err = ExtractionError("LLM failed", episode_id="ep456")
        assert isinstance(err, PipelineError)
        assert isinstance(err, TrebekError)

    def test_retryable_vs_permanent_distinction(self) -> None:
        """Code should be able to distinguish retryable from permanent errors."""
        retryable = RetryableError("rate limited", episode_id="ep789")
        permanent = PermanentError("invalid file", episode_id="ep000")

        assert isinstance(retryable, PipelineError)
        assert isinstance(permanent, PipelineError)
        assert not isinstance(retryable, PermanentError)
        assert not isinstance(permanent, RetryableError)

    def test_except_trebek_error_catches_all(self) -> None:
        """A single except TrebekError should catch any Trebek exception."""
        with pytest.raises(TrebekError):
            raise ExtractionError("test")

        with pytest.raises(TrebekError):
            raise ConfigurationError("test")

        with pytest.raises(TrebekError):
            raise DatabaseError("test")
