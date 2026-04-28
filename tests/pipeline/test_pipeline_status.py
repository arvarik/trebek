"""
Tests for PipelineStatus enum — StrEnum behavior, SQLite compatibility,
and exhaustive status coverage.
"""

import pytest

from trebek.status import PipelineStatus


class TestPipelineStatusEnum:
    """PipelineStatus StrEnum tests."""

    def test_all_statuses_defined(self) -> None:
        expected = {
            "PENDING",
            "TRANSCRIBING",
            "TRANSCRIPT_READY",
            "CLEANED",
            "SAVING",
            "MULTIMODAL_PROCESSING",
            "MULTIMODAL_DONE",
            "VECTORIZING",
            "COMPLETED",
            "FAILED",
        }
        actual = {s.value for s in PipelineStatus}
        assert actual == expected

    def test_strenum_string_equality(self) -> None:
        """StrEnum values should be directly comparable to strings (SQLite compatibility)."""
        assert PipelineStatus.PENDING == "PENDING"
        assert PipelineStatus.COMPLETED == "COMPLETED"
        assert PipelineStatus.FAILED == "FAILED"

    def test_strenum_in_set_membership(self) -> None:
        """StrEnum values should work in string set membership checks."""
        statuses = {"PENDING", "COMPLETED"}
        assert PipelineStatus.PENDING in statuses
        assert PipelineStatus.FAILED not in statuses

    def test_strenum_string_formatting(self) -> None:
        """StrEnum should serialize cleanly in f-strings and SQL."""
        assert f"status = '{PipelineStatus.PENDING}'" == "status = 'PENDING'"

    def test_lookup_by_value(self) -> None:
        assert PipelineStatus("PENDING") is PipelineStatus.PENDING
        assert PipelineStatus("COMPLETED") is PipelineStatus.COMPLETED

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValueError):
            PipelineStatus("INVALID_STATUS")
