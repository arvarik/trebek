"""
Tests for Stage 3.5 verification module.

Validates transcript context building, batch construction, FJ context,
and the verification schema definitions.
"""

import pytest
from pydantic import ValidationError

from trebek.llm.verify import (
    _build_clue_context,
    _build_fj_context,
    SingleClueVerification,
    BatchVerificationResult,
    FJVerification,
    VERIFY_BATCH_SIZE,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _make_segments(n: int = 50) -> list[dict]:
    """Create synthetic transcript segments for testing."""
    segments = []
    for i in range(n):
        segments.append(
            {
                "speaker": f"SPEAKER_{i % 3:02d}",
                "text": f"Segment {i} text content here.",
                "start": float(i) * 2.0,
                "end": float(i) * 2.0 + 1.8,
            }
        )
    return segments


class _MockClue:
    """Minimal clue-like object for testing context building."""

    def __init__(
        self,
        start_line: str = "L10",
        end_line: str = "L12",
        category: str = "Test Category",
        clue_text: str = "This is a test clue.",
        correct_response: str = "What is a test?",
        attempts: list | None = None,
    ):
        self.host_read_start_line_id = start_line
        self.host_read_end_line_id = end_line
        self.category = category
        self.clue_text = clue_text
        self.correct_response = correct_response
        self.attempts = attempts or []


# ── Transcript Context Tests ────────────────────────────────────────


class TestBuildClueContext:
    """Test transcript context window construction."""

    def test_basic_context_window(self):
        segments = _make_segments(30)
        clue = _MockClue(start_line="L10", end_line="L12")
        context = _build_clue_context(clue, segments)

        assert "CLUE START" in context
        assert "CLUE END" in context
        # Should contain lines before and after
        assert "L7" in context or "L8" in context
        assert "L15" in context or "L18" in context

    def test_clue_at_start_of_transcript(self):
        segments = _make_segments(30)
        clue = _MockClue(start_line="L0", end_line="L2")
        context = _build_clue_context(clue, segments)

        assert "CLUE START" in context
        assert "L0" in context

    def test_clue_at_end_of_transcript(self):
        segments = _make_segments(30)
        clue = _MockClue(start_line="L27", end_line="L29")
        context = _build_clue_context(clue, segments)

        assert "CLUE END" in context
        assert "L29" in context

    def test_invalid_line_id(self):
        segments = _make_segments(10)
        clue = _MockClue(start_line="INVALID", end_line="L5")
        context = _build_clue_context(clue, segments)

        assert "Unable to resolve" in context

    def test_out_of_bounds_clamped(self):
        segments = _make_segments(10)
        clue = _MockClue(start_line="L100", end_line="L200")
        context = _build_clue_context(clue, segments)

        # Should clamp to last segment
        assert "L9" in context


class TestBuildFJContext:
    """Test Final Jeopardy context construction."""

    def test_uses_last_40_segments(self):
        segments = _make_segments(100)
        context = _build_fj_context(segments)

        # Should start from segment 60 (100 - 40)
        assert "L60" in context
        assert "L99" in context
        # Should NOT contain early segments
        assert "L0 [" not in context

    def test_short_transcript(self):
        segments = _make_segments(10)
        context = _build_fj_context(segments)

        # Should contain all segments
        assert "L0" in context
        assert "L9" in context


# ── Schema Validation Tests ─────────────────────────────────────────


class TestSingleClueVerification:
    """Test the verification result schema."""

    def test_verified_confidence(self):
        v = SingleClueVerification(
            clue_index=0,
            verified_clue_text="This is the actual clue.",
            verified_correct_response="What is Paris?",
            confidence="verified",
        )
        assert v.confidence == "verified"
        assert v.correction_type == ""

    def test_corrected_with_type(self):
        v = SingleClueVerification(
            clue_index=5,
            verified_clue_text="Corrected text.",
            verified_correct_response="Who is Camus?",
            confidence="corrected",
            correction_type="typo",
            correction_detail="ASR heard 'Kemu' but correct spelling is 'Camus'",
        )
        assert v.confidence == "corrected"
        assert v.correction_type == "typo"
        assert "Camus" in v.correction_detail

    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            SingleClueVerification(
                clue_index=0,
                verified_clue_text="text",
                verified_correct_response="What is test?",
                confidence="invalid_value",
            )


class TestBatchVerificationResult:
    """Test batch verification result schema."""

    def test_empty_batch(self):
        result = BatchVerificationResult(verifications=[])
        assert len(result.verifications) == 0

    def test_full_batch(self):
        verifications = [
            SingleClueVerification(
                clue_index=i,
                verified_clue_text=f"Clue {i}",
                verified_correct_response=f"What is {i}?",
                confidence="verified",
            )
            for i in range(VERIFY_BATCH_SIZE)
        ]
        result = BatchVerificationResult(verifications=verifications)
        assert len(result.verifications) == VERIFY_BATCH_SIZE


class TestFJVerification:
    """Test Final Jeopardy verification schema."""

    def test_fj_verified(self):
        v = FJVerification(
            verified_correct_response="What is Ulysses?",
            confidence="verified",
        )
        assert v.verified_correct_response == "What is Ulysses?"

    def test_fj_corrected(self):
        v = FJVerification(
            verified_correct_response="What is the Treaty of Versailles?",
            confidence="corrected",
            correction_type="response_fixed",
            correction_detail="Original said 'Treaty of Versay' — host revealed 'Treaty of Versailles'",
        )
        assert v.correction_type == "response_fixed"


# ── Batch Construction Tests ────────────────────────────────────────


class TestBatchConstruction:
    """Test that batch sizing works correctly for various episode sizes."""

    def test_standard_60_clues(self):
        """60 clues should produce 5 batches of 12."""
        clues = [_MockClue() for _ in range(60)]
        batches = []
        current = []
        for i, c in enumerate(clues):
            current.append((i, c))
            if len(current) >= VERIFY_BATCH_SIZE:
                batches.append(current)
                current = []
        if current:
            batches.append(current)

        assert len(batches) == 5
        assert all(len(b) == 12 for b in batches)

    def test_55_clues_uneven(self):
        """55 clues should produce 4 batches of 12 + 1 batch of 7."""
        clues = [_MockClue() for _ in range(55)]
        batches = []
        current = []
        for i, c in enumerate(clues):
            current.append((i, c))
            if len(current) >= VERIFY_BATCH_SIZE:
                batches.append(current)
                current = []
        if current:
            batches.append(current)

        assert len(batches) == 5
        assert len(batches[-1]) == 7

    def test_empty_clues(self):
        """No clues should produce no batches."""
        batches = []
        current = []
        if current:
            batches.append(current)
        assert len(batches) == 0
