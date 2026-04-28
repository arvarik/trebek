"""
Tests for buzzer physics and acoustic confidence metrics —
pure math functions, zero external dependencies.
"""

from trebek.analysis.buzzer import (
    WhisperXWordSegment,
    calculate_true_buzzer_latency,
    calculate_true_acoustic_metrics,
)


class TestCalculateTrueBuzzerLatency:
    """Reaction time calculation tests."""

    def test_positive_latency(self) -> None:
        assert calculate_true_buzzer_latency(1.500, 1.200) == 0.300

    def test_zero_latency(self) -> None:
        assert calculate_true_buzzer_latency(1.0, 1.0) == 0.0

    def test_negative_latency_pre_buzz(self) -> None:
        """Negative latency = contestant buzzed before the light (lockout risk)."""
        assert calculate_true_buzzer_latency(1.0, 1.2) == -0.200

    def test_precision_three_decimals(self) -> None:
        result = calculate_true_buzzer_latency(1.5556, 1.2223)
        assert result == 0.333

    def test_large_values(self) -> None:
        result = calculate_true_buzzer_latency(3600.500, 3600.200)
        assert result == 0.300


class TestCalculateTrueAcousticMetrics:
    """Acoustic confidence and disfluency detection tests."""

    def _make_segments(self, words: list[tuple[str, float, float, float]]) -> list[WhisperXWordSegment]:
        return [WhisperXWordSegment(word=w, start=s, end=e, prob=p) for w, s, e, p in words]

    def test_single_word_confidence(self) -> None:
        segments = self._make_segments([("Rome", 1.5, 1.8, 0.95)])
        result = calculate_true_acoustic_metrics(1.0, 2.0, segments)
        assert result["true_acoustic_confidence_score"] == 0.95
        assert result["disfluency_count"] == 0

    def test_multiple_words_average_confidence(self) -> None:
        segments = self._make_segments(
            [
                ("What", 1.5, 1.6, 0.90),
                ("is", 1.6, 1.7, 0.80),
                ("Rome", 1.7, 1.9, 0.70),
            ]
        )
        result = calculate_true_acoustic_metrics(1.0, 2.0, segments)
        expected = round((0.90 + 0.80 + 0.70) / 3, 4)
        assert result["true_acoustic_confidence_score"] == expected

    def test_no_segments_in_window(self) -> None:
        """If no words fall within the buzz window, return zero confidence."""
        segments = self._make_segments([("Rome", 5.0, 5.5, 0.95)])
        result = calculate_true_acoustic_metrics(1.0, 2.0, segments)
        assert result["true_acoustic_confidence_score"] == 0.0
        assert result["disfluency_count"] == 0

    def test_empty_segments_list(self) -> None:
        result = calculate_true_acoustic_metrics(1.0, 2.0, [])
        assert result["true_acoustic_confidence_score"] == 0.0
        assert result["disfluency_count"] == 0

    def test_disfluency_detection_um(self) -> None:
        segments = self._make_segments(
            [
                ("um", 1.5, 1.6, 0.30),
                ("Rome", 1.6, 1.9, 0.85),
            ]
        )
        result = calculate_true_acoustic_metrics(1.0, 2.0, segments)
        assert result["disfluency_count"] == 1

    def test_disfluency_detection_multiple(self) -> None:
        segments = self._make_segments(
            [
                ("uh", 1.5, 1.55, 0.20),
                ("um", 1.55, 1.6, 0.25),
                ("Rome?", 1.6, 1.9, 0.85),  # punctuation stripped before matching
            ]
        )
        result = calculate_true_acoustic_metrics(1.0, 2.0, segments)
        assert result["disfluency_count"] == 2

    def test_disfluency_markers_case_and_punctuation(self) -> None:
        """Markers should be detected after lowering and stripping punctuation."""
        segments = self._make_segments(
            [
                ("Uh,", 1.5, 1.6, 0.30),
                ("er.", 1.6, 1.65, 0.25),
                ("hmm!", 1.65, 1.7, 0.20),
                ("ah?", 1.7, 1.75, 0.22),
                ("Rome", 1.75, 1.9, 0.90),
            ]
        )
        result = calculate_true_acoustic_metrics(1.0, 2.0, segments)
        assert result["disfluency_count"] == 4

    def test_segments_outside_window_excluded(self) -> None:
        """Only segments entirely within the window should be counted."""
        segments = self._make_segments(
            [
                ("Before", 0.5, 0.9, 0.50),  # Entirely before window
                ("Inside", 1.5, 1.8, 0.95),  # Inside window
                ("Partial", 1.9, 2.5, 0.70),  # Starts inside, ends outside — excluded
                ("After", 3.0, 3.5, 0.60),  # Entirely after window
            ]
        )
        result = calculate_true_acoustic_metrics(1.0, 2.0, segments)
        assert result["true_acoustic_confidence_score"] == 0.95
        assert result["disfluency_count"] == 0


class TestWhisperXWordSegment:
    """Pydantic model validation."""

    def test_valid_segment(self) -> None:
        seg = WhisperXWordSegment(word="hello", start=1.0, end=1.5, prob=0.95)
        assert seg.word == "hello"

    def test_zero_prob(self) -> None:
        seg = WhisperXWordSegment(word="uh", start=0.0, end=0.1, prob=0.0)
        assert seg.prob == 0.0
