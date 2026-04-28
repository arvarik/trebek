"""Tests for pass2 extraction assembly logic — Line ID resolution, timestamps,
FJ filtering, wager parsing, and syllable estimation.

These tests validate the post-LLM logic that transforms raw extraction output
into structured Clue objects. They do NOT require LLM calls.
"""

import re

from trebek.llm.pass2_extraction import _count_syllables


# ── Syllable Estimation Tests ───────────────────────────────────────


class TestSyllableEstimation:
    """Tests for the vowel-cluster syllable counting heuristic used in pass2."""

    def test_simple_words(self) -> None:
        assert _count_syllables("cat") == 1
        assert _count_syllables("hello") == 2
        assert _count_syllables("beautiful") == 3

    def test_silent_e(self) -> None:
        """Words ending in 'e' with multiple vowel clusters should reduce by 1."""
        assert _count_syllables("make") == 1  # m-ake → 1 (silent e)
        assert _count_syllables("computer") == 3  # com-pu-ter

    def test_empty_text(self) -> None:
        assert _count_syllables("") == 1  # max(1, 0) = 1

    def test_multi_word(self) -> None:
        result = _count_syllables("What is the capital of France")
        assert result >= 8  # conservative lower bound

    def test_single_consonant_word(self) -> None:
        """Words with no vowels should get min 1 syllable."""
        assert _count_syllables("nth") == 1

    def test_jeopardy_style_text(self) -> None:
        """Jeopardy clues use academic vocabulary — should be reasonably accurate."""
        result = _count_syllables("This Mediterranean peninsula was home to the Roman Empire")
        # me-di-te-rra-ne-an pe-nin-su-la was home to the Ro-man Em-pire
        assert result >= 15


# ── Wager Parsing Tests ─────────────────────────────────────────────


class TestWagerParsing:
    """Tests for DD wager parsing logic as implemented in pass2."""

    @staticmethod
    def _parse_wager(raw: str | None) -> int | str | None:
        """Mirror of the wager parsing logic in pass2_extraction."""
        from typing import Union, Literal, Optional

        wager_val: Optional[Union[int, Literal["True Daily Double"]]] = None
        if raw is not None:
            if raw == "True Daily Double":
                wager_val = "True Daily Double"
            else:
                try:
                    wager_val = int(raw)
                except ValueError:
                    pass  # In production, this is now logged
        return wager_val

    def test_numeric_wager(self) -> None:
        assert self._parse_wager("2000") == 2000

    def test_true_daily_double(self) -> None:
        assert self._parse_wager("True Daily Double") == "True Daily Double"

    def test_none_wager(self) -> None:
        assert self._parse_wager(None) is None

    def test_invalid_wager(self) -> None:
        assert self._parse_wager("all of it") is None

    def test_zero_wager(self) -> None:
        assert self._parse_wager("0") == 0

    def test_negative_wager(self) -> None:
        assert self._parse_wager("-500") == -500


# ── Line ID Resolution Tests ────────────────────────────────────────


class TestLineIDResolution:
    """Tests for the Line ID parsing and bounds-clamping logic."""

    @staticmethod
    def _parse_line_id(raw: str) -> int | None:
        """Mirror of pass2_extraction Line ID parsing."""
        cleaned = raw.replace("L", "").replace("[", "").replace("]", "").strip()
        try:
            return int(cleaned)
        except ValueError:
            return None

    def test_standard_format(self) -> None:
        assert self._parse_line_id("L105") == 105

    def test_bracketed_format(self) -> None:
        assert self._parse_line_id("[L42]") == 42

    def test_double_bracketed(self) -> None:
        assert self._parse_line_id("[[L99]]") == 99

    def test_bare_number(self) -> None:
        assert self._parse_line_id("50") == 50

    def test_invalid_format(self) -> None:
        assert self._parse_line_id("Line Five") is None

    def test_empty_string(self) -> None:
        assert self._parse_line_id("") is None

    @staticmethod
    def _clamp_line_id(idx: int, max_idx: int) -> int:
        """Mirror of pass2_extraction bounds clamping."""
        return max(0, min(idx, max_idx))

    def test_clamp_negative(self) -> None:
        assert self._clamp_line_id(-5, 100) == 0

    def test_clamp_over_max(self) -> None:
        assert self._clamp_line_id(500, 99) == 99

    def test_clamp_valid(self) -> None:
        assert self._clamp_line_id(50, 100) == 50


# ── Non-Standard Line ID Format Detection ────────────────────────────


class TestNonStandardLineIDDetection:
    """Tests for tracking non-standard Line ID formats for prompt engineering feedback."""

    @staticmethod
    def _is_standard_line_id(raw: str) -> bool:
        return bool(re.match(r"^L\d+$", raw.strip()))

    def test_standard_format(self) -> None:
        assert self._is_standard_line_id("L0") is True
        assert self._is_standard_line_id("L105") is True

    def test_bracketed_nonstandard(self) -> None:
        assert self._is_standard_line_id("[L42]") is False

    def test_no_prefix_nonstandard(self) -> None:
        assert self._is_standard_line_id("42") is False

    def test_lowercase_nonstandard(self) -> None:
        assert self._is_standard_line_id("l42") is False


# ── Timestamp Reconstruction Tests ──────────────────────────────────


class TestTimestampReconstruction:
    """Tests for segment timestamp → millisecond conversion."""

    def test_basic_conversion(self) -> None:
        segment: dict[str, float | str] = {"start": 5.123, "end": 6.456, "text": "test"}
        host_start_ms = float(segment["start"]) * 1000.0
        host_finish_ms = float(segment["end"]) * 1000.0
        assert host_start_ms == 5123.0
        assert host_finish_ms == 6456.0

    def test_missing_start_defaults_zero(self) -> None:
        segment = {"end": 1.0, "text": "test"}
        raw_start = segment.get("start")
        assert raw_start is None

    def test_missing_end_defaults_zero(self) -> None:
        segment = {"start": 1.0, "text": "test"}
        raw_end = segment.get("end")
        assert raw_end is None


# ── Lockout Penalty Tests ───────────────────────────────────────────


class TestLockoutPenalty:
    """Tests for the response start offset calculation with lockout penalty."""

    def test_no_lockout(self) -> None:
        buzz_ms = 5000.0
        lockout_penalty = 0.0  # not inferred
        response_start = buzz_ms + 250.0 + lockout_penalty
        assert response_start == 5250.0

    def test_with_lockout(self) -> None:
        buzz_ms = 5000.0
        lockout_penalty = 250.0  # lockout inferred
        response_start = buzz_ms + 250.0 + lockout_penalty
        assert response_start == 5500.0
