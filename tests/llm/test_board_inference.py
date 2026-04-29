"""
Tests for board.py dollar-value parsing and board row inference.

Validates that the hardened _parse_dollar_value function rejects false
positives (contestant scores, wagers, years) while accepting legitimate
clue selection values.
"""

from trebek.llm.board import (
    _parse_dollar_value,
    infer_board_row,
    STANDARD_BOARD,
    _ALL_VALID_CLUE_VALUES,
)


class TestParseValidSelectionValues:
    """Verify that standard clue selection patterns are correctly parsed."""

    def test_standard_for_dollar_pattern(self):
        assert _parse_dollar_value("Let's try Songwriters for $800") == 800

    def test_for_dollar_with_comma(self):
        assert _parse_dollar_value("Geography for $1,200") == 1200

    def test_for_dollar_no_comma(self):
        assert _parse_dollar_value("Science for $1600") == 1600

    def test_spoken_for_pattern(self):
        assert _parse_dollar_value("Songwriters for 800") == 800

    def test_standalone_dollar_value(self):
        assert _parse_dollar_value("$600 clue in Poetry") == 600

    def test_j_round_200(self):
        assert _parse_dollar_value("Trees for $200") == 200

    def test_dj_round_2000(self):
        assert _parse_dollar_value("History for $2,000") == 2000


class TestRejectFalsePositives:
    """Verify that non-clue dollar amounts are rejected."""

    def test_contestant_score_4000(self):
        """$4000 is not a valid clue value on any board format."""
        assert _parse_dollar_value("Stephen now has $4,000") is None

    def test_large_wager_13201(self):
        """Wager amounts like $13,201 should not be parsed as clue values."""
        assert _parse_dollar_value("wagered $13,201") is None

    def test_wager_context_keyword(self):
        """Dollar values preceded by 'wager' context should be rejected."""
        assert _parse_dollar_value("wager of $1,000") is None

    def test_score_context_keyword(self):
        """Dollar values with 'score' context should be rejected."""
        assert _parse_dollar_value("score is $800") is None

    def test_trailing_keyword_with_valid_value(self):
        """'has $600' should be rejected (score context)."""
        assert _parse_dollar_value("She now has $600") is None

    def test_if_youre_right_context(self):
        """DD announcements like 'if you're right' are wager context, not selection."""
        assert _parse_dollar_value("for $3,400 if you're right") is None

    def test_non_board_value_3000(self):
        """$3,000 is not a standard board value."""
        assert _parse_dollar_value("for $3,000") is None

    def test_non_board_value_5000(self):
        """$5,000 is not a standard board value."""
        assert _parse_dollar_value("for $5,000") is None

    def test_year_not_matched(self):
        """A bare year should not be parsed (no $ sign)."""
        assert _parse_dollar_value("in 1973 they invented") is None


class TestAllValidClueValues:
    """Verify the comprehensive set of valid clue values."""

    def test_standard_j_values(self):
        for v in (200, 400, 600, 800, 1000):
            assert v in _ALL_VALID_CLUE_VALUES

    def test_standard_dj_values(self):
        for v in (400, 800, 1200, 1600, 2000):
            assert v in _ALL_VALID_CLUE_VALUES

    def test_invalid_values_not_present(self):
        for v in (4000, 3000, 5000, 7000, 13201):
            assert v not in _ALL_VALID_CLUE_VALUES


class TestInferBoardRow:
    """Test board row inference with LLM fallback."""

    def test_exact_j_value(self):
        assert infer_board_row(200, "J!", STANDARD_BOARD) == 1
        assert infer_board_row(1000, "J!", STANDARD_BOARD) == 5

    def test_exact_dj_value(self):
        assert infer_board_row(400, "Double J!", STANDARD_BOARD) == 1
        assert infer_board_row(2000, "Double J!", STANDARD_BOARD) == 5

    def test_none_uses_llm_fallback(self):
        """When no value is parsed, use the LLM's guessed row."""
        assert infer_board_row(None, "J!", STANDARD_BOARD, llm_fallback_row=4) == 4
        assert infer_board_row(None, "J!", STANDARD_BOARD, llm_fallback_row=2) == 2

    def test_none_clamps_fallback(self):
        """LLM fallback should be clamped to 1-5."""
        assert infer_board_row(None, "J!", STANDARD_BOARD, llm_fallback_row=0) == 1
        assert infer_board_row(None, "J!", STANDARD_BOARD, llm_fallback_row=7) == 5

    def test_unmatched_value_uses_llm_fallback(self):
        """Values outside the grid should fall back to LLM guess, not row 3."""
        result = infer_board_row(4000, "Double J!", STANDARD_BOARD, llm_fallback_row=2)
        assert result == 2  # Should use LLM guess, NOT default to 3

    def test_final_jeopardy_always_row_1(self):
        assert infer_board_row(None, "Final J!", STANDARD_BOARD) == 1
