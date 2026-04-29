"""
Tests for selection text stripping from assembled clue text.

Validates that _strip_selection_preamble correctly removes category+value
selection announcements from the beginning of clue text while preserving
the actual clue content.
"""

from trebek.llm.pass2_extraction import _strip_selection_preamble


class TestStripSelectionPreamble:
    """Test preamble stripping from clue text."""

    def test_category_for_value_dot(self):
        """Standard pattern: 'Category for 600. Actual clue text...'"""
        result = _strip_selection_preamble(
            "Numeric words and phrases for 600. On CB radio, if you want to know someone's location, you ask what's your number.",
            "Numeric words and phrases",
        )
        assert result.startswith("On CB radio")

    def test_category_for_dollar_value(self):
        """Dollar sign pattern: 'Category for $200. Actual clue text...'"""
        result = _strip_selection_preamble(
            "I can adapt for $200. This semi-aquatic Aussie creature has fur keeping it dry.",
            "I can adapt",
        )
        assert result.startswith("This semi-aquatic")

    def test_lets_stick_with_pattern(self):
        """'Let's stick with Category for value.' pattern."""
        result = _strip_selection_preamble(
            "Let's stick with trees for 600. Andrew Jackson's troops nicknamed him Old This.",
            "a little tree-pourri",
        )
        assert result.startswith("Andrew Jackson")

    def test_one_more_time_pattern(self):
        """'One more time, we have Category.' pattern."""
        result = _strip_selection_preamble(
            "One more time, we have 21st century game shows. The New York Times described this show.",
            "21st century game shows",
        )
        assert result.startswith("The New York Times")

    def test_no_preamble_no_change(self):
        """Clue text without preamble should be returned unchanged."""
        original = "This Bengali poet did double duty, composing the anthems of both India and Bangladesh."
        result = _strip_selection_preamble(original, "national anthem lore")
        assert result == original

    def test_short_result_not_stripped(self):
        """If stripping would leave < 10 chars, keep the original."""
        original = "History for 600. Short."
        result = _strip_selection_preamble(original, "History")
        assert result == original

    def test_empty_text(self):
        result = _strip_selection_preamble("", "Some Category")
        assert result == ""

    def test_category_prefix_only(self):
        """When text starts with category name but has no value pattern."""
        original = "The 1980s scandal known by this hyphenated name saw money from illegal Mideast arms sales."
        result = _strip_selection_preamble(original, "The 1980s")
        assert result == original

    def test_im_thinking_pattern(self):
        """'I'm thinking of a James for 2000.' pattern."""
        result = _strip_selection_preamble(
            "I'm thinking of a James for 2000. Who drew hundreds of cartoons and six covers for The New Yorker?",
            "I'm thinking of a James",
        )
        assert result.startswith("Who drew hundreds")
