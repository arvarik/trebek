"""Tests for response format normalization.

Validates that the normalize_response_format function correctly
converts bare answers to J! question format.
"""

from trebek.llm.validation import normalize_response_format, _is_likely_person
from trebek.schemas import Clue


def _make_clue(correct_response: str, category: str = "Test") -> Clue:
    """Create a minimal Clue for testing."""
    return Clue(
        round="J!",
        category=category,
        board_row=1,
        board_col=1,
        selection_order=1,
        is_daily_double=False,
        requires_visual_context=False,
        host_start_timestamp_ms=0,
        host_finish_timestamp_ms=1000,
        clue_syllable_count=10,
        daily_double_wager=None,
        wagerer_name=None,
        clue_text="Some clue text",
        correct_response=correct_response,
        attempts=[],
    )


class TestIsLikelyPerson:
    """Test person detection heuristic."""

    def test_full_name(self) -> None:
        assert _is_likely_person("Charles Dickens") is True

    def test_single_word_not_person(self) -> None:
        assert _is_likely_person("Paris") is False

    def test_person_indicator(self) -> None:
        assert _is_likely_person("President Lincoln") is True

    def test_actor_indicator(self) -> None:
        assert _is_likely_person("the actor Tom Hanks") is True

    def test_title(self) -> None:
        assert _is_likely_person("Sir Isaac Newton") is True

    def test_lowercase_not_name(self) -> None:
        assert _is_likely_person("the color blue") is False


class TestNormalizeResponseFormat:
    """Test response format normalization."""

    def test_bare_word_gets_what_is(self) -> None:
        clues = [_make_clue("Paris")]
        fixed = normalize_response_format(clues)
        assert fixed == 1
        assert clues[0].correct_response == "What is Paris?"

    def test_already_correct_not_touched(self) -> None:
        clues = [_make_clue("What is Paris?")]
        fixed = normalize_response_format(clues)
        assert fixed == 0
        assert clues[0].correct_response == "What is Paris?"

    def test_who_is_for_person(self) -> None:
        clues = [_make_clue("Charles Dickens")]
        fixed = normalize_response_format(clues)
        assert fixed == 1
        assert clues[0].correct_response == "Who is Charles Dickens?"

    def test_who_is_for_single_name_with_indicator(self) -> None:
        clues = [_make_clue("President Lincoln")]
        fixed = normalize_response_format(clues)
        assert fixed == 1
        assert "Who" in clues[0].correct_response

    def test_plural_not_guessed(self) -> None:
        """We don't guess plural — too many false positives (Paris, Jaws, Mars)."""
        clues = [_make_clue("electrons")]
        fixed = normalize_response_format(clues)
        assert fixed == 1
        assert clues[0].correct_response == "What is electrons?"

    def test_who_are_prefix_preserved(self) -> None:
        clues = [_make_clue("Who are the Beatles?")]
        fixed = normalize_response_format(clues)
        assert fixed == 0

    def test_where_is_prefix_preserved(self) -> None:
        clues = [_make_clue("Where is Tokyo?")]
        fixed = normalize_response_format(clues)
        assert fixed == 0

    def test_empty_response_skipped(self) -> None:
        clues = [_make_clue("")]
        fixed = normalize_response_format(clues)
        assert fixed == 0

    def test_multiple_clues_mixed(self) -> None:
        clues = [
            _make_clue("What is gravity?"),  # Already correct
            _make_clue("Paris"),  # Needs fixing
            _make_clue("Who is Newton?"),  # Already correct
            _make_clue("the Nile"),  # Needs fixing
        ]
        fixed = normalize_response_format(clues)
        assert fixed == 2
        assert clues[0].correct_response == "What is gravity?"
        assert clues[1].correct_response == "What is Paris?"
        assert clues[2].correct_response == "Who is Newton?"
        assert clues[3].correct_response == "What is the Nile?"

    def test_name_prefix_preserved(self) -> None:
        clues = [_make_clue("Name two countries...")]
        fixed = normalize_response_format(clues)
        assert fixed == 0

    def test_double_s_ending_not_plural(self) -> None:
        """Words ending in 'ss' like 'grass' should use 'is' not 'are'."""
        clues = [_make_clue("grass")]
        normalize_response_format(clues)
        assert clues[0].correct_response == "What is grass?"

    def test_contraction_whats_preserved(self) -> None:
        """'What's a sink?' is already a valid question form — don't double-prefix."""
        clues = [_make_clue("What's a sink?")]
        fixed = normalize_response_format(clues)
        assert fixed == 0
        assert clues[0].correct_response == "What's a sink?"

    def test_contraction_whos_preserved(self) -> None:
        """'Who's that?' should not be modified."""
        clues = [_make_clue("Who's the little drummer boy?")]
        fixed = normalize_response_format(clues)
        assert fixed == 0
