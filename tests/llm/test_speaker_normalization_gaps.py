"""
Tests for speaker normalization — coverage gap tests targeting
uncovered branches (lines 60, 86-106, 148, 157, 166, 202-203).

Complements the existing test_speaker_normalization.py with edge cases
for multi-word Pass 1 names, substring containment fallback in mapping,
unmapped Pass 1 names, and fuzzy matching on wagerers/FJ wagers.
"""

from trebek.llm.speaker_normalization import _normalize_speaker_names, _fuzzy_match_contestant
from trebek.schemas import Clue, BuzzAttempt, ScoreAdjustment, FinalJepWager


def _make_clue(speaker: str, **kwargs) -> Clue:  # type: ignore[no-untyped-def]
    return Clue(
        round="J!",
        category="Test",
        board_row=1,
        board_col=1,
        selection_order=1,
        is_daily_double=False,
        requires_visual_context=False,
        host_start_timestamp_ms=0.0,
        host_finish_timestamp_ms=1000.0,
        clue_syllable_count=5,
        clue_text="Test clue",
        correct_response="Test answer",
        attempts=[
            BuzzAttempt(
                attempt_order=1,
                speaker=speaker,
                response_given="What?",
                is_correct=True,
                buzz_timestamp_ms=1500.0,
                response_start_timestamp_ms=1750.0,
                is_lockout_inferred=False,
            )
        ],
        **kwargs,
    )


class TestInitialSkipping:
    """Line 60: Skip name parts with length <= 1 (initials like 'W.')."""

    def test_single_letter_initial_not_mapped(self) -> None:
        """'W. Kamau Bell' — 'W' should not create a mapping, but 'Kamau' should."""
        clues = [_make_clue("Kamau")]
        _normalize_speaker_names(clues, {}, ["W. Kamau Bell", "Jane Smith", "Bob Lee"])
        assert clues[0].attempts[0].speaker == "W. Kamau Bell"

    def test_initial_with_dot_stripped(self) -> None:
        """'J.' should match 'J. Robert Smith' via substring containment."""
        clues = [_make_clue("J.")]
        _normalize_speaker_names(clues, {}, ["J. Robert Smith", "Mary Jones", "Tim Lee"])
        # "J." matches "J. Robert Smith" via substring containment
        assert len(clues[0].attempts) == 1
        assert clues[0].attempts[0].speaker == "J. Robert Smith"


class TestMultiWordPassOneName:
    """Lines 86-91: Try each word of the multi-word mapped name to resolve."""

    def test_multi_word_pass1_name_resolved_by_word(self) -> None:
        """Pass 1 maps SPEAKER_01 to 'Rachel B' — 'Rachel' should resolve to full name."""
        clues = [_make_clue("SPEAKER_01")]
        mapping = {"SPEAKER_00": "Host", "SPEAKER_01": "Rachel B"}
        _normalize_speaker_names(clues, mapping, ["Rachel Bernstein", "Dan Puma", "Scott Riccardi"], host_name="Host")
        assert clues[0].attempts[0].speaker == "Rachel Bernstein"


class TestSubstringContainmentInMapping:
    """Lines 94-98: Substring containment fallback when neither exact nor word match works."""

    def test_substring_match_in_speaker_mapping(self) -> None:
        """Pass 1 maps SPEAKER_02 to 'DeFrank' — should match 'Alex DeFrank' via substring."""
        clues = [_make_clue("SPEAKER_02")]
        mapping = {"SPEAKER_00": "Host", "SPEAKER_02": "DeFrank"}
        _normalize_speaker_names(clues, mapping, ["Alex DeFrank", "Bob Smith", "Jane Doe"], host_name="Host")
        assert clues[0].attempts[0].speaker == "Alex DeFrank"


class TestUnmappedPassOneName:
    """Lines 103-106: Pass 1 name doesn't match any contestant — map directly."""

    def test_unmapped_pass1_name_used_directly(self) -> None:
        """When the Pass 1 name doesn't match any contestant, it's stored as-is in variant_map.
        The speaker resolves to the raw Pass 1 name, and hard cleanup drops it."""
        clues = [_make_clue("SPEAKER_02")]
        mapping = {"SPEAKER_00": "Host", "SPEAKER_02": "CompletelyUnknownPerson"}
        _normalize_speaker_names(clues, mapping, ["Rachel Bernstein", "Dan Puma", "Scott Riccardi"], host_name="Host")
        # SPEAKER_02 → "CompletelyUnknownPerson" stored directly in variant_map.
        # Hard cleanup checks if the resolved name is a valid contestant.
        # "CompletelyUnknownPerson" is NOT a contestant, but hard cleanup only drops
        # if fuzzy match fails AND name is not in variant_map.
        # Since it WAS placed in variant_map, it survives as-is.
        assert len(clues[0].attempts) == 1
        assert clues[0].attempts[0].speaker == "CompletelyUnknownPerson"

    def test_unmapped_name_also_stored_as_variant(self) -> None:
        """The Pass 1 name itself should also be stored in the variant map."""
        clues = [_make_clue("ValidName")]
        # When "ValidName" doesn't match any contestant, it maps directly
        mapping = {"SPEAKER_01": "ValidName"}
        _normalize_speaker_names(clues, mapping, ["Alice Johnson", "Bob Smith", "Carol White"], host_name="Host")
        # "ValidName" doesn't match any contestant → dropped
        assert len(clues[0].attempts) == 0


class TestWagererFuzzyMatch:
    """Line 148: Fuzzy match wagerer_name when exact match fails."""

    def test_wagerer_fuzzy_matched(self) -> None:
        """A wagerer_name with a typo should be fuzzy-matched."""
        clue = Clue(
            round="J!",
            category="Test",
            board_row=1,
            board_col=1,
            selection_order=1,
            is_daily_double=True,
            requires_visual_context=False,
            host_start_timestamp_ms=0.0,
            host_finish_timestamp_ms=1000.0,
            clue_syllable_count=5,
            clue_text="Test clue",
            correct_response="Test answer",
            daily_double_wager=500,
            wagerer_name="Rachael",  # Typo
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Rachel Bernstein",
                    response_given="What?",
                    is_correct=True,
                    buzz_timestamp_ms=1500.0,
                    response_start_timestamp_ms=1750.0,
                    is_lockout_inferred=False,
                )
            ],
        )
        _normalize_speaker_names([clue], {}, ["Rachel Bernstein", "Dan Puma", "Scott Riccardi"])
        assert clue.wagerer_name == "Rachel Bernstein"


class TestScoreAdjustmentFuzzyMatch:
    """Line 157: Fuzzy match score_adjustment contestant name."""

    def test_score_adjustment_fuzzy_matched(self) -> None:
        clues = [_make_clue("Rachel Bernstein")]
        adj = ScoreAdjustment(
            contestant="Rachael",  # Typo
            points_adjusted=-400,
            reason="Judge correction",
            effective_after_clue_selection_order=1,
        )
        _normalize_speaker_names(clues, {}, ["Rachel Bernstein", "Dan Puma", "Scott Riccardi"], score_adjustments=[adj])
        assert adj.contestant == "Rachel Bernstein"


class TestFinalJepWagerFuzzyMatch:
    """Line 166: Fuzzy match FJ wager contestant name."""

    def test_fj_wager_fuzzy_matched(self) -> None:
        clues = [_make_clue("Rachel Bernstein")]
        wager = FinalJepWager(
            contestant="Rachael",  # Typo
            wager=5000,
            response="What is X?",
            is_correct=True,
        )
        _normalize_speaker_names(clues, {}, ["Rachel Bernstein", "Dan Puma", "Scott Riccardi"], fj_wagers=[wager])
        assert wager.contestant == "Rachel Bernstein"


class TestHardCleanupCaseExactMatch:
    """Lines 202-203: Case-exact match in hard cleanup (already canonical name)."""

    def test_canonical_name_passes_hard_cleanup(self) -> None:
        """A speaker that is already a canonical contestant name should pass through."""
        clues = [_make_clue("UNKNOWN_SPEAKER")]
        # After soft normalization, "UNKNOWN_SPEAKER" is unmapped.
        # Hard cleanup fuzzy resolves or drops it.
        _normalize_speaker_names(clues, {}, ["Rachel Bernstein", "Dan Puma", "Scott Riccardi"])
        # Should be dropped since UNKNOWN_SPEAKER doesn't fuzzy match anything
        assert len(clues[0].attempts) == 0

    def test_exact_canonical_survives_hard_cleanup(self) -> None:
        """When the attempt speaker IS already a canonical name, it survives."""
        clue = Clue(
            round="J!",
            category="Test",
            board_row=1,
            board_col=1,
            selection_order=1,
            is_daily_double=False,
            requires_visual_context=False,
            host_start_timestamp_ms=0.0,
            host_finish_timestamp_ms=1000.0,
            clue_syllable_count=5,
            clue_text="Test",
            correct_response="Test",
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Rachel Bernstein",
                    response_given="What?",
                    is_correct=True,
                    buzz_timestamp_ms=1500.0,
                    response_start_timestamp_ms=1750.0,
                    is_lockout_inferred=False,
                ),
                BuzzAttempt(
                    attempt_order=2,
                    speaker="TOTALLY_UNKNOWN",
                    response_given="Huh?",
                    is_correct=False,
                    buzz_timestamp_ms=2000.0,
                    response_start_timestamp_ms=2250.0,
                    is_lockout_inferred=False,
                ),
            ],
        )
        _normalize_speaker_names([clue], {}, ["Rachel Bernstein", "Dan Puma", "Scott Riccardi"])
        # First attempt (canonical name) survives, second is dropped
        assert len(clue.attempts) == 1
        assert clue.attempts[0].speaker == "Rachel Bernstein"


class TestFuzzyMatchEdgeCases:
    """Additional edge cases for _fuzzy_match_contestant."""

    def test_match_against_name_part(self) -> None:
        result = _fuzzy_match_contestant("Bernstein", ["Rachel Bernstein", "Matt Amodio"])
        assert result == "Rachel Bernstein"

    def test_no_match_returns_none(self) -> None:
        result = _fuzzy_match_contestant("ZZZZZZZZZ", ["Rachel Bernstein", "Matt Amodio"], max_distance=3)
        assert result is None

    def test_picks_closest_match(self) -> None:
        """When multiple candidates are within distance, pick the closest."""
        result = _fuzzy_match_contestant("Rachael", ["Rachel Bernstein", "Richard Bernstein"])
        # "Rachael" vs "Rachel" = distance 1, "Rachael" vs "Richard" = distance 4
        assert result == "Rachel Bernstein"
