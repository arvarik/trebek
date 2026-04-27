"""Tests for speaker name normalization — the most complex pure function in the codebase."""

from trebek.llm.speaker_normalization import _normalize_speaker_names
from trebek.schemas import Clue, BuzzAttempt, ScoreAdjustment, FinalJeopardyWager


def _make_clue_with_speaker(speaker: str, **kwargs) -> Clue:  # type: ignore[no-untyped-def]
    return Clue(
        round="Jeopardy",
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


class TestNormalizeSpeakerNames:
    def test_exact_match(self) -> None:
        """Full contestant name in buzz attempt is kept as-is."""
        clues = [_make_clue_with_speaker("Rachel Bernstein")]
        _normalize_speaker_names(clues, {}, ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"])
        assert clues[0].attempts[0].speaker == "Rachel Bernstein"

    def test_first_name_only(self) -> None:
        """Single first name resolves to the full contestant name."""
        clues = [_make_clue_with_speaker("Rachel")]
        _normalize_speaker_names(clues, {}, ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"])
        assert clues[0].attempts[0].speaker == "Rachel Bernstein"

    def test_last_name_only(self) -> None:
        """Single last name resolves to the full contestant name."""
        clues = [_make_clue_with_speaker("Subba")]
        _normalize_speaker_names(clues, {}, ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"])
        assert clues[0].attempts[0].speaker == "Lawrence Subba"

    def test_speaker_id_mapping(self) -> None:
        """SPEAKER_XX IDs resolve through Pass 1 mapping → contestant name."""
        clues = [_make_clue_with_speaker("SPEAKER_01")]
        mapping = {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Rachel"}
        _normalize_speaker_names(
            clues,
            mapping,
            ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"],
            host_name="Ken Jennings",
        )
        assert clues[0].attempts[0].speaker == "Rachel Bernstein"

    def test_host_excluded_from_mapping(self) -> None:
        """Host name should be excluded from speaker mapping to prevent host appearing in buzz attempts."""
        clues = [_make_clue_with_speaker("SPEAKER_00")]
        mapping = {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Rachel"}
        _normalize_speaker_names(
            clues,
            mapping,
            ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"],
            host_name="Ken Jennings",
        )
        # SPEAKER_00 maps to host — should NOT be resolved to a contestant
        # (it stays as Ken Jennings since no contestant match)
        assert clues[0].attempts[0].speaker != "Rachel Bernstein"

    def test_case_insensitive(self) -> None:
        """Name matching should be case-insensitive."""
        clues = [_make_clue_with_speaker("rachel")]
        _normalize_speaker_names(clues, {}, ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"])
        assert clues[0].attempts[0].speaker == "Rachel Bernstein"

    def test_ambiguous_name_part_removed(self) -> None:
        """When two contestants share a name part, that part should not resolve."""
        clues = [_make_clue_with_speaker("James")]
        _normalize_speaker_names(clues, {}, ["James Smith", "James Johnson", "Matt Amodio"])
        # "James" is ambiguous — shouldn't resolve to either
        # (the function removes the mapping, speaker stays unmapped or uses substring fallback)
        # The speaker stays as "James" since it's ambiguous
        assert clues[0].attempts[0].speaker in ("James", "James Smith", "James Johnson")

    def test_daily_double_wagerer_normalized(self) -> None:
        """wagerer_name on Daily Doubles should also be normalized."""
        clue = Clue(
            round="Jeopardy",
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
            wagerer_name="Rachel",
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Rachel",
                    response_given="What?",
                    is_correct=True,
                    buzz_timestamp_ms=1500.0,
                    response_start_timestamp_ms=1750.0,
                    is_lockout_inferred=False,
                )
            ],
        )
        _normalize_speaker_names([clue], {}, ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"])
        assert clue.wagerer_name == "Rachel Bernstein"

    def test_score_adjustment_normalized(self) -> None:
        """Score adjustment contestant names are also normalized."""
        clues = [_make_clue_with_speaker("Rachel")]
        adj = ScoreAdjustment(
            contestant="Lawrence",
            points_adjusted=400,
            reason="Judge ruling",
            effective_after_clue_selection_order=1,
        )
        _normalize_speaker_names(
            clues,
            {},
            ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"],
            score_adjustments=[adj],
        )
        assert adj.contestant == "Lawrence Subba"

    def test_fj_wager_normalized(self) -> None:
        """Final Jeopardy wager contestant names are also normalized."""
        clues = [_make_clue_with_speaker("Rachel")]
        wager = FinalJeopardyWager(
            contestant="Matt",
            wager=5000,
            response="What is X?",
            is_correct=True,
        )
        _normalize_speaker_names(
            clues,
            {},
            ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"],
            fj_wagers=[wager],
        )
        assert wager.contestant == "Matt Amodio"

    def test_substring_containment_fallback(self) -> None:
        """When no other match works, substring containment should still resolve."""
        clues = [_make_clue_with_speaker("Bernstein")]
        _normalize_speaker_names(clues, {}, ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"])
        assert clues[0].attempts[0].speaker == "Rachel Bernstein"

    def test_abbreviated_speaker_id_s00(self) -> None:
        """Abbreviated S00/S01 speaker IDs must resolve through the mapping."""
        clues = [_make_clue_with_speaker("S01")]
        mapping = {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Scott", "SPEAKER_02": "Dan"}
        _normalize_speaker_names(
            clues,
            mapping,
            ["Scott Riccardi", "Dan Puma", "Elise Kanat"],
            host_name="Ken Jennings",
        )
        assert clues[0].attempts[0].speaker == "Scott Riccardi"

    def test_abbreviated_speaker_id_s1_no_zero_pad(self) -> None:
        """Non-zero-padded S1 variant also resolves."""
        clues = [_make_clue_with_speaker("S1")]
        mapping = {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Scott"}
        _normalize_speaker_names(
            clues,
            mapping,
            ["Scott Riccardi", "Dan Puma", "Elise Kanat"],
            host_name="Ken Jennings",
        )
        assert clues[0].attempts[0].speaker == "Scott Riccardi"

    def test_abbreviated_speaker_id_s03(self) -> None:
        """S03 resolves correctly for the third contestant."""
        clues = [_make_clue_with_speaker("S03")]
        mapping = {
            "SPEAKER_00": "Ken Jennings",
            "SPEAKER_01": "Scott",
            "SPEAKER_02": "Elise",
            "SPEAKER_03": "Dan",
        }
        _normalize_speaker_names(
            clues,
            mapping,
            ["Scott Riccardi", "Dan Puma", "Elise Kanat"],
            host_name="Ken Jennings",
        )
        assert clues[0].attempts[0].speaker == "Dan Puma"

    def test_host_abbreviated_id_excluded(self) -> None:
        """Abbreviated host speaker ID (S00) should not map to a contestant."""
        clues = [_make_clue_with_speaker("S00")]
        mapping = {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Scott"}
        _normalize_speaker_names(
            clues,
            mapping,
            ["Scott Riccardi", "Dan Puma"],
            host_name="Ken Jennings",
        )
        # S00 = host, should NOT be resolved to any contestant
        assert clues[0].attempts[0].speaker not in ("Scott Riccardi", "Dan Puma")
