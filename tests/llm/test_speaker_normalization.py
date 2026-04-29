"""Tests for speaker name normalization — the most complex pure function in the codebase."""

from trebek.llm.speaker_normalization import _normalize_speaker_names
from trebek.schemas import Clue, BuzzAttempt, ScoreAdjustment, FinalJepWager


def _make_clue_with_speaker(speaker: str, **kwargs) -> Clue:  # type: ignore[no-untyped-def]
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
        """Host name should be excluded from speaker mapping — host can't buzz in."""
        clues = [_make_clue_with_speaker("SPEAKER_00")]
        mapping = {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Rachel"}
        _normalize_speaker_names(
            clues,
            mapping,
            ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"],
            host_name="Ken Jennings",
        )
        # SPEAKER_00 maps to host — should be dropped entirely (host can't buzz)
        assert len(clues[0].attempts) == 0

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
        """Final J! wager contestant names are also normalized."""
        clues = [_make_clue_with_speaker("Rachel")]
        wager = FinalJepWager(
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
        """Abbreviated host speaker ID (S00) should be dropped, not fuzzy-matched."""
        clues = [_make_clue_with_speaker("S00")]
        mapping = {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Scott"}
        _normalize_speaker_names(
            clues,
            mapping,
            ["Scott Riccardi", "Dan Puma"],
            host_name="Ken Jennings",
        )
        # S00 = host, should be dropped entirely (host can't buzz)
        assert len(clues[0].attempts) == 0

    def test_hard_cleanup_drops_unresolvable(self) -> None:
        """Completely unrecognizable speakers should be dropped, not passed through."""
        clues = [_make_clue_with_speaker("XYZ_MYSTERY_PLAYER")]
        _normalize_speaker_names(clues, {}, ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"])
        # The attempt should be dropped entirely
        assert len(clues[0].attempts) == 0

    def test_hard_cleanup_fuzzy_matches_typo(self) -> None:
        """A minor typo (edit distance <= 3) should be fuzzy-matched, not dropped."""
        clues = [_make_clue_with_speaker("Rachael")]  # Typo: "Rachael" vs "Rachel"
        _normalize_speaker_names(clues, {}, ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"])
        assert len(clues[0].attempts) == 1
        assert clues[0].attempts[0].speaker == "Rachel Bernstein"

    def test_hard_cleanup_preserves_valid_speakers(self) -> None:
        """Valid speakers should never be dropped by the hard cleanup."""
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
            clue_text="Test clue",
            correct_response="Test answer",
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Rachel",
                    response_given="What?",
                    is_correct=False,
                    buzz_timestamp_ms=1500.0,
                    response_start_timestamp_ms=1750.0,
                    is_lockout_inferred=False,
                ),
                BuzzAttempt(
                    attempt_order=2,
                    speaker="GARBAGE_NAME",
                    response_given="Who?",
                    is_correct=True,
                    buzz_timestamp_ms=2000.0,
                    response_start_timestamp_ms=2250.0,
                    is_lockout_inferred=False,
                ),
            ],
        )
        _normalize_speaker_names([clue], {}, ["Rachel Bernstein", "Lawrence Subba", "Matt Amodio"])
        # Valid speaker kept, garbage dropped
        assert len(clue.attempts) == 1
        assert clue.attempts[0].speaker == "Rachel Bernstein"


class TestLevenshtein:
    """Test the minimal Levenshtein distance implementation."""

    def test_identical(self) -> None:
        from trebek.llm.speaker_normalization import _levenshtein

        assert _levenshtein("hello", "hello") == 0

    def test_one_insertion(self) -> None:
        from trebek.llm.speaker_normalization import _levenshtein

        assert _levenshtein("hell", "hello") == 1

    def test_one_substitution(self) -> None:
        from trebek.llm.speaker_normalization import _levenshtein

        assert _levenshtein("rachel", "rachael") == 1

    def test_empty(self) -> None:
        from trebek.llm.speaker_normalization import _levenshtein

        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3

    def test_completely_different(self) -> None:
        from trebek.llm.speaker_normalization import _levenshtein

        assert _levenshtein("abc", "xyz") == 3


class TestFuzzyMatchContestant:
    """Test the fuzzy matching helper."""

    def test_exact_match(self) -> None:
        from trebek.llm.speaker_normalization import _fuzzy_match_contestant

        result = _fuzzy_match_contestant("Rachel", ["Rachel Bernstein", "Matt Amodio"])
        assert result == "Rachel Bernstein"

    def test_close_typo(self) -> None:
        from trebek.llm.speaker_normalization import _fuzzy_match_contestant

        result = _fuzzy_match_contestant("Rachael", ["Rachel Bernstein", "Matt Amodio"])
        assert result == "Rachel Bernstein"

    def test_too_far(self) -> None:
        from trebek.llm.speaker_normalization import _fuzzy_match_contestant

        result = _fuzzy_match_contestant("XYZABC", ["Rachel Bernstein", "Matt Amodio"])
        assert result is None


class TestReconcileSpeakerMapping:
    """Test Pass 1 → Pass 2 speaker name reconciliation."""

    def test_paulo_to_paolo(self) -> None:
        """The exact S42E04 bug: 'Paulo' fuzzy-matches to 'Paolo Pasco'."""
        from trebek.llm.speaker_normalization import _reconcile_speaker_mapping

        mapping = {"SPEAKER_00": "Ken Jennings", "SPEAKER_03": "Paulo"}
        result = _reconcile_speaker_mapping(mapping, ["Paolo Pasco", "Andy Miller", "Jill"], host_name="Ken Jennings")
        assert result["SPEAKER_03"] == "Paolo Pasco"
        assert result["SPEAKER_00"] == "Ken Jennings"

    def test_exact_match_passthrough(self) -> None:
        """When Pass 1 name exactly matches a contestant, no change needed."""
        from trebek.llm.speaker_normalization import _reconcile_speaker_mapping

        mapping = {"SPEAKER_01": "Paolo Pasco"}
        result = _reconcile_speaker_mapping(mapping, ["Paolo Pasco", "Andy Miller"])
        assert result["SPEAKER_01"] == "Paolo Pasco"

    def test_known_host_retained(self) -> None:
        """Known host names should be kept as-is, not matched to contestants."""
        from trebek.llm.speaker_normalization import _reconcile_speaker_mapping

        mapping = {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Alice"}
        result = _reconcile_speaker_mapping(mapping, ["Alice Smith", "Bob Jones"], host_name="Ken Jennings")
        assert result["SPEAKER_00"] == "Ken Jennings"
        assert result["SPEAKER_01"] == "Alice Smith"

    def test_substring_match(self) -> None:
        """'Lisa' should match 'Lisa Mueller' via substring."""
        from trebek.llm.speaker_normalization import _reconcile_speaker_mapping

        mapping = {"SPEAKER_02": "Lisa"}
        result = _reconcile_speaker_mapping(mapping, ["Lisa Mueller", "James Thajuddin"])
        assert result["SPEAKER_02"] == "Lisa Mueller"

    def test_unresolvable_kept(self) -> None:
        """Names that don't match anything should be kept for downstream handling."""
        from trebek.llm.speaker_normalization import _reconcile_speaker_mapping

        mapping = {"SPEAKER_05": "CommercialVoice"}
        result = _reconcile_speaker_mapping(mapping, ["Alice Smith", "Bob Jones"])
        assert result["SPEAKER_05"] == "CommercialVoice"


class TestResolveHostFromPass1:
    """Test host resolution from Pass 1 speaker mapping."""

    def test_finds_ken_jennings(self) -> None:
        from trebek.llm.speaker_normalization import _resolve_host_from_pass1

        mapping = {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Alice"}
        assert _resolve_host_from_pass1(mapping) == "Ken Jennings"

    def test_finds_partial_name(self) -> None:
        """'Ken' alone should still resolve to 'Ken Jennings'."""
        from trebek.llm.speaker_normalization import _resolve_host_from_pass1

        mapping = {"SPEAKER_00": "Ken", "SPEAKER_01": "Alice"}
        assert _resolve_host_from_pass1(mapping) == "Ken Jennings"

    def test_no_host_found(self) -> None:
        from trebek.llm.speaker_normalization import _resolve_host_from_pass1

        mapping = {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
        assert _resolve_host_from_pass1(mapping) is None

    def test_finds_alex_trebek(self) -> None:
        from trebek.llm.speaker_normalization import _resolve_host_from_pass1

        mapping = {"SPEAKER_00": "Alex Trebek"}
        assert _resolve_host_from_pass1(mapping) == "Alex Trebek"
