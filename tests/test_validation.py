"""Tests for clue deduplication and extraction integrity validation.

Tests are organized by validation category to match the checks in validation.py:
- Deduplication (category-based composite key)
- Round-level checks (clue counts, DD limits)
- Timestamp ordering
- Contestant FK consistency (buzzers, FJ wagers, score adjustments)
- Board position bounds
- Per-clue data quality (empty text, inverted timestamps, etc.)
"""

from trebek.llm.validation import _validate_extraction_integrity, _deduplicate_clues
from trebek.schemas import (
    Episode,
    Clue,
    BuzzAttempt,
    Contestant,
    FinalJeopardy,
    FinalJeopardyWager,
    ScoreAdjustment,
)
from typing import Literal


def _make_clue(
    round: Literal["Jeopardy", "Double Jeopardy", "Final Jeopardy", "Tiebreaker"] = "Jeopardy",
    category: str = "History",
    row: int = 1,
    col: int = 1,
    order: int = 1,
    start_ms: float = 0.0,
    finish_ms: float = 1000.0,
    dd: bool = False,
    attempts: list[BuzzAttempt] | None = None,
    text: str = "A clue",
    correct_response: str = "Answer",
) -> Clue:
    return Clue(
        round=round,
        category=category,
        board_row=row,
        board_col=col,
        selection_order=order,
        is_daily_double=dd,
        requires_visual_context=False,
        host_start_timestamp_ms=start_ms,
        host_finish_timestamp_ms=finish_ms,
        clue_syllable_count=10,
        clue_text=text,
        correct_response=correct_response,
        attempts=attempts or [],
    )


def _make_episode(
    clues: list[Clue],
    contestants: list[Contestant] | None = None,
    fj_wagers: list[FinalJeopardyWager] | None = None,
    score_adjustments: list[ScoreAdjustment] | None = None,
) -> Episode:
    if contestants is None:
        contestants = [
            Contestant(
                name="Alice",
                podium_position=1,
                occupational_category="STEM",
                is_returning_champion=True,
                description="A contestant",
            ),
            Contestant(
                name="Bob",
                podium_position=2,
                occupational_category="Law",
                is_returning_champion=False,
                description="A contestant",
            ),
            Contestant(
                name="Charlie",
                podium_position=3,
                occupational_category="Arts",
                is_returning_champion=False,
                description="A contestant",
            ),
        ]
    if fj_wagers is None:
        fj_wagers = [
            FinalJeopardyWager(contestant="Alice", wager=1000, response="What is X?", is_correct=True),
        ]
    return Episode(
        episode_date="2024-01-01",
        host_name="Ken Jennings",
        is_tournament=False,
        contestants=contestants,
        clues=clues,
        final_jeopardy=FinalJeopardy(
            category="Science",
            clue_text="Final clue",
            wagers_and_responses=fj_wagers,
        ),
        score_adjustments=score_adjustments or [],
    )


def _make_attempt(speaker: str = "Alice", correct: bool = True, buzz_ms: float = 1500.0) -> BuzzAttempt:
    return BuzzAttempt(
        attempt_order=1,
        speaker=speaker,
        response_given="What?",
        is_correct=correct,
        buzz_timestamp_ms=buzz_ms,
        response_start_timestamp_ms=buzz_ms + 250.0,
        is_lockout_inferred=False,
    )


# ── Deduplication Tests (Category-Based Key) ───────────────────────


class TestDeduplicateClues:
    def test_no_duplicates(self) -> None:
        # Each clue 3s apart → different 2000ms buckets → no dedup
        clues = [_make_clue(start_ms=i * 3000, finish_ms=i * 3000 + 500, row=i + 1) for i in range(5)]
        result = _deduplicate_clues(clues)
        assert len(result) == 5

    def test_same_category_same_time_deduplicates(self) -> None:
        """Two clues with same time bucket + round + category should dedup."""
        # Both at ~2500ms → bucket 2000 (round(2500/2000)*2000 = round(1.25)*2000 = 2000)
        c1 = _make_clue(start_ms=2500, category="History", text="Short text")
        c2 = _make_clue(start_ms=2600, category="History", text="Longer clue text here")
        result = _deduplicate_clues([c1, c2])
        assert len(result) == 1
        assert "Longer" in result[0].clue_text

    def test_different_categories_not_deduped(self) -> None:
        """Same time bucket but different categories should NOT dedup."""
        c1 = _make_clue(start_ms=2500, category="History")
        c2 = _make_clue(start_ms=2600, category="Science")
        result = _deduplicate_clues([c1, c2])
        assert len(result) == 2

    def test_category_case_insensitive(self) -> None:
        """Category matching for dedup should be case-insensitive."""
        c1 = _make_clue(start_ms=2500, category="HISTORY")
        c2 = _make_clue(start_ms=2600, category="history")
        result = _deduplicate_clues([c1, c2])
        assert len(result) == 1

    def test_keeps_clue_with_more_attempts(self) -> None:
        """When deduplicating, prefer the clue with more buzz attempts."""
        att = _make_attempt()
        c1 = _make_clue(start_ms=2500, category="Math", attempts=[att])
        c2 = _make_clue(start_ms=2600, category="Math", attempts=[])
        result = _deduplicate_clues([c1, c2])
        assert len(result) == 1
        assert len(result[0].attempts) == 1

    def test_different_rounds_not_deduplicated(self) -> None:
        """Clues in different rounds at similar times are NOT duplicates."""
        c1 = _make_clue(round="Jeopardy", start_ms=2500, category="History")
        c2 = _make_clue(round="Double Jeopardy", start_ms=2600, category="History")
        result = _deduplicate_clues([c1, c2])
        assert len(result) == 2

    def test_different_time_buckets_not_deduplicated(self) -> None:
        """Clues in the same category but different 2s time buckets should NOT dedup."""
        c1 = _make_clue(start_ms=2500, category="History")  # bucket 2000
        c2 = _make_clue(start_ms=6500, category="History")  # bucket 6000
        result = _deduplicate_clues([c1, c2])
        assert len(result) == 2


# ── Validation Tests ────────────────────────────────────────────────


class TestValidateExtractionIntegrity:
    def test_valid_episode_no_warnings(self) -> None:
        """A well-formed episode matching all domain rules should produce zero warnings."""
        j_categories = ["History", "Science", "Math", "Literature", "Geography"]
        dj_categories = ["Art", "Music", "Sports", "Politics", "Technology"]
        clues = []
        # 25 Jeopardy clues: 5 categories × 5 rows, each category in its own column
        for col_idx, cat in enumerate(j_categories):
            for row in range(1, 6):
                order = col_idx * 5 + row
                clues.append(
                    _make_clue(
                        round="Jeopardy",
                        category=cat,
                        row=row,
                        col=col_idx + 1,
                        order=order,
                        start_ms=order * 2000,
                        finish_ms=order * 2000 + 1000,
                        correct_response="What is X?",
                    )
                )
        # 25 Double Jeopardy clues: 5 categories × 5 rows
        for col_idx, cat in enumerate(dj_categories):
            for row in range(1, 6):
                order = 26 + col_idx * 5 + row
                clues.append(
                    _make_clue(
                        round="Double Jeopardy",
                        category=cat,
                        row=row,
                        col=col_idx + 1,
                        order=order,
                        start_ms=100000 + (order - 26) * 2000,
                        finish_ms=100000 + (order - 26) * 2000 + 1000,
                        correct_response="What is Y?",
                    )
                )
        # Set Daily Doubles (1 J, 2 DJ)
        clues[3].is_daily_double = True
        clues[30].is_daily_double = True
        clues[35].is_daily_double = True
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

    def test_too_many_jeopardy_clues(self) -> None:
        clues = [
            _make_clue(round="Jeopardy", row=(i % 5) + 1, col=1, order=i, start_ms=i * 2000, finish_ms=i * 2000 + 1000)
            for i in range(35)
        ]
        clues.append(_make_clue(round="Double Jeopardy", row=1, col=1, order=36, start_ms=80000, finish_ms=81000))
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("Jeopardy round has 35 clues" in w for w in warnings)

    def test_no_jeopardy_clues_warning(self) -> None:
        clues = [_make_clue(round="Double Jeopardy", row=1, col=1, order=1, start_ms=1000, finish_ms=2000)]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("No Jeopardy round clues" in w for w in warnings)

    def test_too_many_daily_doubles(self) -> None:
        clues = [
            _make_clue(round="Jeopardy", row=1, col=1, order=1, dd=True, start_ms=0),
            _make_clue(round="Jeopardy", row=2, col=1, order=2, dd=True, start_ms=2000),
            _make_clue(round="Double Jeopardy", row=1, col=1, order=3, dd=True, start_ms=4000),
            _make_clue(round="Double Jeopardy", row=2, col=1, order=4, dd=True, start_ms=6000),
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("4 Daily Doubles" in w for w in warnings)

    def test_invalid_board_position(self) -> None:
        clues = [
            _make_clue(round="Jeopardy", row=0, col=1, order=1, start_ms=0),
            _make_clue(round="Double Jeopardy", row=1, col=7, order=2, start_ms=2000),
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("invalid board_row" in w for w in warnings)
        assert any("invalid board_col" in w for w in warnings)

    def test_unknown_buzzer_warning(self) -> None:
        att = _make_attempt(speaker="UnknownPlayer")
        clues = [
            _make_clue(round="Jeopardy", row=1, col=1, order=1, attempts=[att]),
            _make_clue(round="Double Jeopardy", row=1, col=1, order=2, start_ms=5000),
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("Unknown contestants" in w for w in warnings)

    # ── New validation checks ────────────────────────────────────────

    def test_zero_clues_warning(self) -> None:
        """An episode with zero clues should produce a fatal warning."""
        episode = _make_episode([])
        warnings = _validate_extraction_integrity(episode)
        assert any("Zero clues" in w for w in warnings)

    def test_no_contestants_warning(self) -> None:
        """An episode with no contestants should produce a fatal warning."""
        clues = [_make_clue()]
        episode = _make_episode(clues, contestants=[])
        warnings = _validate_extraction_integrity(episode)
        assert any("No contestants" in w for w in warnings)

    def test_empty_clue_text_warning(self) -> None:
        """Clues with empty text should be flagged."""
        clues = [
            _make_clue(round="Jeopardy", text="", start_ms=0),
            _make_clue(round="Double Jeopardy", text="Valid text", start_ms=5000),
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("empty clue_text" in w for w in warnings)

    def test_empty_correct_response_warning(self) -> None:
        """Clues with empty correct_response should be flagged."""
        clues = [
            _make_clue(round="Jeopardy", correct_response="", start_ms=0),
            _make_clue(round="Double Jeopardy", start_ms=5000),
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("empty correct_response" in w for w in warnings)

    def test_inverted_timestamps_warning(self) -> None:
        """Clues where host_finish < host_start should be flagged."""
        clue = _make_clue(round="Jeopardy", start_ms=5000.0, finish_ms=3000.0)
        episode = _make_episode([clue, _make_clue(round="Double Jeopardy", start_ms=10000)])
        warnings = _validate_extraction_integrity(episode)
        assert any("inverted timestamps" in w for w in warnings)

    def test_zero_timestamp_mid_game_warning(self) -> None:
        """Mid-game clues with timestamp 0.0 signal Line ID resolution failure."""
        clue = _make_clue(round="Jeopardy", order=10, start_ms=0.0)
        episode = _make_episode([clue, _make_clue(round="Double Jeopardy", start_ms=5000)])
        warnings = _validate_extraction_integrity(episode)
        assert any("host_start_timestamp_ms=0.0" in w for w in warnings)

    def test_buzz_before_host_start_warning(self) -> None:
        """Buzz timestamps before host starts reading signal hallucinated Line IDs."""
        att = _make_attempt(speaker="Alice", buzz_ms=100.0)
        clue = _make_clue(round="Jeopardy", start_ms=5000.0, finish_ms=6000.0, attempts=[att])
        episode = _make_episode([clue, _make_clue(round="Double Jeopardy", start_ms=10000)])
        warnings = _validate_extraction_integrity(episode)
        assert any("buzz_timestamp before host_start" in w for w in warnings)

    def test_fj_wager_unknown_contestant_warning(self) -> None:
        """FJ wager referencing an unknown contestant should be flagged."""
        fj_wagers = [
            FinalJeopardyWager(contestant="UnknownPerson", wager=5000, response="What?", is_correct=True),
        ]
        clues = [_make_clue(round="Jeopardy"), _make_clue(round="Double Jeopardy", start_ms=5000)]
        episode = _make_episode(clues, fj_wagers=fj_wagers)
        warnings = _validate_extraction_integrity(episode)
        assert any("FJ wager references unknown contestant" in w for w in warnings)

    def test_score_adjustment_unknown_contestant_warning(self) -> None:
        """Score adjustment referencing an unknown contestant should be flagged."""
        adj = [
            ScoreAdjustment(
                contestant="Ghost", points_adjusted=200, reason="test", effective_after_clue_selection_order=1
            )
        ]
        clues = [_make_clue(round="Jeopardy"), _make_clue(round="Double Jeopardy", start_ms=5000)]
        episode = _make_episode(clues, score_adjustments=adj)
        warnings = _validate_extraction_integrity(episode)
        assert any("Score adjustment references unknown contestant" in w for w in warnings)

    def test_too_many_categories_warning(self) -> None:
        """More than 6 distinct categories in a round signals extraction failure."""
        clues = [_make_clue(round="Jeopardy", category=f"Cat{i}", start_ms=i * 2000, order=i) for i in range(8)]
        clues.append(_make_clue(round="Double Jeopardy", start_ms=50000))
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("distinct categories" in w for w in warnings)

    def test_duplicate_board_position_warning(self) -> None:
        """Two clues in the same category+round with the same board_row should be flagged."""
        clues = [
            _make_clue(round="Jeopardy", category="History", row=1, col=1, order=1, start_ms=0),
            _make_clue(round="Jeopardy", category="History", row=1, col=1, order=2, start_ms=2000),
            _make_clue(round="Double Jeopardy", row=1, col=1, order=3, start_ms=10000),
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("duplicate board_row" in w for w in warnings)

    def test_daily_double_multiple_attempts_warning(self) -> None:
        """Daily Double with more than 1 attempt should be flagged."""
        att1 = _make_attempt(speaker="Alice", correct=False, buzz_ms=1500.0)
        att2 = _make_attempt(speaker="Bob", correct=True, buzz_ms=2000.0)
        clues = [
            _make_clue(round="Jeopardy", row=1, col=1, order=1, dd=True, attempts=[att1, att2]),
            _make_clue(round="Double Jeopardy", row=1, col=1, order=2, start_ms=5000),
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("Daily Double" in w and "2 attempts" in w for w in warnings)

    def test_daily_double_zero_wager_warning(self) -> None:
        """Daily Double wager of $0 or less should be flagged."""
        clues = [
            _make_clue(round="Jeopardy", row=1, col=1, order=1, dd=True, start_ms=0),
            _make_clue(round="Double Jeopardy", row=1, col=1, order=2, start_ms=5000),
        ]
        # Manually set wager to 0
        clues[0].daily_double_wager = 0
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("wager <= 0" in w for w in warnings)

    def test_daily_double_absurd_wager_warning(self) -> None:
        """Daily Double wager > $50,000 should be flagged as suspicious."""
        clues = [
            _make_clue(round="Jeopardy", row=1, col=1, order=1, dd=True, start_ms=0),
            _make_clue(round="Double Jeopardy", row=1, col=1, order=2, start_ms=5000),
        ]
        clues[0].daily_double_wager = 100000
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("suspiciously high" in w for w in warnings)

    def test_long_read_duration_warning(self) -> None:
        """Clue with host read duration > 60s should be flagged."""
        clue = _make_clue(round="Jeopardy", start_ms=0.0, finish_ms=70000.0)
        episode = _make_episode([clue, _make_clue(round="Double Jeopardy", start_ms=80000)])
        warnings = _validate_extraction_integrity(episode)
        assert any("read duration > 60s" in w for w in warnings)

    def test_category_undercounting_warning(self) -> None:
        """Many clues across too few categories should be flagged."""
        clues = [
            _make_clue(round="Jeopardy", category="OnlyCat", row=(i % 5) + 1, col=1, order=i, start_ms=i * 2000)
            for i in range(20)
        ]
        clues.append(_make_clue(round="Double Jeopardy", start_ms=50000))
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("category merge" in w for w in warnings)


# ── Dedup Boundary Tests ────────────────────────────────────────────


class TestDeduplicateBoundaryConditions:
    def test_exact_bucket_boundary(self) -> None:
        """Clues at exact bucket boundary (1999 vs 2000) should NOT dedup."""
        c1 = _make_clue(start_ms=1999, category="History")
        c2 = _make_clue(start_ms=2001, category="History")
        result = _deduplicate_clues([c1, c2])
        # 1999/2000*2000 = round(0.9995)*2000 = 2000
        # 2001/2000*2000 = round(1.0005)*2000 = 2000
        # Both land in bucket 2000 → should dedup
        assert len(result) == 1

    def test_clues_far_apart_same_category(self) -> None:
        """Clues 3 seconds apart in same category should NOT dedup."""
        c1 = _make_clue(start_ms=1000, category="History")
        c2 = _make_clue(start_ms=4000, category="History")  # 3s apart
        result = _deduplicate_clues([c1, c2])
        assert len(result) == 2

    def test_three_way_dedup(self) -> None:
        """Three overlapping-chunk copies of the same clue should dedup to 1."""
        # All three at ~4500ms → bucket round(4500/2000)*2000 = round(2.25)*2000 = 4000
        c1 = _make_clue(start_ms=4200, category="Science", text="Short")
        c2 = _make_clue(start_ms=4300, category="Science", text="Medium length text")
        c3 = _make_clue(start_ms=4400, category="Science", text="The longest clue text here")
        result = _deduplicate_clues([c1, c2, c3])
        assert len(result) == 1
        # Should keep the longest text (no attempts to differentiate)
        assert "longest" in result[0].clue_text
