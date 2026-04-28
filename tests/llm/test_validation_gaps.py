"""
Tests targeting uncovered validation branches — timestamp overlap warnings,
DJ Daily Double count, and dedup tie-breaking by text length.
"""

from trebek.schemas import Clue, BuzzAttempt, Episode, Contestant, FinalJep
from trebek.llm.validation import _validate_extraction_integrity, _deduplicate_clues


from typing import Literal


def _make_clue(
    round: Literal["J!", "Double J!", "Final J!", "Tiebreaker"] = "J!",
    category: str = "Test",
    row: int = 1,
    col: int = 1,
    selection_order: int = 1,
    host_start: float = 0.0,
    host_finish: float = 1000.0,
    is_dd: bool = False,
    clue_text: str = "Test clue",
    attempts: list[BuzzAttempt] | None = None,
) -> Clue:
    return Clue(
        round=round,
        category=category,
        board_row=row,
        board_col=col,
        selection_order=selection_order,
        is_daily_double=is_dd,
        requires_visual_context=False,
        host_start_timestamp_ms=host_start,
        host_finish_timestamp_ms=host_finish,
        clue_syllable_count=5,
        clue_text=clue_text,
        correct_response="Test answer",
        attempts=attempts or [],
    )


def _make_episode(clues: list[Clue]) -> Episode:
    return Episode(
        episode_date="2025-01-01",
        host_name="Ken Jennings",
        is_tournament=False,
        contestants=[
            Contestant(
                name="Alice",
                podium_position=1,
                is_returning_champion=False,
                occupational_category="STEM",
                description="from NY",
            ),
            Contestant(
                name="Bob",
                podium_position=2,
                is_returning_champion=False,
                occupational_category="Law",
                description="from LA",
            ),
            Contestant(
                name="Carol",
                podium_position=3,
                is_returning_champion=False,
                occupational_category="Arts",
                description="from TX",
            ),
        ],
        clues=clues,
        final_jep=FinalJep(
            category="Final Category",
            clue_text="Final clue",
            wagers_and_responses=[],
        ),
        score_adjustments=[],
    )


class TestTimestampOverlapWarning:
    """Line 59-65: Overlapping timestamps within a round trigger warnings."""

    def test_overlapping_j_clues_produce_warning(self) -> None:
        clues = [
            _make_clue(round="J!", row=1, col=1, selection_order=1, host_start=0.0, host_finish=5000.0),
            _make_clue(round="J!", row=1, col=2, selection_order=2, host_start=3000.0, host_finish=8000.0),  # Overlaps!
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("overlaps with clue" in w for w in warnings)

    def test_non_overlapping_clues_no_warning(self) -> None:
        clues = [
            _make_clue(round="J!", row=1, col=1, selection_order=1, host_start=0.0, host_finish=5000.0),
            _make_clue(round="J!", row=1, col=2, selection_order=2, host_start=6000.0, host_finish=10000.0),
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert not any("overlaps with clue" in w for w in warnings)


class TestDailyDoubleCount:
    """Lines 43, 78: Warnings for excess DJ round clues and DD counts."""

    def test_excess_dj_clues_warning(self) -> None:
        clues = [
            _make_clue(
                round="Double J!",
                row=i % 5 + 1,
                col=i % 6 + 1,
                selection_order=i,
                host_start=i * 1000.0,
                host_finish=(i + 1) * 1000.0,
            )
            for i in range(31)
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("Double J! round has 31 clues" in w for w in warnings)

    def test_excess_dj_daily_doubles_warning(self) -> None:
        clues = [
            _make_clue(
                round="Double J!",
                row=i + 1,
                col=1,
                selection_order=i,
                is_dd=True,
                host_start=i * 1000.0,
                host_finish=(i + 1) * 1000.0,
            )
            for i in range(3)
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("3 Daily Doubles in Double J!" in w for w in warnings)


class TestDedupTieBreakByTextLength:
    """Lines 250-252: When attempts are equal, longer clue_text wins."""

    def test_longer_text_wins_dedup(self) -> None:
        clue_short = _make_clue(round="J!", category="History", row=1, col=1, host_start=1000.0, clue_text="Short")
        clue_long = _make_clue(
            round="J!",
            category="History",
            row=1,
            col=1,
            host_start=1000.0,
            clue_text="A much longer and more detailed clue text",
        )
        result = _deduplicate_clues([clue_short, clue_long])
        assert len(result) == 1
        assert "longer" in result[0].clue_text

    def test_same_length_keeps_first(self) -> None:
        clue_a = _make_clue(round="J!", category="Science", row=2, col=1, host_start=2000.0, clue_text="AAAA")
        clue_b = _make_clue(round="J!", category="Science", row=2, col=1, host_start=2000.0, clue_text="BBBB")
        result = _deduplicate_clues([clue_a, clue_b])
        assert len(result) == 1
        assert result[0].clue_text == "AAAA"  # First one kept
