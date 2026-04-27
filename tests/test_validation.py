"""Tests for clue deduplication and extraction integrity validation."""

from trebek.llm.validation import _validate_extraction_integrity, _deduplicate_clues
from trebek.schemas import (
    Episode,
    Clue,
    BuzzAttempt,
    Contestant,
    FinalJeopardy,
    FinalJeopardyWager,
)


def _make_clue(
    round: str = "Jeopardy",
    row: int = 1,
    col: int = 1,
    order: int = 1,
    start_ms: float = 0.0,
    finish_ms: float = 1000.0,
    dd: bool = False,
    attempts: list[BuzzAttempt] | None = None,
    text: str = "A clue",
) -> Clue:
    return Clue(
        round=round,
        category="History",
        board_row=row,
        board_col=col,
        selection_order=order,
        is_daily_double=dd,
        requires_visual_context=False,
        host_start_timestamp_ms=start_ms,
        host_finish_timestamp_ms=finish_ms,
        clue_syllable_count=10,
        clue_text=text,
        correct_response="Answer",
        attempts=attempts or [],
    )


def _make_episode(clues: list[Clue], num_dd_j: int = 1, num_dd_dj: int = 2) -> Episode:
    return Episode(
        episode_date="2024-01-01",
        host_name="Ken Jennings",
        is_tournament=False,
        contestants=[
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
        ],
        clues=clues,
        final_jeopardy=FinalJeopardy(
            category="Science",
            clue_text="Final clue",
            wagers_and_responses=[
                FinalJeopardyWager(contestant="Alice", wager=1000, response="What is X?", is_correct=True),
            ],
        ),
        score_adjustments=[],
    )


# ── Deduplication Tests ─────────────────────────────────────────────


class TestDeduplicateClues:
    def test_no_duplicates(self) -> None:
        clues = [_make_clue(start_ms=i * 1000, finish_ms=i * 1000 + 500, row=i + 1) for i in range(5)]
        result = _deduplicate_clues(clues)
        assert len(result) == 5

    def test_exact_duplicate_removed(self) -> None:
        """Two clues with identical time bucket + round + board position should deduplicate."""
        c1 = _make_clue(start_ms=1000, row=1, col=1, text="Short text")
        c2 = _make_clue(start_ms=1050, row=1, col=1, text="Longer clue text here")
        result = _deduplicate_clues([c1, c2])
        assert len(result) == 1
        # Should keep the longer text version
        assert "Longer" in result[0].clue_text

    def test_keeps_clue_with_more_attempts(self) -> None:
        """When deduplicating, prefer the clue with more buzz attempts."""
        att = BuzzAttempt(
            attempt_order=1,
            speaker="Alice",
            response_given="What?",
            is_correct=True,
            buzz_timestamp_ms=1500,
            response_start_timestamp_ms=1600,
            is_lockout_inferred=False,
        )
        c1 = _make_clue(start_ms=1000, row=2, col=3, attempts=[att])
        c2 = _make_clue(start_ms=1050, row=2, col=3, attempts=[])
        result = _deduplicate_clues([c1, c2])
        assert len(result) == 1
        assert len(result[0].attempts) == 1

    def test_different_rounds_not_deduplicated(self) -> None:
        """Clues in different rounds at similar times are NOT duplicates."""
        c1 = _make_clue(round="Jeopardy", start_ms=1000, row=1, col=1)
        c2 = _make_clue(round="Double Jeopardy", start_ms=1050, row=1, col=1)
        result = _deduplicate_clues([c1, c2])
        assert len(result) == 2


# ── Validation Tests ─────────────────────────────────────────────


class TestValidateExtractionIntegrity:
    def test_valid_episode_no_warnings(self) -> None:
        clues = []
        for i in range(25):
            clues.append(
                _make_clue(
                    round="Jeopardy",
                    row=(i % 5) + 1,
                    col=(i // 5) + 1,
                    order=i + 1,
                    start_ms=i * 2000,
                    finish_ms=i * 2000 + 1000,
                )
            )
        for i in range(25):
            clues.append(
                _make_clue(
                    round="Double Jeopardy",
                    row=(i % 5) + 1,
                    col=(i // 5) + 1,
                    order=26 + i,
                    start_ms=100000 + i * 2000,
                    finish_ms=100000 + i * 2000 + 1000,
                )
            )
        # Add one DD in Jeopardy and two in Double Jeopardy (replacing existing clues at valid positions)
        clues[3] = _make_clue(round="Jeopardy", row=4, col=1, order=4, start_ms=6000, finish_ms=7000, dd=True)
        clues[30] = _make_clue(
            round="Double Jeopardy", row=1, col=1, order=31, start_ms=110000, finish_ms=111000, dd=True
        )
        clues[35] = _make_clue(
            round="Double Jeopardy", row=1, col=2, order=36, start_ms=120000, finish_ms=121000, dd=True
        )

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
        att = BuzzAttempt(
            attempt_order=1,
            speaker="UnknownPlayer",
            response_given="What?",
            is_correct=True,
            buzz_timestamp_ms=1500,
            response_start_timestamp_ms=1600,
            is_lockout_inferred=False,
        )
        clues = [
            _make_clue(round="Jeopardy", row=1, col=1, order=1, attempts=[att]),
            _make_clue(round="Double Jeopardy", row=1, col=1, order=2, start_ms=5000),
        ]
        episode = _make_episode(clues)
        warnings = _validate_extraction_integrity(episode)
        assert any("Unknown contestants" in w for w in warnings)
