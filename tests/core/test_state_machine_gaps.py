"""
Tests targeting uncovered state machine branches —
unknown round type and Daily Double without wager (lines 51, 55).
"""

from trebek.state_machine import TrebekStateMachine
from trebek.schemas import Clue, BuzzAttempt


from typing import Literal


def _make_clue(
    round: Literal["J!", "Double J!", "Final J!", "Tiebreaker"] = "J!",
    row: int = 1,
    col: int = 1,
    selection_order: int = 1,
    is_dd: bool = False,
    dd_wager: int | None = None,
    wagerer: str | None = None,
    attempts: list[BuzzAttempt] | None = None,
) -> Clue:
    return Clue(
        round=round,
        category="Test",
        board_row=row,
        board_col=col,
        selection_order=selection_order,
        is_daily_double=is_dd,
        requires_visual_context=False,
        host_start_timestamp_ms=0.0,
        host_finish_timestamp_ms=1000.0,
        clue_syllable_count=5,
        clue_text="Test clue",
        correct_response="Test answer",
        daily_double_wager=dd_wager,
        wagerer_name=wagerer,
        attempts=attempts or [],
    )


class TestUnknownRoundType:
    """Line 51: clue.round not J! or Double J! → clue_value = 0."""

    def test_final_jep_clue_value_zero(self) -> None:
        sm = TrebekStateMachine(
            initial_scores={"Alice": 0, "Bob": 0, "Carol": 0},
            valid_contestants={"Alice", "Bob", "Carol"},
        )
        clue = _make_clue(
            round="Final J!",
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Alice",
                    response_given="What is X?",
                    is_correct=True,
                    buzz_timestamp_ms=1500.0,
                    response_start_timestamp_ms=1750.0,
                    is_lockout_inferred=False,
                )
            ],
        )
        sm.process_clue(clue)
        assert sm.scores["Alice"] == 0

    def test_unknown_round_clue_value_zero(self) -> None:
        sm = TrebekStateMachine(
            initial_scores={"Alice": 0, "Bob": 0, "Carol": 0},
            valid_contestants={"Alice", "Bob", "Carol"},
        )
        clue = _make_clue(
            round="Tiebreaker",
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Alice",
                    response_given="What is X?",
                    is_correct=True,
                    buzz_timestamp_ms=1500.0,
                    response_start_timestamp_ms=1750.0,
                    is_lockout_inferred=False,
                )
            ],
        )
        sm.process_clue(clue)
        assert sm.scores["Alice"] == 0


class TestDailyDoubleWithoutWager:
    """Line 55: DD flagged but wager or wagerer missing → falls back to standard scoring."""

    def test_dd_without_wager_falls_back(self) -> None:
        sm = TrebekStateMachine(
            initial_scores={"Alice": 0, "Bob": 0, "Carol": 0},
            valid_contestants={"Alice", "Bob", "Carol"},
        )
        clue = _make_clue(
            round="J!",
            row=3,
            is_dd=True,
            dd_wager=None,
            wagerer=None,
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Alice",
                    response_given="What is X?",
                    is_correct=True,
                    buzz_timestamp_ms=1500.0,
                    response_start_timestamp_ms=1750.0,
                    is_lockout_inferred=False,
                )
            ],
        )
        sm.process_clue(clue)
        # Falls back to standard scoring: row 3 × $200 = $600
        assert sm.scores["Alice"] == 600

    def test_dd_with_wager_no_wagerer_falls_back(self) -> None:
        sm = TrebekStateMachine(
            initial_scores={"Alice": 0, "Bob": 0, "Carol": 0},
            valid_contestants={"Alice", "Bob", "Carol"},
        )
        clue = _make_clue(
            round="J!",
            row=2,
            is_dd=True,
            dd_wager=500,
            wagerer=None,
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Alice",
                    response_given="What is X?",
                    is_correct=True,
                    buzz_timestamp_ms=1500.0,
                    response_start_timestamp_ms=1750.0,
                    is_lockout_inferred=False,
                )
            ],
        )
        sm.process_clue(clue)
        # Falls back to standard: row 2 × $200 = $400
        assert sm.scores["Alice"] == 400
