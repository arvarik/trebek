"""
Tests for the deterministic Jeopardy state machine.

Covers: basic scoring, rebounds, triple stumpers, Daily Doubles (including
True Daily Double with negative/zero scores), Double Jeopardy value scaling,
and chronologically anchored score adjustments.
"""

from trebek.state_machine import TrebekStateMachine
from trebek.schemas import Clue, BuzzAttempt, ScoreAdjustment
from typing import Literal, Union


def _make_attempt(speaker: str, correct: bool, order: int = 1) -> BuzzAttempt:
    return BuzzAttempt(
        attempt_order=order,
        speaker=speaker,
        response_given="What?" if not correct else "What is X?",
        is_correct=correct,
        buzz_timestamp_ms=1500,
        response_start_timestamp_ms=1600,
        is_lockout_inferred=False,
    )


def _make_clue(
    round: Literal["Jeopardy", "Double Jeopardy", "Final Jeopardy", "Tiebreaker"] = "Jeopardy",
    row: int = 1,
    col: int = 1,
    order: int = 1,
    dd: bool = False,
    wager: Union[int, Literal["True Daily Double"], None] = None,
    wagerer: str | None = None,
    attempts: list[BuzzAttempt] | None = None,
) -> Clue:
    return Clue(
        round=round,
        category="Test",
        board_row=row,
        board_col=col,
        selection_order=order,
        is_daily_double=dd,
        requires_visual_context=False,
        host_start_timestamp_ms=0,
        host_finish_timestamp_ms=1000,
        clue_syllable_count=10,
        clue_text="Test clue",
        correct_response="Test answer",
        daily_double_wager=wager,
        wagerer_name=wagerer,
        attempts=attempts or [],
    )


class TestBasicScoring:
    def test_correct_answer_adds_value(self) -> None:
        sm = TrebekStateMachine({"A": 0, "B": 0})
        clue = _make_clue(row=1, attempts=[_make_attempt("A", True)])
        sm.process_clue(clue)
        assert sm.scores["A"] == 200
        assert sm.scores["B"] == 0

    def test_incorrect_answer_deducts_value(self) -> None:
        sm = TrebekStateMachine({"A": 0, "B": 0})
        clue = _make_clue(row=1, attempts=[_make_attempt("A", False)])
        sm.process_clue(clue)
        assert sm.scores["A"] == -200

    def test_board_control_shifts_on_correct(self) -> None:
        sm = TrebekStateMachine({"A": 0, "B": 0})
        sm.current_board_control_contestant = "A"
        clue = _make_clue(row=1, attempts=[_make_attempt("B", True)])
        sm.process_clue(clue)
        assert sm.current_board_control_contestant == "B"


class TestRebounds:
    def test_rebound_scoring(self) -> None:
        """First player wrong (-$200), second player correct (+$200)."""
        sm = TrebekStateMachine({"A": 0, "B": 0})
        clue = _make_clue(
            row=1,
            attempts=[
                _make_attempt("A", False, order=1),
                _make_attempt("B", True, order=2),
            ],
        )
        sm.process_clue(clue)
        assert sm.scores["A"] == -200
        assert sm.scores["B"] == 200
        assert sm.current_board_control_contestant == "B"

    def test_triple_rebound(self) -> None:
        """All three players wrong — all deducted."""
        sm = TrebekStateMachine({"A": 0, "B": 0, "C": 0})
        clue = _make_clue(
            row=2,  # $400
            attempts=[
                _make_attempt("A", False, order=1),
                _make_attempt("B", False, order=2),
                _make_attempt("C", False, order=3),
            ],
        )
        sm.process_clue(clue)
        assert sm.scores["A"] == -400
        assert sm.scores["B"] == -400
        assert sm.scores["C"] == -400


class TestTripleStumper:
    def test_no_attempts_no_score_change(self) -> None:
        """Triple Stumper: no buzz attempts, scores unchanged."""
        sm = TrebekStateMachine({"A": 200, "B": 0})
        sm.current_board_control_contestant = "A"
        clue = _make_clue(row=1, attempts=[])
        sm.process_clue(clue)
        assert sm.scores["A"] == 200
        assert sm.scores["B"] == 0
        # Board control stays with the current holder
        assert sm.current_board_control_contestant == "A"


class TestDailyDoubles:
    def test_daily_double_correct(self) -> None:
        sm = TrebekStateMachine({"A": 1000})
        clue = _make_clue(
            dd=True,
            wager=500,
            wagerer="A",
            attempts=[_make_attempt("A", True)],
        )
        sm.process_clue(clue)
        assert sm.scores["A"] == 1500

    def test_daily_double_incorrect(self) -> None:
        sm = TrebekStateMachine({"A": 1000})
        clue = _make_clue(
            dd=True,
            wager=500,
            wagerer="A",
            attempts=[_make_attempt("A", False)],
        )
        sm.process_clue(clue)
        assert sm.scores["A"] == 500

    def test_true_daily_double_positive_score(self) -> None:
        """True Daily Double: wager = max(current_score, max_board_value)."""
        sm = TrebekStateMachine({"A": 3000})
        clue = _make_clue(
            round="Jeopardy",
            dd=True,
            wager="True Daily Double",
            wagerer="A",
            attempts=[_make_attempt("A", True)],
        )
        sm.process_clue(clue)
        # max(3000, 1000) = 3000 → score = 3000 + 3000 = 6000
        assert sm.scores["A"] == 6000

    def test_true_daily_double_zero_score(self) -> None:
        """True DD with $0: wager should be max_board_value (not $0)."""
        sm = TrebekStateMachine({"A": 0})
        clue = _make_clue(
            round="Jeopardy",
            dd=True,
            wager="True Daily Double",
            wagerer="A",
            attempts=[_make_attempt("A", True)],
        )
        sm.process_clue(clue)
        # max(0, 1000) = 1000 → score = 0 + 1000 = 1000
        assert sm.scores["A"] == 1000

    def test_true_daily_double_negative_score(self) -> None:
        """True DD with negative score: wager should be max_board_value."""
        sm = TrebekStateMachine({"A": -200})
        clue = _make_clue(
            round="Jeopardy",
            dd=True,
            wager="True Daily Double",
            wagerer="A",
            attempts=[_make_attempt("A", True)],
        )
        sm.process_clue(clue)
        # max(-200, 1000) = 1000 → score = -200 + 1000 = 800
        assert sm.scores["A"] == 800

    def test_true_daily_double_dj_max_value(self) -> None:
        """True DD in Double Jeopardy uses $2000 as max board value."""
        sm = TrebekStateMachine({"A": 500})
        clue = _make_clue(
            round="Double Jeopardy",
            dd=True,
            wager="True Daily Double",
            wagerer="A",
            attempts=[_make_attempt("A", True)],
        )
        sm.process_clue(clue)
        # max(500, 2000) = 2000 → score = 500 + 2000 = 2500
        assert sm.scores["A"] == 2500


class TestDoubleJeopardyValues:
    def test_dj_row_values(self) -> None:
        """Double Jeopardy values are row * $400 (not * $200)."""
        sm = TrebekStateMachine({"A": 0})
        for row in range(1, 6):
            clue = _make_clue(
                round="Double Jeopardy",
                row=row,
                order=row,
                attempts=[_make_attempt("A", True)],
            )
            sm.process_clue(clue)
        # Sum: 400 + 800 + 1200 + 1600 + 2000 = 6000
        assert sm.scores["A"] == 6000


class TestScoreAdjustments:
    def test_adjustment_applied_at_correct_index(self) -> None:
        sm = TrebekStateMachine({"A": 200})
        adj = ScoreAdjustment(
            contestant="A",
            points_adjusted=400,
            reason="Judge reversed ruling",
            effective_after_clue_selection_order=1,
        )
        sm.load_adjustments([adj])
        clue = _make_clue(
            row=2,
            order=1,
            attempts=[_make_attempt("A", False)],
        )
        sm.process_clue(clue)
        # A misses $400: 200 - 400 = -200, then adj: -200 + 400 = 200
        assert sm.scores["A"] == 200

    def test_multiple_adjustments_same_index(self) -> None:
        """Two adjustments at the same clue index should both apply."""
        sm = TrebekStateMachine({"A": 0, "B": 0})
        sm.load_adjustments(
            [
                ScoreAdjustment(
                    contestant="A", points_adjusted=200, reason="R1", effective_after_clue_selection_order=1
                ),
                ScoreAdjustment(
                    contestant="B", points_adjusted=-100, reason="R2", effective_after_clue_selection_order=1
                ),
            ]
        )
        clue = _make_clue(order=1, attempts=[])
        sm.process_clue(clue)
        assert sm.scores["A"] == 200
        assert sm.scores["B"] == -100

    def test_adjustment_not_applied_early(self) -> None:
        """Adjustment at index 5 should NOT apply when processing index 1."""
        sm = TrebekStateMachine({"A": 0})
        sm.load_adjustments(
            [
                ScoreAdjustment(
                    contestant="A", points_adjusted=999, reason="Late", effective_after_clue_selection_order=5
                ),
            ]
        )
        clue = _make_clue(order=1, attempts=[])
        sm.process_clue(clue)
        assert sm.scores["A"] == 0
        # Adjustment should still be pending
        assert len(sm.pending_adjustments) == 1


class TestStateMachineContestantValidation:
    """Tests for the valid_contestants gating in the state machine."""

    def test_unknown_speaker_skipped(self) -> None:
        """Unknown speakers should not create score entries."""
        sm = TrebekStateMachine(valid_contestants={"Alice", "Bob"})
        clue = _make_clue(row=1, attempts=[_make_attempt("UNKNOWN_PERSON", True)])
        sm.process_clue(clue)
        assert "UNKNOWN_PERSON" not in sm.scores
        assert sm.unknown_speaker_warnings == 1

    def test_known_speaker_scored(self) -> None:
        """Known speakers should be scored normally."""
        sm = TrebekStateMachine(valid_contestants={"Alice", "Bob"})
        clue = _make_clue(row=1, attempts=[_make_attempt("Alice", True)])
        sm.process_clue(clue)
        assert sm.scores["Alice"] == 200
        assert sm.unknown_speaker_warnings == 0

    def test_no_validation_when_none(self) -> None:
        """When valid_contestants is None, all speakers are accepted (backward compat)."""
        sm = TrebekStateMachine()
        clue = _make_clue(row=1, attempts=[_make_attempt("AnyName", True)])
        sm.process_clue(clue)
        assert sm.scores["AnyName"] == 200
