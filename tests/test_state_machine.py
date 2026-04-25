from trebek.state_machine import TrebekStateMachine
from trebek.schemas import Clue, BuzzAttempt, ScoreAdjustment


def test_process_clue_basic() -> None:
    sm = TrebekStateMachine({"PlayerA": 0, "PlayerB": 0})

    clue = Clue(
        round="Jeopardy",
        category="History",
        board_row=1,
        board_col=1,
        selection_order=1,
        is_daily_double=False,
        requires_visual_context=False,
        host_start_timestamp_ms=0,
        host_finish_timestamp_ms=1000,
        clue_syllable_count=10,
        clue_text="A famous war.",
        correct_response="WWII",
        attempts=[
            BuzzAttempt(
                attempt_order=1,
                speaker="PlayerA",
                response_given="What is WWII?",
                is_correct=True,
                buzz_timestamp_ms=1500,
                response_start_timestamp_ms=1600,
                is_lockout_inferred=False,
            )
        ],
    )

    sm.process_clue(clue)
    assert sm.scores["PlayerA"] == 200
    assert sm.scores["PlayerB"] == 0
    assert sm.current_board_control_contestant == "PlayerA"


def test_score_adjustment_chronological() -> None:
    sm = TrebekStateMachine({"PlayerA": 200})

    adjustment = ScoreAdjustment(
        contestant="PlayerA",
        points_adjusted=400,
        reason="Judge reversed ruling",
        effective_after_clue_selection_order=1,
    )
    sm.load_adjustments([adjustment])

    # Process a clue with selection_order 1
    clue = Clue(
        round="Jeopardy",
        category="Math",
        board_row=2,
        board_col=1,
        selection_order=1,
        is_daily_double=False,
        requires_visual_context=False,
        host_start_timestamp_ms=0,
        host_finish_timestamp_ms=1000,
        clue_syllable_count=10,
        clue_text="2+2",
        correct_response="4",
        attempts=[
            BuzzAttempt(
                attempt_order=1,
                speaker="PlayerA",
                response_given="4",
                is_correct=False,
                buzz_timestamp_ms=1500,
                response_start_timestamp_ms=1600,
                is_lockout_inferred=False,
            )
        ],
    )
    sm.process_clue(clue)

    # PlayerA missed $400, score goes to -200
    # Then adjustment applies at index 1: +400
    assert sm.scores["PlayerA"] == 200
