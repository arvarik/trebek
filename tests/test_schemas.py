import pytest
from pydantic import ValidationError
from src.schemas import Clue, Contestant


def test_contestant_podium_constraint() -> None:
    # Valid podium positions: 1, 2, 3
    valid_contestant = Contestant(
        name="Ken",
        podium_position=1,
        occupational_category="Tech",
        is_returning_champion=False,
        description="A guy from Utah",
    )
    assert valid_contestant.name == "Ken"

    with pytest.raises(ValidationError):
        Contestant(
            name="Ken",
            podium_position=4,  # Invalid
            occupational_category="Tech",
            is_returning_champion=False,
            description="A guy from Utah",
        )


def test_clue_daily_double_wager_types() -> None:
    clue_data = {
        "round": "Jeopardy",
        "category": "Math",
        "board_row": 1,
        "board_col": 1,
        "selection_order": 1,
        "is_daily_double": True,
        "daily_double_wager": "True Daily Double",
        "requires_visual_context": False,
        "host_start_timestamp_ms": 0.0,
        "host_finish_timestamp_ms": 1000.0,
        "clue_syllable_count": 10,
        "clue_text": "text",
        "correct_response": "response",
    }

    # String literal 'True Daily Double'
    clue = Clue(**clue_data)
    assert clue.daily_double_wager == "True Daily Double"

    # Integer wager
    clue_data["daily_double_wager"] = 1500
    clue = Clue(**clue_data)
    assert clue.daily_double_wager == 1500
