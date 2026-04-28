import pytest
from trebek.database import DatabaseWriter
from trebek.database.operations import commit_episode_to_relational_tables
from trebek.schemas import (
    Episode,
    Clue,
    BuzzAttempt,
    Contestant,
    FinalJeopardy,
    FinalJeopardyWager,
    ScoreAdjustment,
)
from trebek.state_machine import TrebekStateMachine


def _make_clue(round="Jeopardy", row=1, col=1, order=1, dd=False, text="C", is_correct=True, attempts=None):
    return Clue(
        round=round,
        category="Cat",
        board_row=row,
        board_col=col,
        selection_order=order,
        is_daily_double=dd,
        requires_visual_context=False,
        host_start_timestamp_ms=0,
        host_finish_timestamp_ms=1000,
        clue_syllable_count=10,
        clue_text=text,
        correct_response="Ans",
        daily_double_wager=500 if dd else None,
        wagerer_name="Alice" if dd else None,
        attempts=attempts
        or [
            BuzzAttempt(
                attempt_order=1,
                speaker="Alice",
                response_given="What is?",
                is_correct=is_correct,
                buzz_timestamp_ms=1500,
                response_start_timestamp_ms=2000,
                is_lockout_inferred=False,
            )
        ],
    )


@pytest.fixture
def sample_episode():
    # 30 J, 30 DJ (1 DD in J, 2 DD in DJ) -> 60 clues total
    clues = []
    for i in range(30):
        clues.append(_make_clue(round="Jeopardy", order=i + 1, dd=(i == 15)))
    for i in range(30):
        clues.append(_make_clue(round="Double Jeopardy", order=31 + i, dd=(i == 15 or i == 20)))

    return Episode(
        episode_date="2024-01-01",
        host_name="Ken",
        is_tournament=False,
        contestants=[
            Contestant(
                name="Alice", podium_position=1, occupational_category="A", is_returning_champion=False, description=""
            ),
            Contestant(
                name="Bob", podium_position=2, occupational_category="B", is_returning_champion=False, description=""
            ),
        ],
        clues=clues,
        final_jeopardy=FinalJeopardy(
            category="Sci",
            clue_text="FJ",
            wagers_and_responses=[
                FinalJeopardyWager(contestant="Alice", wager=1000, response="X", is_correct=True),
                FinalJeopardyWager(contestant="Bob", wager=2000, response="Y", is_correct=False),
            ],
        ),
        score_adjustments=[
            ScoreAdjustment(
                contestant="Alice", points_adjusted=200, reason="fix", effective_after_clue_selection_order=1
            )
        ],
    )


@pytest.mark.asyncio
async def test_commit_episode_to_relational_tables(memory_db_path: str, sample_episode: Episode):
    writer = DatabaseWriter(memory_db_path)
    await writer.start()

    try:
        # Run state machine
        valid_contestants = {c.name for c in sample_episode.contestants}
        state_machine = TrebekStateMachine(valid_contestants=valid_contestants)
        state_machine.load_adjustments(sample_episode.score_adjustments)
        for clue in sample_episode.clues:
            state_machine.process_clue(clue)
        state_machine.process_final_jeopardy(sample_episode.final_jeopardy)

        # Commit
        await commit_episode_to_relational_tables(writer, "ep_1", sample_episode, state_machine)

        # Assertions
        clues_count = (await writer.execute("SELECT COUNT(*) FROM clues"))[0][0]
        assert clues_count == 61  # 60 regular + 1 FJ

        wagers_count = (await writer.execute("SELECT COUNT(*) FROM wagers"))[0][0]
        assert wagers_count == 5  # 3 DDs + 2 FJ wagers

        buzz_count = (await writer.execute("SELECT COUNT(*) FROM buzz_attempts"))[0][0]
        assert buzz_count == 62  # 60 regular + 2 FJ wagers

        # Test true_buzzer_latency_ms
        latency_rows = await writer.execute(
            "SELECT true_buzzer_latency_ms FROM buzz_attempts WHERE clue_id != 'ep_1_fj'"
        )
        for row in latency_rows:
            assert row[0] is not None
            assert row[0] == 500  # buzz(1500) - host_finish(1000)

        # Test final scores
        score_rows = await writer.execute(
            "SELECT contestant_id, final_score FROM episode_performances ORDER BY contestant_id"
        )
        # Ensure we have scores
        assert len(score_rows) == 2
    finally:
        await writer.stop()
