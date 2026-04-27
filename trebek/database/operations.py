"""
Relational data commit operations for episode data.

Transforms Pydantic episode models into normalized relational rows
and commits them to the analytical database tables (episodes, contestants,
clues, buzz_attempts, final_jeopardy, score_adjustments) within a
single atomic transaction.
"""

import structlog
from typing import TYPE_CHECKING, Any, Tuple

from trebek.database.writer import DatabaseWriter

if TYPE_CHECKING:
    from trebek.schemas import Episode
    from trebek.state_machine import TrebekStateMachine

logger = structlog.get_logger()


async def commit_episode_to_relational_tables(
    db_writer: DatabaseWriter,
    episode_id: str,
    episode_data: "Episode",
    state_machine: "TrebekStateMachine",
) -> None:
    """
    Commits verified episode data into the normalized relational tables.
    This is the Stage 8 relational commit — writing to episodes, contestants,
    clues, buzz_attempts, wagers, score_adjustments, and episode_performances.
    """

    # 1. Insert episode metadata
    payload: list[Tuple[str, Any]] = []

    payload.append(
        (
            "INSERT OR REPLACE INTO episodes (episode_id, air_date, host_name, is_tournament) VALUES (?, ?, ?, ?)",
            (episode_id, episode_data.episode_date, episode_data.host_name, episode_data.is_tournament),
        )
    )

    # 2. Insert contestants
    for contestant in episode_data.contestants:
        contestant_id = f"{episode_id}_{contestant.name.replace(' ', '_').lower()}"
        payload.append(
            (
                "INSERT OR REPLACE INTO contestants "
                "(contestant_id, name, occupational_category, is_returning_champion) VALUES (?, ?, ?, ?)",
                (contestant_id, contestant.name, contestant.occupational_category, contestant.is_returning_champion),
            )
        )

        # 3. Insert episode_performances with final scores from state machine
        final_score = state_machine.scores.get(contestant.name, 0)
        payload.append(
            (
                "INSERT OR REPLACE INTO episode_performances "
                "(episode_id, contestant_id, podium_position, final_score) VALUES (?, ?, ?, ?)",
                (episode_id, contestant_id, contestant.podium_position, final_score),
            )
        )

    # 4. Insert clues and buzz_attempts
    for clue in episode_data.clues:
        clue_id = f"{episode_id}_c{clue.selection_order}"
        is_triple_stumper = len(clue.attempts) == 0 or all(not a.is_correct for a in clue.attempts)

        dd_wager_str = str(clue.daily_double_wager) if clue.daily_double_wager is not None else None

        payload.append(
            (
                "INSERT OR REPLACE INTO clues "
                "(clue_id, episode_id, round, category, board_row, board_col, selection_order, "
                "clue_text, correct_response, is_daily_double, is_triple_stumper, "
                "daily_double_wager, wagerer_name, requires_visual_context, "
                "host_start_timestamp_ms, host_finish_timestamp_ms, clue_syllable_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    clue_id,
                    episode_id,
                    clue.round,
                    clue.category,
                    clue.board_row,
                    clue.board_col,
                    clue.selection_order,
                    clue.clue_text,
                    clue.correct_response,
                    clue.is_daily_double,
                    is_triple_stumper,
                    dd_wager_str,
                    clue.wagerer_name,
                    clue.requires_visual_context,
                    clue.host_start_timestamp_ms,
                    clue.host_finish_timestamp_ms,
                    clue.clue_syllable_count,
                ),
            )
        )

        # Insert buzz_attempts for this clue
        for attempt in clue.attempts:
            attempt_id = f"{clue_id}_a{attempt.attempt_order}"
            contestant_id = f"{episode_id}_{attempt.speaker.replace(' ', '_').lower()}"
            payload.append(
                (
                    "INSERT OR REPLACE INTO buzz_attempts "
                    "(attempt_id, clue_id, contestant_id, attempt_order, buzz_timestamp_ms, "
                    "response_given, is_correct, response_start_timestamp_ms, is_lockout_inferred) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        attempt_id,
                        clue_id,
                        contestant_id,
                        attempt.attempt_order,
                        attempt.buzz_timestamp_ms,
                        attempt.response_given,
                        attempt.is_correct,
                        attempt.response_start_timestamp_ms,
                        attempt.is_lockout_inferred,
                    ),
                )
            )

    # 5. Insert score_adjustments
    for adj in episode_data.score_adjustments:
        adjustment_id = f"{episode_id}_adj_{adj.effective_after_clue_selection_order}_{adj.contestant.replace(' ', '_').lower()}"
        contestant_id = f"{episode_id}_{adj.contestant.replace(' ', '_').lower()}"
        payload.append(
            (
                "INSERT OR REPLACE INTO score_adjustments "
                "(adjustment_id, episode_id, contestant_id, points_adjusted, reason, "
                "effective_after_clue_selection_order) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    adjustment_id,
                    episode_id,
                    contestant_id,
                    adj.points_adjusted,
                    adj.reason,
                    adj.effective_after_clue_selection_order,
                ),
            )
        )

    await db_writer.execute_transaction(payload)

    logger.info(
        "Relational commit complete",
        episode_id=episode_id,
        clues=len(episode_data.clues),
        contestants=len(episode_data.contestants),
    )
