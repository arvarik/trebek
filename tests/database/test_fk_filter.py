"""
Tests for database operations FK pre-commit filter — exercises the inline
validation that drops buzz_attempts, score_adjustments, and wagers with
invalid contestant_id references before committing to the database.

Uses commit_episode_to_relational_tables with crafted Episode data
containing invalid contestant references.
"""

import sqlite3
import pytest
from pathlib import Path

from trebek.database.writer import DatabaseWriter
from trebek.database.operations import commit_episode_to_relational_tables
from trebek.schemas import (
    Episode,
    Clue,
    BuzzAttempt,
    Contestant,
    FinalJeopardy,
    ScoreAdjustment,
)
from trebek.state_machine import TrebekStateMachine

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "trebek" / "schema.sql"


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    with sqlite3.connect(path) as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
    return path


def _make_episode_with_invalid_fk() -> Episode:
    """Create an Episode where a buzz attempt references a non-existent contestant."""
    return Episode(
        episode_date="2025-01-15",
        host_name="Ken Jennings",
        is_tournament=False,
        contestants=[
            Contestant(
                name="Alice Smith",
                podium_position=1,
                is_returning_champion=False,
                occupational_category="STEM",
                description="from NY",
            ),
            Contestant(
                name="Bob Jones",
                podium_position=2,
                is_returning_champion=False,
                occupational_category="Law",
                description="from LA",
            ),
            Contestant(
                name="Carol White",
                podium_position=3,
                is_returning_champion=False,
                occupational_category="Arts",
                description="from TX",
            ),
        ],
        clues=[
            Clue(
                round="Jeopardy",
                category="History",
                board_row=1,
                board_col=1,
                selection_order=1,
                is_daily_double=False,
                requires_visual_context=False,
                host_start_timestamp_ms=0.0,
                host_finish_timestamp_ms=5000.0,
                clue_syllable_count=10,
                clue_text="This city was the capital of the Roman Empire",
                correct_response="What is Rome?",
                attempts=[
                    BuzzAttempt(
                        attempt_order=1,
                        speaker="Alice Smith",  # Valid
                        response_given="What is Rome?",
                        is_correct=True,
                        buzz_timestamp_ms=5500.0,
                        response_start_timestamp_ms=5800.0,
                        is_lockout_inferred=False,
                    ),
                    BuzzAttempt(
                        attempt_order=2,
                        speaker="GHOST_PLAYER",  # Invalid - not a contestant
                        response_given="What is Athens?",
                        is_correct=False,
                        buzz_timestamp_ms=5200.0,
                        response_start_timestamp_ms=5400.0,
                        is_lockout_inferred=False,
                    ),
                ],
            ),
        ],
        final_jeopardy=FinalJeopardy(
            category="Science",
            clue_text="Final clue",
            wagers_and_responses=[],
        ),
        score_adjustments=[
            ScoreAdjustment(
                contestant="GHOST_PLAYER",  # Invalid - not a contestant
                points_adjusted=-400,
                reason="Judge correction",
                effective_after_clue_selection_order=1,
            ),
        ],
    )


class TestFkFilterDropsInvalidContestants:
    """Tests that commit_episode_to_relational_tables filters out invalid contestant references."""

    async def test_commit_succeeds_despite_invalid_contestants(self, db_path: str) -> None:
        """The commit should succeed, silently dropping invalid FK references."""
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            episode = _make_episode_with_invalid_fk()
            sm = TrebekStateMachine(
                initial_scores={c.name: 0 for c in episode.contestants},
                valid_contestants={c.name for c in episode.contestants},
            )
            await commit_episode_to_relational_tables(writer, "test_ep_001", episode, sm)

            # Verify the episode was committed
            rows = await writer.execute("SELECT COUNT(*) FROM episodes")
            assert rows == [(1,)]

            # Valid contestant should be inserted
            contestants = await writer.execute("SELECT name FROM contestants")
            names = {r[0] for r in contestants}
            assert "Alice Smith" in names
            assert "GHOST_PLAYER" not in names

            # Valid buzz attempt should be inserted, invalid one dropped
            buzz_count = await writer.execute("SELECT COUNT(*) FROM buzz_attempts")
            assert buzz_count[0][0] == 1  # Only Alice's attempt survives

            # Invalid score adjustment should be dropped
            adj_count = await writer.execute("SELECT COUNT(*) FROM score_adjustments")
            assert adj_count[0][0] == 0  # GHOST_PLAYER's adjustment dropped
        finally:
            await writer.stop()

    async def test_all_valid_contestants_pass_through(self, db_path: str) -> None:
        """When all contestants are valid, nothing should be filtered."""
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            episode = Episode(
                episode_date="2025-01-15",
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
                clues=[
                    Clue(
                        round="Jeopardy",
                        category="Test",
                        board_row=1,
                        board_col=1,
                        selection_order=1,
                        is_daily_double=False,
                        requires_visual_context=False,
                        host_start_timestamp_ms=0.0,
                        host_finish_timestamp_ms=1000.0,
                        clue_syllable_count=5,
                        clue_text="Clue",
                        correct_response="Answer",
                        attempts=[
                            BuzzAttempt(
                                attempt_order=1,
                                speaker="Alice",
                                response_given="What?",
                                is_correct=True,
                                buzz_timestamp_ms=1500.0,
                                response_start_timestamp_ms=1750.0,
                                is_lockout_inferred=False,
                            ),
                        ],
                    ),
                ],
                final_jeopardy=FinalJeopardy(
                    category="Final",
                    clue_text="Final clue",
                    wagers_and_responses=[],
                ),
                score_adjustments=[],
            )
            sm = TrebekStateMachine(
                initial_scores={c.name: 0 for c in episode.contestants},
                valid_contestants={c.name for c in episode.contestants},
            )
            await commit_episode_to_relational_tables(writer, "test_ep_002", episode, sm)

            buzz_count = await writer.execute("SELECT COUNT(*) FROM buzz_attempts")
            assert buzz_count[0][0] == 1  # Alice's attempt passes
        finally:
            await writer.stop()
