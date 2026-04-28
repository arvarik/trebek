import pytest
import sqlite3
from trebek.database import DatabaseWriter


@pytest.mark.asyncio
async def test_foreign_key_constraints_enforced(memory_db_path: str) -> None:
    writer = DatabaseWriter(memory_db_path)
    await writer.start()

    try:
        # Attempt to insert a clue without an existing episode should fail due to foreign keys
        with pytest.raises(sqlite3.IntegrityError):
            await writer.execute(
                "INSERT INTO clues (clue_id, episode_id, round) VALUES (?, ?, ?)",
                ("clue_1", "non_existent_ep", "Jeopardy"),
            )
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_check_constraints_enforced(memory_db_path: str) -> None:
    writer = DatabaseWriter(memory_db_path)
    await writer.start()

    try:
        # First create an episode so foreign key is valid
        await writer.execute("INSERT INTO episodes (episode_id) VALUES (?)", ("ep_1",))

        # Attempt to insert an invalid round
        with pytest.raises(sqlite3.IntegrityError):
            await writer.execute(
                "INSERT INTO clues (clue_id, episode_id, round) VALUES (?, ?, ?)", ("clue_2", "ep_1", "InvalidRound")
            )
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_not_null_constraints_enforced(memory_db_path: str) -> None:
    writer = DatabaseWriter(memory_db_path)
    await writer.start()

    try:
        # Status is NOT NULL in pipeline_state
        with pytest.raises(sqlite3.IntegrityError):
            await writer.execute("INSERT INTO pipeline_state (episode_id) VALUES (?)", ("ep_null_test",))
    finally:
        await writer.stop()
