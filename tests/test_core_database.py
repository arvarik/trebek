import pytest
from trebek.core_database import DatabaseWriter


@pytest.mark.asyncio
async def test_database_writer_execution(memory_db_path: str) -> None:
    writer = DatabaseWriter(memory_db_path)
    await writer.start()

    try:
        # Test insert
        await writer.execute("INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)", ("ep_test_1", "PENDING"))

        # Test select
        result = await writer.execute("SELECT status FROM pipeline_state WHERE episode_id = ?", ("ep_test_1",))
        assert result[0][0] == "PENDING"
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_poll_for_work(memory_db_path: str) -> None:
    writer = DatabaseWriter(memory_db_path)
    await writer.start()

    try:
        await writer.execute("INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)", ("ep_test_1", "PENDING"))

        episode_id = await writer.poll_for_work("PENDING", "PROCESSING")
        assert episode_id == "ep_test_1"

        # Verify status changed
        result = await writer.execute("SELECT status FROM pipeline_state WHERE episode_id = ?", ("ep_test_1",))
        assert result[0][0] == "PROCESSING"
    finally:
        await writer.stop()
