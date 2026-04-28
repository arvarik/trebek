import pytest
import sqlite3
from trebek.database.writer import DatabaseWriter


@pytest.mark.asyncio
async def test_database_writer_integrity_rollback(memory_db_path: str) -> None:
    """Test that DatabaseWriter correctly recovers from an IntegrityError without deadlocking."""
    writer = DatabaseWriter(memory_db_path)
    await writer.start()

    try:
        # Create schema with NOT NULL constraints
        await writer.execute("""
            CREATE TABLE test_strict (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)

        # Insert a valid row
        await writer.execute("INSERT INTO test_strict (name) VALUES (?)", ("valid",))

        # Attempt to violate NOT NULL constraint
        with pytest.raises(sqlite3.IntegrityError):
            await writer.execute("INSERT INTO test_strict (name) VALUES (?)", (None,))

        # Verify the writer is STILL alive and processes the next query
        await writer.execute("INSERT INTO test_strict (name) VALUES (?)", ("recovered",))

        # Assert data is correct (the transaction rolled back the invalid row)
        result = await writer.execute("SELECT name FROM test_strict ORDER BY id")
        assert len(result) == 2
        assert result[0][0] == "valid"
        assert result[1][0] == "recovered"
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_database_writer_transaction_rollback(memory_db_path: str) -> None:
    """Test that a failed query in a transaction rolls back the entire transaction."""
    writer = DatabaseWriter(memory_db_path)
    await writer.start()

    try:
        await writer.execute("""
            CREATE TABLE test_tx (
                id INTEGER PRIMARY KEY,
                val TEXT NOT NULL
            )
        """)

        # Execute a transaction where the second query fails
        queries = [
            ("INSERT INTO test_tx (val) VALUES (?)", ("first_part",)),
            ("INSERT INTO test_tx (val) VALUES (?)", (None,)),  # Fails
            ("INSERT INTO test_tx (val) VALUES (?)", ("third_part",)),
        ]

        with pytest.raises(sqlite3.IntegrityError):
            await writer.execute_transaction(queries)

        # Verify none of the transaction committed
        result = await writer.execute("SELECT COUNT(*) FROM test_tx")
        assert result[0][0] == 0

        # Writer should still be active
        await writer.execute("INSERT INTO test_tx (val) VALUES (?)", ("still_alive",))
        result = await writer.execute("SELECT COUNT(*) FROM test_tx")
        assert result[0][0] == 1
    finally:
        await writer.stop()
