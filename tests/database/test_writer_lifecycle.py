"""
Tests for DatabaseWriter lifecycle — start(), stop(), WAL pragma,
and the background vacuum task.

Targets uncovered lines: 50-51, 56-57, 71, 102-103, 107-113,
127-128, 141-143, 156-157, 168-170, 174-175.
"""

import asyncio
import sqlite3
import pytest
from pathlib import Path

from trebek.database.writer import DatabaseWriter


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Creates a fresh database with schema."""
    path = str(tmp_path / "test.db")
    schema_path = Path(__file__).resolve().parents[2] / "trebek" / "schema.sql"
    with sqlite3.connect(path) as conn:
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
    return path


class TestWriterLifecycle:
    """Start/stop lifecycle tests."""

    async def test_start_creates_connection(self, db_path: str) -> None:
        writer = DatabaseWriter(db_path)
        assert writer.conn is None
        await writer.start()
        assert writer.conn is not None
        await writer.stop()

    async def test_stop_closes_connection(self, db_path: str) -> None:
        writer = DatabaseWriter(db_path)
        await writer.start()
        await writer.stop()
        # After stop, conn should be closed (accessing it would fail)
        # The writer stores it but it's closed
        assert writer.conn is not None  # still referenced
        with pytest.raises(Exception):
            writer.conn.execute("SELECT 1")

    async def test_start_enables_wal_mode(self, db_path: str) -> None:
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            # Check WAL mode was set
            with sqlite3.connect(db_path) as conn:
                result = conn.execute("PRAGMA journal_mode;").fetchone()
                assert result[0] == "wal"
        finally:
            await writer.stop()

    async def test_start_enables_foreign_keys(self, db_path: str) -> None:
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            # FK pragma is per-connection — must check via the writer's own connection
            result = await writer.execute("PRAGMA foreign_keys;")
            assert result == [(1,)]
        finally:
            await writer.stop()

    async def test_stop_cancels_vacuum_task(self, db_path: str) -> None:
        writer = DatabaseWriter(db_path)
        await writer.start()
        assert writer.vacuum_task is not None
        assert not writer.vacuum_task.done()
        await writer.stop()
        assert writer.vacuum_task.done()

    async def test_stop_cancels_queue_task(self, db_path: str) -> None:
        writer = DatabaseWriter(db_path)
        await writer.start()
        assert writer.task is not None
        assert not writer.task.done()
        await writer.stop()
        assert writer.task.done()


class TestQueueProcessing:
    """Queue dispatch and message handling tests."""

    async def test_execute_with_no_params(self, db_path: str) -> None:
        """execute() with no params should work (for queries like SELECT 1)."""
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            result = await writer.execute("SELECT COUNT(*) FROM pipeline_state")
            assert result == [(0,)]
        finally:
            await writer.stop()

    async def test_execute_returns_rows_for_select(self, db_path: str) -> None:
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            await writer.execute(
                "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                ("ep1", "PENDING"),
            )
            rows = await writer.execute("SELECT episode_id, status FROM pipeline_state")
            assert len(rows) == 1
            assert rows[0] == ("ep1", "PENDING")
        finally:
            await writer.stop()

    async def test_executemany_batch_inserts(self, db_path: str) -> None:
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            params = [(f"ep{i}", "PENDING") for i in range(50)]
            await writer.executemany(
                "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                params,
            )
            rows = await writer.execute("SELECT COUNT(*) FROM pipeline_state")
            assert rows == [(50,)]
        finally:
            await writer.stop()


class TestTransactionDispatch:
    """Transaction handling through the queue dispatch."""

    async def test_transaction_with_list_params_uses_executemany(self, db_path: str) -> None:
        """When transaction params is a list, it should dispatch to executemany."""
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            queries = [
                (
                    "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                    [("ep_a", "PENDING"), ("ep_b", "PENDING"), ("ep_c", "PENDING")],
                ),
            ]
            await writer.execute_transaction(queries)
            rows = await writer.execute("SELECT COUNT(*) FROM pipeline_state")
            assert rows == [(3,)]
        finally:
            await writer.stop()

    async def test_transaction_rollback_preserves_prior_data(self, db_path: str) -> None:
        """A failed transaction should not affect data inserted before the transaction."""
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            # Insert a row outside the transaction
            await writer.execute(
                "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                ("existing", "COMPLETED"),
            )

            # Now try a transaction that fails
            with pytest.raises(Exception):
                await writer.execute_transaction(
                    [
                        ("INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)", ("new1", "PENDING")),
                        # Duplicate PK — will fail
                        ("INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)", ("new1", "PENDING")),
                    ]
                )

            # The pre-existing row should still be there
            rows = await writer.execute("SELECT COUNT(*) FROM pipeline_state")
            assert rows == [(1,)]
        finally:
            await writer.stop()


class TestConcurrentQueueAccess:
    """Verify the actor serialization under concurrent pressure."""

    async def test_interleaved_reads_and_writes(self, db_path: str) -> None:
        """Interleaved read/write operations should all succeed."""
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            tasks = []
            for i in range(30):
                tasks.append(
                    writer.execute(
                        "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                        (f"ep_{i}", "PENDING"),
                    )
                )
                tasks.append(writer.execute("SELECT COUNT(*) FROM pipeline_state"))

            await asyncio.gather(*tasks)
            # Last SELECT should return at least some rows
            final_count = await writer.execute("SELECT COUNT(*) FROM pipeline_state")
            assert final_count == [(30,)]
        finally:
            await writer.stop()

    async def test_rapid_fire_executemany(self, db_path: str) -> None:
        """Multiple executemany calls in rapid succession should not deadlock."""
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            batch_tasks = [
                writer.executemany(
                    "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
                    [(f"batch{b}_ep{i}", "PENDING") for i in range(10)],
                )
                for b in range(5)
            ]
            await asyncio.gather(*batch_tasks)
            rows = await writer.execute("SELECT COUNT(*) FROM pipeline_state")
            assert rows == [(50,)]
        finally:
            await writer.stop()
