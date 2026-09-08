"""
Tests for SQLite FTS5 full-text search, trigger synchronization, and search_clues.
"""

import sqlite3
import pytest
from pathlib import Path

from trebek.database.writer import DatabaseWriter


@pytest.fixture
async def fts_db_writer(tmp_path: Path) -> DatabaseWriter:
    """Creates a DatabaseWriter with schema and seeded clue data."""
    db_path = str(tmp_path / "test_fts.db")
    schema_path = Path(__file__).resolve().parents[2] / "trebek" / "schema.sql"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

        # Insert test episode
        conn.execute(
            "INSERT INTO episodes (episode_id, air_date, host_name, is_tournament) VALUES (?, ?, ?, ?)",
            ("ep_2024_01", "2024-01-15", "Ken Jennings", False),
        )

        # Insert initial clues
        clues = [
            (
                "ep_2024_01_c1",
                "ep_2024_01",
                "J!",
                "WORLD HISTORY",
                1,
                1,
                1,
                "In 1789, the storming of this Paris fortress ignited the French Revolution",
                "Bastille",
                True,
            ),
            (
                "ep_2024_01_c2",
                "ep_2024_01",
                "J!",
                "SCIENCE & NATURE",
                2,
                1,
                2,
                "This elemental gas makes up approximately 78 percent of Earth's atmosphere",
                "Nitrogen",
                True,
            ),
            (
                "ep_2024_01_c3",
                "ep_2024_01",
                "Double J!",
                "LITERATURE",
                1,
                2,
                3,
                "This 1851 Herman Melville novel begins with the famous line 'Call me Ishmael'",
                "Moby-Dick",
                True,
            ),
            (
                "ep_2024_01_c4",
                "ep_2024_01",
                "Final J!",
                "FAMOUS SHIPS",
                1,
                1,
                4,
                "In 1912, this White Star Line ocean liner sank after striking an iceberg in the North Atlantic",
                "Titanic",
                True,
            ),
        ]
        conn.executemany(
            "INSERT INTO clues (clue_id, episode_id, round, category, board_row, board_col, selection_order, "
            "clue_text, correct_response, is_verified) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            clues,
        )
        conn.commit()

    writer = DatabaseWriter(db_path)
    await writer.start()
    yield writer
    await writer.stop()


class TestFTS5Search:
    """Full-text search queries using FTS5."""

    async def test_search_clue_text(self, fts_db_writer: DatabaseWriter) -> None:
        results = await fts_db_writer.search_clues("revolution")
        assert len(results) == 1
        assert results[0]["clue_id"] == "ep_2024_01_c1"
        assert results[0]["correct_response"] == "Bastille"

    async def test_search_category(self, fts_db_writer: DatabaseWriter) -> None:
        results = await fts_db_writer.search_clues("LITERATURE")
        assert len(results) == 1
        assert results[0]["correct_response"] == "Moby-Dick"

    async def test_search_correct_response(self, fts_db_writer: DatabaseWriter) -> None:
        results = await fts_db_writer.search_clues("Titanic")
        assert len(results) == 1
        assert results[0]["round"] == "Final J!"
        assert results[0]["category"] == "FAMOUS SHIPS"

    async def test_search_round_filter(self, fts_db_writer: DatabaseWriter) -> None:
        # Search term present in J! and Double J!
        all_res = await fts_db_writer.search_clues("this")
        assert len(all_res) >= 2

        filtered_res = await fts_db_writer.search_clues("this", round_filter="Double J!")
        assert len(filtered_res) == 1
        assert filtered_res[0]["round"] == "Double J!"

    async def test_search_empty_query(self, fts_db_writer: DatabaseWriter) -> None:
        results = await fts_db_writer.search_clues("   ")
        assert results == []

    async def test_search_no_matches(self, fts_db_writer: DatabaseWriter) -> None:
        results = await fts_db_writer.search_clues("xyznonexistentterm123")
        assert results == []


class TestFTS5Triggers:
    """Automatic trigger synchronization between clues and clues_fts."""

    async def test_insert_or_replace_no_duplicates(self, fts_db_writer: DatabaseWriter) -> None:
        # Initial check
        res = await fts_db_writer.search_clues("atmosphere")
        assert len(res) == 1

        # Replace existing clue with updated text
        await fts_db_writer.execute(
            "INSERT OR REPLACE INTO clues (clue_id, episode_id, round, category, board_row, board_col, "
            "selection_order, clue_text, correct_response, is_verified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ep_2024_01_c2",
                "ep_2024_01",
                "J!",
                "PLANETARY SCIENCE",
                2,
                1,
                2,
                "This elemental gas makes up 78% of the atmosphere on Earth",
                "Nitrogen Gas",
                True,
            ),
        )

        # FTS5 index should reflect the new category and response without duplicate rows
        res_new = await fts_db_writer.search_clues("PLANETARY")
        assert len(res_new) == 1
        assert res_new[0]["category"] == "PLANETARY SCIENCE"

        # Old category should be gone
        res_old = await fts_db_writer.search_clues("SCIENCE & NATURE")
        assert len(res_old) == 0

        # Searching common word should still return exactly 1 match for this clue
        res_common = await fts_db_writer.search_clues("atmosphere")
        assert len(res_common) == 1

    async def test_delete_trigger(self, fts_db_writer: DatabaseWriter) -> None:
        # Delete Moby-Dick clue
        await fts_db_writer.execute("DELETE FROM clues WHERE clue_id = ?", ("ep_2024_01_c3",))

        # Should no longer be found in FTS
        res = await fts_db_writer.search_clues("Melville")
        assert len(res) == 0

    async def test_update_trigger(self, fts_db_writer: DatabaseWriter) -> None:
        await fts_db_writer.execute(
            "UPDATE clues SET clue_text = ? WHERE clue_id = ?",
            ("Updated novel text by Herman Melville", "ep_2024_01_c3"),
        )
        res = await fts_db_writer.search_clues("Updated")
        assert len(res) == 1
        assert res[0]["clue_id"] == "ep_2024_01_c3"


class TestFTS5MigrationBackfill:
    """Automatic migration and backfill for databases created without clues_fts."""

    async def test_migration_backfills_unindexed_database(self, tmp_path: Path) -> None:
        old_db_path = str(tmp_path / "old_version.db")
        # Create legacy table without clues_fts
        with sqlite3.connect(old_db_path) as conn:
            conn.execute(
                "CREATE TABLE clues ("
                "clue_id TEXT PRIMARY KEY, episode_id TEXT, round TEXT, category TEXT, "
                "board_row INTEGER, board_col INTEGER, selection_order INTEGER, "
                "clue_text TEXT, correct_response TEXT, is_verified BOOLEAN, "
                "is_daily_double BOOLEAN, is_triple_stumper BOOLEAN"
                ")"
            )
            conn.execute(
                "INSERT INTO clues VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "c_legacy",
                    "ep_legacy",
                    "J!",
                    "ASTRONOMY",
                    1,
                    1,
                    1,
                    "This dwarf planet was visited by New Horizons in 2015",
                    "Pluto",
                    True,
                    False,
                    False,
                ),
            )
            conn.commit()

        # Starting DatabaseWriter on this database should auto-detect and migrate
        writer = DatabaseWriter(old_db_path)
        await writer.start()
        try:
            results = await writer.search_clues("Pluto")
            assert len(results) == 1
            assert results[0]["clue_id"] == "c_legacy"
            assert results[0]["category"] == "ASTRONOMY"
        finally:
            await writer.stop()
