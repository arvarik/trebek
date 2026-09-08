import pytest
import sqlite3
from pathlib import Path


# Centralized schema path — used by all test modules regardless of directory depth.
SCHEMA_PATH = Path(__file__).parent.parent / "trebek" / "schema.sql"


@pytest.fixture
def memory_db_path(tmp_path: Path) -> str:
    db_path = tmp_path / "test.db"

    # Create schema
    conn = sqlite3.connect(db_path)
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()

    return str(db_path)
