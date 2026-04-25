import pytest
import sqlite3
from pathlib import Path


@pytest.fixture
def memory_db_path(tmp_path: Path) -> str:
    db_path = tmp_path / "test.db"

    # Create schema
    schema_path = Path(__file__).parent.parent / "src" / "schema.sql"
    with sqlite3.connect(db_path) as conn:
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

    return str(db_path)
