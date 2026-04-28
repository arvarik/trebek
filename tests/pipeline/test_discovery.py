"""
Tests for pipeline file discovery — filesystem scanning, stage filtering,
and database status enrichment.
"""

import os
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch

from trebek.pipeline.discovery import discover_video_files, _STAGE_COMPLETED_STATUSES


@pytest.fixture
def video_dir(tmp_path: Path) -> Path:
    """Creates a temp directory with various video files."""
    # Flat video files
    (tmp_path / "episode1.mp4").write_text("fake_video_1")
    (tmp_path / "episode2.mkv").write_text("fake_video_2")
    (tmp_path / "episode3.avi").write_text("fake_video_3")

    # Non-video files (should be ignored)
    (tmp_path / "notes.txt").write_text("not a video")
    (tmp_path / "thumbnail.jpg").write_bytes(b"\x00")

    # Nested directory
    subdir = tmp_path / "Season 41"
    subdir.mkdir()
    (subdir / "S41E01.mp4").write_text("fake_video_4")
    (subdir / "S41E02.mp4").write_text("fake_video_5")

    return tmp_path


@pytest.fixture
def db_with_statuses(tmp_path: Path) -> str:
    """Creates a database with pipeline_state entries at various stages."""
    db_path = str(tmp_path / "trebek.db")
    schema_path = Path(__file__).resolve().parents[2] / "trebek" / "schema.sql"
    with sqlite3.connect(db_path) as conn:
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.execute(
            "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
            ("episode1", "PENDING"),
        )
        conn.execute(
            "INSERT INTO pipeline_state (episode_id, status, retry_count, last_error) VALUES (?, ?, ?, ?)",
            ("episode2", "FAILED", 3, "some error"),
        )
        conn.execute(
            "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
            ("episode3", "COMPLETED"),
        )
        conn.execute(
            "INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)",
            ("Season_41_S41E01", "TRANSCRIPT_READY"),
        )
        conn.commit()
    return db_path


class TestDiscoverVideoFiles:
    """Basic discovery tests."""

    def test_discovers_all_video_extensions(self, video_dir: Path) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = str(video_dir / "nonexistent.db")
            files = discover_video_files(str(video_dir))

        extensions = {f["format"] for f in files}
        assert ".mp4" in extensions
        assert ".mkv" in extensions
        assert ".avi" in extensions

    def test_ignores_non_video_files(self, video_dir: Path) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = str(video_dir / "nonexistent.db")
            files = discover_video_files(str(video_dir))

        filenames = {f["filename"] for f in files}
        assert "notes.txt" not in filenames
        assert "thumbnail.jpg" not in filenames

    def test_discovers_nested_files(self, video_dir: Path) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = str(video_dir / "nonexistent.db")
            files = discover_video_files(str(video_dir))

        filenames = {f["filename"] for f in files}
        assert os.path.join("Season 41", "S41E01.mp4") in filenames
        assert os.path.join("Season 41", "S41E02.mp4") in filenames

    def test_total_file_count(self, video_dir: Path) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = str(video_dir / "nonexistent.db")
            files = discover_video_files(str(video_dir))

        assert len(files) == 5  # 3 flat + 2 nested

    def test_nonexistent_dir_returns_empty(self) -> None:
        files = discover_video_files("/nonexistent/path/abc123")
        assert files == []

    def test_new_files_have_new_status(self, video_dir: Path) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = str(video_dir / "nonexistent.db")
            files = discover_video_files(str(video_dir))

        for f in files:
            assert f["status"] == "New"
            assert f["retry_count"] == 0
            assert f["last_error"] is None

    def test_file_metadata_populated(self, video_dir: Path) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = str(video_dir / "nonexistent.db")
            files = discover_video_files(str(video_dir))

        for f in files:
            assert "filepath" in f
            assert "size_bytes" in f
            assert f["size_bytes"] > 0
            assert os.path.isabs(f["filepath"])


class TestDiscoveryWithDatabase:
    """Discovery enriched with database pipeline status."""

    def test_enriches_with_database_status(self, video_dir: Path, db_with_statuses: str) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = db_with_statuses
            files = discover_video_files(str(video_dir))

        status_map = {f["filename"]: f["status"] for f in files}
        assert status_map["episode1.mp4"] == "PENDING"
        assert status_map["episode2.mkv"] == "FAILED"
        assert status_map["episode3.avi"] == "COMPLETED"

    def test_failed_episode_has_error_info(self, video_dir: Path, db_with_statuses: str) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = db_with_statuses
            files = discover_video_files(str(video_dir))

        failed = [f for f in files if f["status"] == "FAILED"][0]
        assert failed["retry_count"] == 3
        assert failed["last_error"] == "some error"

    def test_nested_episode_id_matches_db(self, video_dir: Path, db_with_statuses: str) -> None:
        """Episode ID for nested files uses path separators replaced with underscores."""
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = db_with_statuses
            files = discover_video_files(str(video_dir))

        nested = [f for f in files if "S41E01" in f["filename"]][0]
        assert nested["status"] == "TRANSCRIPT_READY"

    def test_unknown_file_has_new_status(self, video_dir: Path, db_with_statuses: str) -> None:
        """Files not in the database should have 'New' status."""
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = db_with_statuses
            files = discover_video_files(str(video_dir))

        new_files = [f for f in files if f["status"] == "New"]
        assert len(new_files) >= 1  # At least S41E02 is not in DB


class TestStageFiltering:
    """Stage-based filtering tests."""

    def test_transcribe_filter_excludes_completed(self, video_dir: Path, db_with_statuses: str) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = db_with_statuses
            files = discover_video_files(str(video_dir), stage_filter="transcribe")

        statuses = {f["status"] for f in files}
        assert "COMPLETED" not in statuses
        assert "TRANSCRIPT_READY" not in statuses

    def test_verify_filter_excludes_only_completed(self, video_dir: Path, db_with_statuses: str) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = db_with_statuses
            files = discover_video_files(str(video_dir), stage_filter="verify")

        statuses = {f["status"] for f in files}
        assert "COMPLETED" not in statuses
        # TRANSCRIPT_READY should still be included (not past verify stage)
        assert "TRANSCRIPT_READY" in statuses

    def test_no_filter_returns_all(self, video_dir: Path, db_with_statuses: str) -> None:
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = db_with_statuses
            files_filtered = discover_video_files(str(video_dir), stage_filter="transcribe")
            files_all = discover_video_files(str(video_dir), stage_filter=None)

        assert len(files_all) >= len(files_filtered)

    def test_invalid_stage_filter_returns_all(self, video_dir: Path, db_with_statuses: str) -> None:
        """An unrecognized stage filter should not filter anything."""
        with patch("trebek.pipeline.discovery.settings") as mock_settings:
            mock_settings.db_path = db_with_statuses
            files = discover_video_files(str(video_dir), stage_filter="nonexistent_stage")

        assert len(files) == 5  # All files returned


class TestStageCompletedStatuses:
    """Verify the stage completion status sets are consistent."""

    def test_verify_is_subset_of_augment(self) -> None:
        assert _STAGE_COMPLETED_STATUSES["verify"].issubset(_STAGE_COMPLETED_STATUSES["augment"] | {"COMPLETED"})

    def test_augment_is_subset_of_extract(self) -> None:
        assert _STAGE_COMPLETED_STATUSES["augment"].issubset(_STAGE_COMPLETED_STATUSES["extract"])

    def test_extract_is_subset_of_transcribe(self) -> None:
        assert _STAGE_COMPLETED_STATUSES["extract"].issubset(_STAGE_COMPLETED_STATUSES["transcribe"])
