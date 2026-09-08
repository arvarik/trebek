"""
Tests for pipeline file discovery — filesystem scanning, stage filtering,
and database status enrichment.
"""

import os
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch

from trebek.pipeline.discovery import (
    discover_video_files,
    _STAGE_COMPLETED_STATUSES,
    derive_episode_id,
    is_candidate_video_file,
    is_file_stable,
    scan_video_files,
    async_scan_video_files,
)


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


class TestDeriveEpisodeId:
    """Tests for normalized, clean episode_id derivation."""

    def test_standard_flat_path(self) -> None:
        assert derive_episode_id("episode1.mp4") == "episode1"
        assert derive_episode_id("show_s01e01.mkv") == "show_s01e01"

    def test_nested_directory_path(self) -> None:
        assert derive_episode_id("Season 41/S41E01.mp4") == "Season_41_S41E01"
        assert derive_episode_id("shows/Season 1/ep01.mp4") == "shows_Season_1_ep01"

    def test_special_characters_sanitized(self) -> None:
        assert derive_episode_id("Jeopardy! [1080p] 2024-01-15.mp4") == "Jeopardy_1080p_2024-01-15"
        assert derive_episode_id("Show (Tournament) #42 @Home.mkv") == "Show_Tournament_42_Home"

    def test_windows_backslashes_normalized(self) -> None:
        assert derive_episode_id(r"Season 41\S41E01.mp4") == "Season_41_S41E01"

    def test_consecutive_underscores_collapsed(self) -> None:
        assert derive_episode_id("Season   41 // S41.mp4") == "Season_41_S41"

    def test_empty_or_all_special_fallback(self) -> None:
        assert derive_episode_id("!.mp4") == "episode"
        assert derive_episode_id("///.mkv") == "episode"


class TestCandidateVideoFile:
    """Tests for valid vs candidate video files."""

    def test_supported_extensions(self) -> None:
        assert is_candidate_video_file("video.mp4") is True
        assert is_candidate_video_file("video.mkv") is True
        assert is_candidate_video_file("VIDEO.MP4") is True
        assert is_candidate_video_file("show.ts") is True

    def test_unsupported_extensions(self) -> None:
        assert is_candidate_video_file("notes.txt") is False
        assert is_candidate_video_file("thumb.jpg") is False
        assert is_candidate_video_file("sub.srt") is False

    def test_temporary_download_extensions_rejected(self) -> None:
        assert is_candidate_video_file("video.mp4.part") is False
        assert is_candidate_video_file("video.mkv.crdownload") is False
        assert is_candidate_video_file("video.mp4.tmp") is False
        assert is_candidate_video_file("video.mp4.temp") is False
        assert is_candidate_video_file("video.ytdl") is False

    def test_hidden_and_backup_files_rejected(self) -> None:
        assert is_candidate_video_file(".DS_Store") is False
        assert is_candidate_video_file("._video.mp4") is False
        assert is_candidate_video_file(".tmp.mp4") is False
        assert is_candidate_video_file("video.mp4~") is False


class TestFileStability:
    """Tests for file stability and in-progress copy detection."""

    def test_zero_byte_file_is_not_stable(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.mp4"
        empty.touch()
        assert is_file_stable(str(empty)) is False

    def test_readable_non_empty_file_is_stable(self, tmp_path: Path) -> None:
        valid = tmp_path / "valid.mp4"
        valid.write_bytes(b"\x00" * 64)
        assert is_file_stable(str(valid)) is True

    def test_recent_file_fails_min_age_check(self, tmp_path: Path) -> None:
        recent = tmp_path / "recent.mp4"
        recent.write_bytes(b"\x00" * 64)
        # min_age_seconds=100.0 will fail for a freshly written file
        assert is_file_stable(str(recent), min_age_seconds=100.0) is False

    def test_missing_file_is_not_stable(self) -> None:
        assert is_file_stable("/path/does/not/exist.mp4") is False


class TestScanVideoFiles:
    """Tests for scan_video_files and async_scan_video_files."""

    def test_skips_zero_byte_and_temporary_files(self, tmp_path: Path) -> None:
        (tmp_path / "valid.mp4").write_text("valid content")
        (tmp_path / "zero.mp4").touch()  # 0 bytes
        (tmp_path / "downloading.mp4.part").write_text("part content")
        (tmp_path / ".hidden.mp4").write_text("hidden content")

        files = scan_video_files(str(tmp_path))
        filenames = {f["filename"] for f in files}
        assert "valid.mp4" in filenames
        assert "zero.mp4" not in filenames
        assert "downloading.mp4.part" not in filenames
        assert ".hidden.mp4" not in filenames

    def test_extension_collision_disambiguation(self, tmp_path: Path) -> None:
        (tmp_path / "S41E01.mp4").write_text("content mp4")
        (tmp_path / "S41E01.mkv").write_text("content mkv")

        files = scan_video_files(str(tmp_path))
        assert len(files) == 2
        episode_ids = {f["episode_id"] for f in files}
        assert "S41E01" in episode_ids
        assert "S41E01_mp4" in episode_ids

    def test_deterministic_sorting(self, tmp_path: Path) -> None:
        (tmp_path / "S41E03.mp4").write_text("ep3")
        (tmp_path / "S41E01.mp4").write_text("ep1")
        (tmp_path / "S41E02.mp4").write_text("ep2")

        files = scan_video_files(str(tmp_path))
        assert [f["filename"] for f in files] == ["S41E01.mp4", "S41E02.mp4", "S41E03.mp4"]

    @pytest.mark.asyncio
    async def test_async_scan_video_files_matches_sync(self, tmp_path: Path) -> None:
        (tmp_path / "test.mp4").write_text("content")
        sync_res = scan_video_files(str(tmp_path))
        async_res = await async_scan_video_files(str(tmp_path))
        assert sync_res == async_res
