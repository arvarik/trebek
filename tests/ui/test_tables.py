import pytest
import subprocess
from unittest.mock import patch, MagicMock
from rich.table import Table
from trebek.ui.tables import (
    _format_file_size,
    _get_video_duration,
    render_dry_run_table,
    render_episode_status_table,
    render_episode_completion_summary
)

@pytest.mark.parametrize(
    "size_bytes, expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024**2, "1.0 MB"),
        (int(1024**2 * 1.5), "1.5 MB"),
        (1024**3, "1.00 GB"),
        (int(1024**3 * 2.5), "2.50 GB"),
    ],
)
def test_format_file_size(size_bytes: int, expected: str) -> None:
    assert _format_file_size(size_bytes) == expected

class TestGetVideoDuration:
    def test_duration_under_hour(self) -> None:
        with patch("trebek.ui.tables.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="45.0")
            assert _get_video_duration("dummy.mp4") == "0:45"

    def test_duration_over_hour(self) -> None:
        with patch("trebek.ui.tables.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="3661.0")
            assert _get_video_duration("dummy.mp4") == "1:01:01"

    def test_ffprobe_failure(self) -> None:
        with patch("trebek.ui.tables.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert _get_video_duration("dummy.mp4") is None

    def test_timeout(self) -> None:
        with patch("trebek.ui.tables.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)):
            assert _get_video_duration("dummy.mp4") is None

    def test_not_found(self) -> None:
        with patch("trebek.ui.tables.subprocess.run", side_effect=FileNotFoundError):
            assert _get_video_duration("dummy.mp4") is None

class TestRenderFunctions:
    def test_render_dry_run_table_empty(self) -> None:
        with patch("trebek.ui.tables.console") as mock_console:
            render_dry_run_table([])
            mock_console.print.assert_called_once()

    def test_render_dry_run_table_with_files(self) -> None:
        files = [
            {
                "filename": "ep1.mp4",
                "filepath": "/path/ep1.mp4",
                "format": ".mp4",
                "size_bytes": 1024 * 1024 * 500,
                "status": "New"
            },
            {
                "filename": "ep2.mp4",
                "filepath": "/path/ep2.mp4",
                "format": ".mp4",
                "size_bytes": 1024 * 1024 * 600,
                "status": "FAILED",
                "retry_count": 1
            },
            {
                "filename": "ep3.mp4",
                "filepath": "/path/ep3.mp4",
                "format": ".mp4",
                "size_bytes": 1024 * 1024 * 700,
                "status": "COMPLETED"
            }
        ]
        with (
            patch("trebek.ui.tables.console") as mock_console,
            patch("trebek.ui.tables._get_video_duration", return_value="20:00")
        ):
            render_dry_run_table(files)
            # Should print something (table and summary)
            assert mock_console.print.called

    def test_render_episode_status_table(self) -> None:
        episodes = [
            {"episode_id": "J_1234", "status": "PENDING", "elapsed": "0:05"},
            {"episode_id": "J_5678", "status": "COMPLETED", "elapsed": "2:30"}
        ]
        table = render_episode_status_table(episodes)
        assert isinstance(table, Table)
        assert table.title == "Pipeline Status"

    def test_render_episode_completion_summary(self) -> None:
        with patch("trebek.ui.tables.console") as mock_console:
            render_episode_completion_summary(
                "J_1234", 61, 3, 3, 25000, 50000, 0.15, 120.5
            )
            mock_console.print.assert_called_once()
