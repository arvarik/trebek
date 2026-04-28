"""
Tests for system diagnostics utility functions — _check_item and _check_binary.
"""

from unittest.mock import patch, MagicMock

from trebek.ui.diagnostics import _check_binary, _check_item, render_system_diagnostics


class TestCheckBinary:
    """Tests for the binary availability checker."""

    def test_found_binary(self) -> None:
        with (
            patch("trebek.ui.diagnostics.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("trebek.ui.diagnostics.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(stdout="ffmpeg version 6.1.1\nmore info\n")
            result = _check_binary("ffmpeg")
            assert result is not None
            assert "ffmpeg version" in result

    def test_not_found_binary(self) -> None:
        with patch("trebek.ui.diagnostics.shutil.which", return_value=None):
            result = _check_binary("nonexistent_tool")
            assert result is None

    def test_timeout_returns_found(self) -> None:
        import subprocess

        with (
            patch("trebek.ui.diagnostics.shutil.which", return_value="/usr/bin/slow"),
            patch("trebek.ui.diagnostics.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="slow", timeout=5)),
        ):
            result = _check_binary("slow")
            assert result == "found"

    def test_empty_stdout_returns_found(self) -> None:
        with (
            patch("trebek.ui.diagnostics.shutil.which", return_value="/usr/bin/tool"),
            patch("trebek.ui.diagnostics.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(stdout="")
            result = _check_binary("tool")
            assert result == "found"

    def test_ffmpeg_uses_dash_version_flag(self) -> None:
        """FFmpeg uses -version (not --version)."""
        with (
            patch("trebek.ui.diagnostics.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("trebek.ui.diagnostics.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(stdout="ffmpeg v6")
            _check_binary("ffmpeg")
            args = mock_run.call_args[0][0]
            assert args == ["ffmpeg", "-version"]

    def test_non_ffmpeg_uses_double_dash(self) -> None:
        """Non-ffmpeg tools use --version."""
        with (
            patch("trebek.ui.diagnostics.shutil.which", return_value="/usr/bin/whisperx"),
            patch("trebek.ui.diagnostics.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(stdout="whisperx v1")
            _check_binary("whisperx")
            args = mock_run.call_args[0][0]
            assert args == ["whisperx", "--version"]


class TestCheckItem:
    """Tests for the diagnostics line formatter."""

    def test_ok_item_green(self) -> None:
        text = _check_item("Python", ok=True, detail="3.12.1")
        plain = text.plain
        assert "Python" in plain
        assert "3.12.1" in plain

    def test_failed_item_red(self) -> None:
        text = _check_item("FFmpeg", ok=False, detail="not found")
        plain = text.plain
        assert "FFmpeg" in plain

    def test_warn_item(self) -> None:
        text = _check_item("WhisperX", ok=True, detail="optional", warn=True)
        plain = text.plain
        assert "WhisperX" in plain

    def test_no_detail(self) -> None:
        text = _check_item("Check", ok=True)
        plain = text.plain
        assert "Check" in plain


class TestRenderSystemDiagnostics:
    """Integration test for the full diagnostics panel render."""

    def test_renders_without_crash(self) -> None:
        mock_settings = MagicMock()
        mock_settings.db_path = "/tmp/test.db"
        mock_settings.input_dir = "/tmp/input"
        mock_settings.output_dir = "/tmp/output"
        mock_settings.gpu_vram_target_gb = 8
        mock_settings.whisper_batch_size = 16
        mock_settings.whisper_compute_type = "float16"
        mock_settings.gemini_api_key = ""

        with patch("trebek.ui.diagnostics.console") as mock_console:
            mock_console.width = 120
            result = render_system_diagnostics(mock_settings)
            assert isinstance(result, bool)

    def test_no_api_key_shows_warning(self) -> None:
        mock_settings = MagicMock()
        mock_settings.db_path = "/tmp/test.db"
        mock_settings.input_dir = "/tmp/input"
        mock_settings.output_dir = "/tmp/output"
        mock_settings.gpu_vram_target_gb = 8
        mock_settings.whisper_batch_size = 16
        mock_settings.whisper_compute_type = "float16"
        mock_settings.gemini_api_key = ""

        with patch("trebek.ui.diagnostics.console") as mock_console, patch.dict("os.environ", {}, clear=True):
            mock_console.width = 120
            result = render_system_diagnostics(mock_settings)
            # No blocker just because API key is missing (it's a warning)
            assert isinstance(result, bool)
