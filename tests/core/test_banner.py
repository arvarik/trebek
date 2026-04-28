"""
Tests for the startup banner rendering — verifies all mode labels render without errors.
"""

from unittest.mock import patch

from trebek.ui.banner import render_startup_banner, TREBEK_ASCII


class TestRenderStartupBanner:
    """Banner rendering tests — verify no crashes for all modes."""

    def test_daemon_mode_renders(self) -> None:
        with patch("trebek.ui.banner.console") as mock_console:
            render_startup_banner(mode="daemon")
            mock_console.print.assert_called_once()

    def test_once_mode_renders(self) -> None:
        with patch("trebek.ui.banner.console") as mock_console:
            render_startup_banner(mode="once")
            mock_console.print.assert_called_once()

    def test_dry_run_mode_renders(self) -> None:
        with patch("trebek.ui.banner.console") as mock_console:
            render_startup_banner(mode="dry-run")
            mock_console.print.assert_called_once()

    def test_stats_mode_renders(self) -> None:
        with patch("trebek.ui.banner.console") as mock_console:
            render_startup_banner(mode="stats")
            mock_console.print.assert_called_once()

    def test_unknown_mode_renders(self) -> None:
        """Unknown modes should use a generic fallback label."""
        with patch("trebek.ui.banner.console") as mock_console:
            render_startup_banner(mode="custom-mode")
            mock_console.print.assert_called_once()

    def test_default_mode_is_daemon(self) -> None:
        with patch("trebek.ui.banner.console") as mock_console:
            render_startup_banner()
            mock_console.print.assert_called_once()


class TestTrebekAscii:
    """Verify the ASCII art constant is well-formed."""

    def test_ascii_art_not_empty(self) -> None:
        assert len(TREBEK_ASCII.strip()) > 0

    def test_ascii_art_contains_trebek(self) -> None:
        # The ASCII art spells "TREBEK" using box-drawing characters
        assert "┳" in TREBEK_ASCII  # Part of the T
