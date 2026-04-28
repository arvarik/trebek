"""
Tests for pipeline logging — structlog configuration, contextvars binding,
and JSON/console renderer selection.
"""

import sys
from unittest.mock import patch

from trebek.pipeline.logging import (
    configure_logging,
    bind_episode_context,
    clear_episode_context,
)
import structlog


class TestConfigureLogging:
    """Logging configuration tests."""

    def test_configure_logging_tty(self) -> None:
        """When stderr is a TTY, should use ConsoleRenderer."""
        with patch.object(sys, "stderr") as mock_stderr:
            mock_stderr.isatty.return_value = True
            configure_logging()
        # Verify structlog is configured (no crash)

    def test_configure_logging_piped(self) -> None:
        """When stderr is piped, should use JSONRenderer."""
        with patch.object(sys, "stderr") as mock_stderr:
            mock_stderr.isatty.return_value = False
            configure_logging()
        # Verify structlog is configured (no crash)


class TestEpisodeContextBinding:
    """Contextvars episode_id binding tests."""

    def test_bind_and_clear(self) -> None:
        bind_episode_context("ep_test_123")
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("episode_id") == "ep_test_123"

        clear_episode_context()
        ctx = structlog.contextvars.get_contextvars()
        assert "episode_id" not in ctx

    def test_bind_overwrites_previous(self) -> None:
        bind_episode_context("ep_001")
        bind_episode_context("ep_002")
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("episode_id") == "ep_002"
        clear_episode_context()

    def test_clear_without_bind_no_error(self) -> None:
        """Clearing without prior bind should not raise."""
        clear_episode_context()
