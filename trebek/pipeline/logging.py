"""
Structured logging configuration for the Trebek pipeline.

Configures structlog with Rich ConsoleRenderer for interactive terminals
and JSONRenderer for piped/redirected output. This module should be imported
early in the application lifecycle.

Also provides context-binding helpers for automatic ``episode_id`` correlation
across all log messages within an episode's processing scope.
"""

import sys
from typing import Any

import structlog


def configure_logging() -> None:
    """Configures structlog with Rich ConsoleRenderer for TTY, JSONRenderer for piped output."""
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    renderer: Any
    if sys.stderr.isatty():
        # Interactive terminal — beautiful Rich output
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
        )
    else:
        # Piped / redirected — machine-parseable JSON lines
        renderer = structlog.processors.JSONRenderer()

    processors: list[Any] = [*shared_processors, renderer]
    structlog.configure(processors=processors)


def bind_episode_context(episode_id: str) -> None:
    """Bind ``episode_id`` to all subsequent log messages in this async context.

    Call at the start of processing an episode. All log messages from any module
    will automatically include ``episode_id`` until ``clear_episode_context()``
    is called.
    """
    structlog.contextvars.bind_contextvars(episode_id=episode_id)


def clear_episode_context() -> None:
    """Remove ``episode_id`` from the logging context."""
    structlog.contextvars.unbind_contextvars("episode_id")
