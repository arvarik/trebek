"""
Structured logging configuration for the Trebek pipeline.

Configures structlog with Rich ConsoleRenderer for interactive terminals
and JSONRenderer for piped/redirected output. This module should be imported
early in the application lifecycle.
"""

import sys
from typing import Any

import structlog


def configure_logging() -> None:
    """Configures structlog with Rich ConsoleRenderer for TTY, JSONRenderer for piped output."""
    shared_processors = [
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
