"""
Dashboard package — Rich-based analytics dashboard with live-updating panels.

Components render pipeline health, Gemini API telemetry, stage timing
distributions, and recent episode history from the SQLite database.
"""

from .layout import generate_stats_layout, render_stats_dashboard
from .summary import render_shutdown_summary
from .components import (
    generate_health_panel,
    generate_telemetry_panel,
    generate_timing_panel,
    generate_recent_panel,
)

__all__ = [
    "generate_stats_layout",
    "render_stats_dashboard",
    "render_shutdown_summary",
    "generate_health_panel",
    "generate_telemetry_panel",
    "generate_timing_panel",
    "generate_recent_panel",
]
