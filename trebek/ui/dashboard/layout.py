import os
import sqlite3
from typing import Any
from rich.panel import Panel
from rich.console import Group
from rich.table import Table
from rich import box
from trebek.ui.core import console
from trebek.ui.banner import render_startup_banner

from trebek.ui.dashboard.components import (
    generate_health_panel,
    generate_telemetry_panel,
    generate_timing_panel,
    generate_recent_panel,
)


def generate_stats_layout(db_path: str) -> Any:
    """Generates the Rich Group containing the stats dashboard UI."""
    if not os.path.exists(db_path):
        return Panel(
            "[yellow]No database found.[/yellow]\n"
            f"Expected: [bold]{db_path}[/bold]\n\n"
            "Run [bold]trebek[/bold] to start the pipeline and create the database.",
            title="📊 Stats Dashboard",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")

    health_panel = generate_health_panel(conn)
    telemetry_panel = generate_telemetry_panel(conn)
    timing_panel = generate_timing_panel(conn)
    recent_panel = generate_recent_panel(conn)

    conn.close()

    # Top row: Health + Telemetry side by side
    top_grid = Table.grid(padding=2, expand=True)
    top_grid.add_column(ratio=1)
    top_grid.add_column(ratio=1)
    top_grid.add_row(health_panel, telemetry_panel)

    return Group(top_grid, timing_panel, recent_panel)


def render_stats_dashboard(db_path: str) -> None:
    """
    Renders a live-updating analytics dashboard showing pipeline health,
    telemetry, and recent episode history.
    """
    from rich.live import Live
    import time

    console.print()
    render_startup_banner(mode="stats")

    console.print("[dim]Press Ctrl+C to exit dashboard[/dim]\n")

    layout = generate_stats_layout(db_path)

    # If the layout is a Panel indicating no database, just print and exit
    if isinstance(layout, Panel):
        console.print(layout)
        return

    try:
        with Live(layout, console=console, refresh_per_second=1) as live:
            while True:
                time.sleep(2)
                live.update(generate_stats_layout(db_path))
    except KeyboardInterrupt:
        console.print("\n[dim]Exiting dashboard...[/dim]\n")
