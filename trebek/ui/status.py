"""
Queue status UI — Rich-rendered display for `trebek status`.
"""

import asyncio
import json
from typing import Any, Dict
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich import box
from rich.live import Live

from trebek.ui.core import console
from trebek.ui.banner import render_startup_banner
from trebek.analysis.status import get_queue_status


def render_queue_status(status_data: Dict[str, Any]) -> Group:
    """Renders real-time queue health as Rich renderables."""
    if not status_data.get("database_found", False):
        panel = Panel(
            f"[yellow]No database found at [bold]{status_data.get('db_path')}[/bold][/yellow]\n"
            "Run [bold]trebek run[/bold] or [bold]trebek scan[/bold] to initialize the pipeline.",
            title="[bold yellow]Queue Health[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        return Group(panel)

    # ── Status Breakdown Table ──
    status_table = Table(box=None, show_header=False, padding=(0, 2), expand=True)
    status_table.add_column("Stage", style="dim white", width=26)
    status_table.add_column("Count", style="bold", justify="right", width=8)

    counts = status_data.get("status_counts", {})
    total = status_data.get("total", 0)

    status_table.add_row("Total Registered Episodes", f"[bold cyan]{total}[/bold cyan]")
    status_table.add_row("─" * 26, "─" * 8)

    stages_config = [
        ("PENDING", "⏳ Pending (Unprocessed)", "dim white"),
        ("TRANSCRIBING", "🎤 Transcribing (GPU)", "bold yellow"),
        ("TRANSCRIPT_READY", "📝 Transcript Ready", "cyan"),
        ("CLEANED", "🧹 Extracting (LLM)", "bold blue"),
        ("SAVING", "💾 Saving Structured Data", "magenta"),
        ("MULTIMODAL_PROCESSING", "🔬 Multimodal Sniping", "bold magenta"),
        ("MULTIMODAL_DONE", "🔬 Multimodal Ready", "cyan"),
        ("VECTORIZING", "🧠 State Machine / Commit", "bold green"),
        ("COMPLETED", "✅ Completed", "green"),
        ("FAILED", "❌ Failed", "bold red"),
    ]

    for key, label, color in stages_config:
        cnt = counts.get(key, 0)
        status_table.add_row(f"  {label}", f"[{color}]{cnt}[/{color}]")

    queue_panel = Panel(
        status_table,
        title="[bold cyan]Episode Queue Health[/bold cyan]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )

    # ── In-Flight Jobs Table ──
    in_flight = status_data.get("in_flight", [])
    if in_flight:
        flight_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), expand=True)
        flight_table.add_column("Episode ID", style="bold white", width=28)
        flight_table.add_column("Active Stage", style="cyan", width=22)
        flight_table.add_column("Retries", justify="center", width=8)
        flight_table.add_column("Updated At", style="dim", justify="right")

        for item in in_flight:
            flight_table.add_row(
                item["episode_id"][:28],
                item["status"],
                str(item["retry_count"]),
                str(item["updated_at"]),
            )

        flight_panel = Panel(
            flight_table,
            title="[bold yellow]⚡ In-Flight Processing[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    else:
        flight_panel = Panel(
            "[dim]No active workers currently executing jobs.[/dim]",
            title="[bold yellow]⚡ In-Flight Processing[/bold yellow]",
            border_style="dim yellow",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    # ── Recent Errors Table ──
    recent_errors = status_data.get("recent_errors", [])
    renderables = [queue_panel, flight_panel]

    if recent_errors:
        err_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), expand=True)
        err_table.add_column("Episode ID", style="bold red", width=26)
        err_table.add_column("Last Error", style="white", width=42)
        err_table.add_column("Retries", justify="center", width=8)
        err_table.add_column("Failed At", style="dim", justify="right")

        for err in recent_errors:
            err_msg = (err["last_error"] or "Unknown error").replace("\n", " ")
            if len(err_msg) > 60:
                err_msg = err_msg[:57] + "..."
            err_table.add_row(
                err["episode_id"][:26],
                err_msg,
                str(err["retry_count"]),
                str(err["updated_at"]),
            )

        error_panel = Panel(
            err_table,
            title="[bold red]⚠️ Recent Failures[/bold red]",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 1),
        )
        renderables.append(error_panel)

    return Group(*renderables)


async def run_status_display(
    db_path: str,
    watch: bool = False,
    as_json: bool = False,
    refresh_interval: float = 2.0,
) -> None:
    """Renders queue status once or in watch mode."""
    if as_json:
        data = get_queue_status(db_path)
        print(json.dumps(data, indent=2, default=str))
        return

    render_startup_banner(mode="status")

    if not watch:
        data = get_queue_status(db_path)
        console.print(render_queue_status(data))
        return

    console.print("[dim]Watching queue health (refreshes every 2s) • Press Ctrl+C to exit[/dim]\n")
    try:
        with Live(render_queue_status(get_queue_status(db_path)), console=console, refresh_per_second=2) as live:
            while True:
                await asyncio.sleep(refresh_interval)
                live.update(render_queue_status(get_queue_status(db_path)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
