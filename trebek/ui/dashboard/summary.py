from typing import Dict, Optional
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich import box

from trebek.ui.core import console


def render_shutdown_summary(stats: Dict[str, int], telemetry_stats: Optional[Dict[str, float]] = None) -> None:
    """Renders a final premium split-pane summary panel after pipeline shutdown."""
    completed = stats.get("completed", 0)
    failed = stats.get("failed", 0)
    total = stats.get("total", 0)
    skipped = total - completed - failed

    if failed > 0 and completed == 0:
        border = "red"
        icon = "[bold red]✗[/bold red]"
        verdict = "[bold red]All episodes failed.[/bold red]"
    elif failed > 0:
        border = "yellow"
        icon = "[bold yellow]![/bold yellow]"
        verdict = "[yellow]Some episodes failed. Check logs for details.[/yellow]"
    elif completed > 0:
        border = "cyan"
        icon = "[bold cyan]✓[/bold cyan]"
        verdict = "[bold cyan]All episodes processed successfully.[/bold cyan]"
    else:
        border = "dim"
        icon = "[dim]ℹ[/dim]"
        verdict = "[dim]No episodes were processed.[/dim]"

    # ── Left Pane: Job Outcomes ──
    outcomes_table = Table(box=None, show_header=False, padding=(0, 2))
    outcomes_table.add_column("Label", style="dim white", width=12)
    outcomes_table.add_column("Value", style="bold", justify="right", width=6)

    outcomes_table.add_row("Queued", str(total))
    outcomes_table.add_row("Completed", f"[green]{completed}[/green]")
    if failed > 0:
        outcomes_table.add_row("Failed", f"[red]{failed}[/red]")
    if skipped > 0:
        outcomes_table.add_row("Remaining", f"[yellow]{skipped}[/yellow]")

    outcomes_panel = Panel(
        outcomes_table,
        title="[bold]Job Outcomes[/bold]",
        border_style="dim",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=True,
    )

    # ── Right Pane: Performance Telemetry ──
    telemetry_table = Table(box=None, show_header=False, padding=(0, 2))
    telemetry_table.add_column("Metric", style="dim white", width=18)
    telemetry_table.add_column("Value", style="bold", justify="right", width=12)

    if telemetry_stats and completed > 0:
        tokens = int(telemetry_stats.get("total_tokens", 0))
        cost = telemetry_stats.get("total_cost", 0.0)
        vram = telemetry_stats.get("avg_peak_vram", 0.0)
        ext_time = telemetry_stats.get("avg_extraction_ms", 0.0) / 1000.0

        telemetry_table.add_row("Total Tokens", f"[cyan]{tokens:,}[/cyan]")
        telemetry_table.add_row("Est. API Cost", f"[green]${cost:.4f}[/green]")
        telemetry_table.add_row("Avg Peak VRAM", f"[magenta]{vram / 1024:.1f} GB[/magenta]")
        telemetry_table.add_row("Avg Ext. Time", f"[blue]{ext_time:.1f}s[/blue]")
    else:
        telemetry_table.add_row("No telemetry data", "")

    telemetry_panel = Panel(
        telemetry_table,
        title="[bold]Performance[/bold]",
        border_style="dim",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=True,
    )

    # Assemble Split Pane using Table.grid for strict horizontal layout
    split_pane = Table.grid(padding=2, expand=True)
    split_pane.add_column(ratio=1)
    split_pane.add_column(ratio=1)
    split_pane.add_row(outcomes_panel, telemetry_panel)

    content = Group(
        Text.from_markup(f"  {icon}  [bold]Pipeline shutdown complete[/bold]\n"),
        split_pane,
        Text.from_markup(f"\n  {verdict}"),
    )

    console.print()
    console.print(
        Panel(
            content,
            title="[bold]Trebek — Session Summary[/bold]",
            border_style=border,
            box=box.DOUBLE_EDGE,
            padding=(1, 3),
        )
    )
    console.print()
