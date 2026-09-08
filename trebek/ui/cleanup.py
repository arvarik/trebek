"""
Cleanup UI — Rich-rendered display for `trebek clean`.
"""

from typing import List
from rich.panel import Panel
from rich.table import Table
from rich import box

from trebek.ui.core import console
from trebek.ui.banner import render_startup_banner
from trebek.pipeline.cleanup import CleanupItem, scan_cleanup_candidates, execute_cleanup


def format_size(size_bytes: int) -> str:
    """Formats byte counts into human-readable strings."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def render_cleanup_table(items: List[CleanupItem]) -> Panel:
    """Renders a formatted table of files eligible for cleanup."""
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), expand=True)
    table.add_column("File Name", style="bold white", width=34)
    table.add_column("Category", style="cyan", width=26)
    table.add_column("Size", justify="right", style="yellow", width=12)
    table.add_column("Reason", style="dim", width=40)

    total_bytes = 0
    for item in items:
        total_bytes += item.size_bytes
        table.add_row(
            item.filename[:34],
            item.category,
            format_size(item.size_bytes),
            item.reason,
        )

    return Panel(
        table,
        title=f"[bold yellow]Purge Candidates ({len(items)} files · {format_size(total_bytes)})[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def handle_clean_command(output_dir: str, db_path: str, apply: bool = False) -> None:
    """Handles the `trebek clean` CLI subcommand."""
    render_startup_banner(mode="clean --apply" if apply else "clean (dry-run)")

    items = scan_cleanup_candidates(output_dir, db_path)

    if not items:
        console.print("\n  [bold green]✔[/bold green] Output directory is clean. No purgeable files found.\n")
        return

    console.print(render_cleanup_table(items))

    total_bytes = sum(i.size_bytes for i in items)

    if apply:
        count, freed = execute_cleanup(items)
        console.print(
            f"\n  [bold green]✔ Successfully purged {count} files ({format_size(freed)} freed).[/bold green]\n"
        )
    else:
        console.print(
            f"\n  [bold yellow]Dry-run mode:[/bold yellow] Found [bold]{len(items)}[/bold] purgeable files "
            f"([bold]{format_size(total_bytes)}[/bold] reclaimable space).\n"
            f"  [dim]Run [bold]trebek clean --apply[/bold] to delete these files.[/dim]\n"
        )
