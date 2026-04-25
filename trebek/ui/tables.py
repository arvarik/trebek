"""
Centralized Rich console, progress bars, and formatted output for the Trebek pipeline.
All visual output flows through this module to ensure a consistent, premium experience.
"""

from trebek.ui.core import console
import subprocess
from typing import Any, Dict, List, Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from trebek.ui.progress import PIPELINE_STAGES


def _get_video_duration(filepath: str) -> Optional[str]:
    """Uses ffprobe to get video duration. Returns formatted string or None."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filepath,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            seconds = float(result.stdout.strip())
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hours > 0:
                return f"{hours}:{minutes:02d}:{secs:02d}"
            return f"{minutes}:{secs:02d}"
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


def _format_file_size(size_bytes: int) -> str:
    """Formats bytes into a human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def render_dry_run_table(files: List[Dict[str, Any]]) -> None:
    """
    Renders a beautifully formatted table of discovered video files.

    Each entry in `files` should have:
        - filename: str
        - filepath: str
        - format: str (extension)
        - size_bytes: int
        - status: str ("New" or "Already Queued")
    """
    if not files:
        console.print(
            Panel(
                "[yellow]No video files found in input directory.[/yellow]\n"
                "Place video files (.mp4, .ts, .mkv, etc.) in the input directory to begin.",
                title="📂 Discovery Results",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        return

    table = Table(
        title="📂 Discovered Video Files",
        box=box.ROUNDED,
        title_style="bold cyan",
        header_style="bold white on rgb(40,40,60)",
        border_style="dim cyan",
        row_styles=["", "dim"],
        padding=(0, 1),
        show_lines=False,
    )

    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Filename", style="white", min_width=30)
    table.add_column("Format", style="magenta", justify="center", width=8)
    table.add_column("Size", style="green", justify="right", width=10)
    table.add_column("Duration", style="cyan", justify="center", width=10)
    table.add_column("Status", justify="center", width=14)

    total_size = 0
    new_count = 0

    for i, f in enumerate(files, 1):
        duration = _get_video_duration(f["filepath"])
        size_str = _format_file_size(f["size_bytes"])
        total_size += f["size_bytes"]

        if f["status"] == "New":
            status_text = Text("● New", style="bold green")
            new_count += 1
        else:
            status_text = Text("○ Queued", style="dim yellow")

        table.add_row(
            str(i),
            f["filename"],
            f["format"].upper().lstrip("."),
            size_str,
            duration or "—",
            status_text,
        )

    console.print()
    console.print(table)

    # Summary line
    summary = (
        f"\n  [bold]{len(files)}[/bold] files discovered  ·  "
        f"[bold green]{new_count}[/bold green] new  ·  "
        f"[bold]{_format_file_size(total_size)}[/bold] total size"
    )
    console.print(summary)

    if new_count > 0:
        console.print(
            "\n  [dim]Run [bold]trebek --once[/bold] to process these files, "
            "or [bold]trebek[/bold] for continuous daemon mode.[/dim]\n"
        )
    else:
        console.print("\n  [dim]All discovered files are already queued for processing.[/dim]\n")


def render_episode_status_table(episodes: List[Dict[str, str]]) -> Table:
    """Creates a Rich Table showing current status of all episodes in the pipeline."""
    table = Table(
        title="Pipeline Status",
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold white",
        border_style="dim",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Episode", style="white", min_width=20)
    table.add_column("Stage", min_width=22)
    table.add_column("Elapsed", style="dim", width=10, justify="right")

    for ep in episodes:
        stage_label, stage_style = PIPELINE_STAGES.get(ep.get("status", "PENDING"), ("Unknown", "white"))
        table.add_row(
            ep.get("episode_id", "—"),
            f"[{stage_style}]{stage_label}[/{stage_style}]",
            ep.get("elapsed", "—"),
        )

    return table


def render_episode_completion_summary(
    episode_id: str,
    clue_count: int,
    contestant_count: int,
    daily_double_count: int,
    final_score: int,
    tokens_used: int,
    cost_usd: float,
    processing_time_s: float,
) -> None:
    """Prints a concise one-line summary when an episode finishes processing."""
    console.print(
        f"  [bold green]✅[/bold green] [bold]{episode_id}[/bold] — "
        f"[cyan]{clue_count}[/cyan] clues · "
        f"[cyan]{contestant_count}[/cyan] contestants · "
        f"[magenta]{daily_double_count}[/magenta] DDs · "
        f"[green]${final_score:,}[/green] final · "
        f"[dim]{tokens_used:,} tok · ${cost_usd:.4f} · {processing_time_s:.1f}s[/dim]"
    )
