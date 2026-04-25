"""
Centralized Rich console, progress bars, and formatted output for the Trebek pipeline.
All visual output flows through this module to ensure a consistent, premium experience.
"""
import os
import shutil
import subprocess
import sqlite3
import sys
from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.columns import Columns
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box

# ──────────────────────────────────────────────────────────────
# Singleton console — shared across the entire pipeline
# ──────────────────────────────────────────────────────────────

console = Console(stderr=True)


# ──────────────────────────────────────────────────────────────
# Startup Banner
# ──────────────────────────────────────────────────────────────

TREBEK_ASCII = r"""
 ╺┳╸┏━┓┏━╸┏┓ ┏━╸┏┓╻
  ┃ ┣┳┛┣╸ ┣┻┓┣╸ ┣┻┓
  ╹ ╹┗╸┗━╸┗━┛┗━╸╹ ╹
"""


def render_startup_banner(mode: str = "daemon") -> None:
    """Renders the Trebek branded startup panel."""
    mode_label = {
        "daemon": "[bold green]▶ Daemon Mode[/bold green] — Continuous polling",
        "once": "[bold yellow]▶ One-Shot Mode[/bold yellow] — Process queue then exit",
        "dry-run": "[bold blue]▶ Dry Run[/bold blue] — Preview only, no processing",
    }.get(mode, f"[white]▶ {mode}[/white]")

    ascii_art = Text(TREBEK_ASCII.strip(), style="bold cyan")
    tagline = Text("  High-fidelity Jeopardy! data extraction pipeline\n", style="dim white")

    panel = Panel(
        Group(ascii_art, tagline),
        subtitle=f"  {mode_label}  ",
        border_style="cyan",
        box=box.DOUBLE_EDGE,
        padding=(1, 4),
    )
    console.print(panel)


# ──────────────────────────────────────────────────────────────
# System Diagnostics
# ──────────────────────────────────────────────────────────────


def _check_binary(name: str) -> Optional[str]:
    """Checks if a binary is available in PATH and returns its version string."""
    path = shutil.which(name)
    if not path:
        return None
    try:
        result = subprocess.run(
            [name, "-version"] if name == "ffmpeg" else [name, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        # Extract just the first line of version output
        first_line = result.stdout.strip().split("\n")[0] if result.stdout else ""
        return first_line[:60] if first_line else "found"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "found"


def _check_item(label: str, ok: bool, detail: str = "", warn: bool = False) -> Text:
    """Creates a single diagnostics line."""
    if ok and not warn:
        icon = Text("  ✅ ", style="green")
    elif warn:
        icon = Text("  ⚠️  ", style="yellow")
    else:
        icon = Text("  ❌ ", style="red")

    line = Text.assemble(icon, Text(label, style="bold white"))
    if detail:
        line.append(f"  {detail}", style="dim")
    return line


def render_system_diagnostics(settings: Any) -> bool:
    """
    Renders a comprehensive system diagnostics panel at startup.
    Returns True if all critical checks pass, False if there are blockers.
    """
    checks: List[Text] = []
    has_blocker = False

    # ── Python version ──
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)
    checks.append(_check_item("Python", py_ok, py_version, warn=not py_ok))
    if not py_ok:
        has_blocker = True

    # ── SQLite version ──
    sqlite_version = sqlite3.sqlite_version
    sqlite_ok = tuple(int(x) for x in sqlite_version.split(".")) >= (3, 35, 0)
    checks.append(_check_item("SQLite", sqlite_ok, f"{sqlite_version} (≥3.35 required for RETURNING)"))
    if not sqlite_ok:
        has_blocker = True

    # ── FFmpeg ──
    ffmpeg_info = _check_binary("ffmpeg")
    checks.append(_check_item("FFmpeg", ffmpeg_info is not None, ffmpeg_info or "not found in PATH"))
    if not ffmpeg_info:
        has_blocker = True

    # ── WhisperX ──
    whisperx_info = _check_binary("whisperx")
    if whisperx_info:
        checks.append(_check_item("WhisperX", True, whisperx_info))
    else:
        checks.append(_check_item("WhisperX", False, "not found — GPU transcription will fail", warn=True))

    # ── Gemini API Key ──
    api_key = os.environ.get("GEMINI_API_KEY", settings.gemini_api_key)
    if api_key:
        masked = api_key[:4] + "•" * 12 + api_key[-4:] if len(api_key) > 8 else "•" * len(api_key)
        checks.append(_check_item("Gemini API Key", True, masked))
    else:
        checks.append(_check_item("Gemini API Key", False, "not set — LLM stages will fail", warn=True))

    # Build the System Check panel
    check_group = Group(*checks)
    check_panel = Panel(
        check_group,
        title="[bold]System Check[/bold]",
        border_style="dim cyan" if not has_blocker else "red",
        box=box.ROUNDED,
        padding=(0, 1),
    )

    # ── Configuration table ──
    config_table = Table(
        box=None,
        show_header=False,
        padding=(0, 2, 0, 2),
        expand=True,
    )
    config_table.add_column("Key", style="dim white", width=14)
    config_table.add_column("Value", style="bold white")

    config_table.add_row("Database", settings.db_path)
    config_table.add_row("Input Dir", settings.input_dir)
    config_table.add_row("Output Dir", settings.output_dir)
    config_table.add_row("GPU VRAM", f"{settings.gpu_vram_target_gb} GB")
    config_table.add_row("Batch Size", str(settings.whisper_batch_size))
    config_table.add_row("Compute", settings.whisper_compute_type)

    config_panel = Panel(
        config_table,
        title="[bold]Configuration[/bold]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(0, 0),
    )

    # Render side by side if terminal is wide enough, stacked otherwise
    width = console.width
    if width >= 100:
        console.print(Columns([check_panel, config_panel], padding=1, expand=True))
    else:
        console.print(check_panel)
        console.print(config_panel)

    if has_blocker:
        console.print(
            "\n  [bold red]⛔ Critical prerequisites missing.[/bold red] "
            "Resolve the issues above before starting the pipeline.\n"
        )

    return not has_blocker


# ──────────────────────────────────────────────────────────────
# Dry-Run File Discovery Table
# ──────────────────────────────────────────────────────────────

def _get_video_duration(filepath: str) -> Optional[str]:
    """Uses ffprobe to get video duration. Returns formatted string or None."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                filepath,
            ],
            capture_output=True,
            text=True,
            timeout=5,
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
            "\n  [dim]Run [bold]python src/cli.py --once[/bold] to process these files, "
            "or [bold]python src/cli.py[/bold] for continuous daemon mode.[/dim]\n"
        )
    else:
        console.print(
            "\n  [dim]All discovered files are already queued for processing.[/dim]\n"
        )


# ──────────────────────────────────────────────────────────────
# Pipeline Progress Bar
# ──────────────────────────────────────────────────────────────

# Stage display names and their visual style
PIPELINE_STAGES: Dict[str, tuple[str, str]] = {
    "PENDING": ("⏳ Queued", "dim white"),
    "TRANSCRIBING": ("🎤 GPU Transcription", "yellow"),
    "TRANSCRIPT_READY": ("📝 Transcript Ready", "cyan"),
    "CLEANED": ("🧹 LLM Extraction", "blue"),
    "SAVING": ("💾 State Verification", "magenta"),
    "VECTORIZING": ("🧠 Relational Commit", "green"),
    "COMPLETED": ("✅ Done", "bold green"),
    "FAILED": ("❌ Failed", "bold red"),
}


def create_pipeline_progress() -> Progress:
    """Creates a Rich Progress instance configured for pipeline tracking."""
    return Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=35, complete_style="cyan", finished_style="bold green"),
        MofNCompleteColumn(),
        TextColumn("[dim]•[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]•[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def get_stage_display(status: str) -> str:
    """Returns a Rich-formatted stage display string."""
    label, style = PIPELINE_STAGES.get(status, (status, "white"))
    return f"[{style}]{label}[/{style}]"


# ──────────────────────────────────────────────────────────────
# Episode Status Table (for live display)
# ──────────────────────────────────────────────────────────────

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
        stage_label, stage_style = PIPELINE_STAGES.get(
            ep.get("status", "PENDING"), ("Unknown", "white")
        )
        table.add_row(
            ep.get("episode_id", "—"),
            f"[{stage_style}]{stage_label}[/{stage_style}]",
            ep.get("elapsed", "—"),
        )

    return table


# ──────────────────────────────────────────────────────────────
# Shutdown Summary
# ──────────────────────────────────────────────────────────────

def render_shutdown_summary(stats: Dict[str, int]) -> None:
    """Renders a final summary panel after pipeline shutdown."""
    completed = stats.get("completed", 0)
    failed = stats.get("failed", 0)
    total = stats.get("total", 0)
    skipped = total - completed - failed

    if failed > 0 and completed == 0:
        border = "red"
        icon = "❌"
        verdict = "[bold red]All episodes failed.[/bold red]"
    elif failed > 0:
        border = "yellow"
        icon = "⚠️"
        verdict = "[yellow]Some episodes failed. Check logs for details.[/yellow]"
    elif completed > 0:
        border = "green"
        icon = "✅"
        verdict = "[bold green]All episodes processed successfully.[/bold green]"
    else:
        border = "dim"
        icon = "ℹ️"
        verdict = "[dim]No episodes were processed.[/dim]"

    # Build stats table
    stats_table = Table(box=None, show_header=False, padding=(0, 2))
    stats_table.add_column("Label", style="dim")
    stats_table.add_column("Value", style="bold", justify="right")

    stats_table.add_row("Queued", str(total))
    stats_table.add_row("Completed", f"[green]{completed}[/green]")
    if failed > 0:
        stats_table.add_row("Failed", f"[red]{failed}[/red]")
    if skipped > 0:
        stats_table.add_row("Remaining", f"[yellow]{skipped}[/yellow]")

    content = Group(
        Text(f"  {icon}  Pipeline shutdown complete\n", style="bold"),
        stats_table,
        Text(f"\n  {verdict}"),
    )

    console.print()
    console.print(
        Panel(
            content,
            title="[bold]Trebek — Session Summary[/bold]",
            border_style=border,
            box=box.ROUNDED,
            padding=(1, 3),
        )
    )
    console.print()
