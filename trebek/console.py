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
        "stats": "[bold magenta]▶ Stats[/bold magenta] — Database analytics dashboard",
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
            capture_output=True,
            text=True,
            timeout=5,
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
    has_blocker = False

    # ── Diagnostic Table Setup ──
    diag_table = Table(box=None, show_header=False, padding=(0, 2, 0, 1), expand=True)
    diag_table.add_column("Icon", justify="center", width=4)
    diag_table.add_column("Component", style="bold white", width=16)
    diag_table.add_column("Details", style="dim white")

    def add_check(label: str, ok: bool, detail: str = "", warn: bool = False) -> None:
        if ok and not warn:
            icon = "[bold green]✓[/bold green]"
        elif warn:
            icon = "[bold yellow]![/bold yellow]"
        else:
            icon = "[bold red]✗[/bold red]"
        diag_table.add_row(icon, label, detail)

    # ── Python version ──
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)
    add_check("Python", py_ok, py_version, warn=not py_ok)
    if not py_ok:
        has_blocker = True

    # ── SQLite version ──
    sqlite_version = sqlite3.sqlite_version
    sqlite_ok = tuple(int(x) for x in sqlite_version.split(".")) >= (3, 35, 0)
    add_check("SQLite", sqlite_ok, f"{sqlite_version} (≥3.35 required)")
    if not sqlite_ok:
        has_blocker = True

    # ── FFmpeg ──
    ffmpeg_info = _check_binary("ffmpeg")
    add_check("FFmpeg", ffmpeg_info is not None, ffmpeg_info or "not found in PATH")
    if not ffmpeg_info:
        has_blocker = True

    # ── WhisperX ──
    whisperx_info = _check_binary("whisperx")
    if whisperx_info:
        add_check("WhisperX", True, whisperx_info)
    else:
        add_check("WhisperX", False, "not found — GPU transcription will fail", warn=True)

    # ── Gemini API Key ──
    api_key = os.environ.get("GEMINI_API_KEY", settings.gemini_api_key)
    if api_key:
        masked = api_key[:4] + "•" * 12 + api_key[-4:] if len(api_key) > 8 else "•" * len(api_key)
        add_check("Gemini API Key", True, masked)
    else:
        add_check("Gemini API Key", False, "not set — LLM stages will fail", warn=True)

    check_panel = Panel(
        diag_table,
        title="[bold]System Check[/bold]",
        border_style="dim cyan" if not has_blocker else "red",
        box=box.ROUNDED,
        padding=(1, 1),
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

    config_table.add_row("Database", f"[cyan]{settings.db_path}[/cyan]")
    config_table.add_row("Input Dir", f"[blue]{settings.input_dir}[/blue]")
    config_table.add_row("Output Dir", f"[blue]{settings.output_dir}[/blue]")
    config_table.add_row("GPU VRAM", f"[magenta]{settings.gpu_vram_target_gb} GB[/magenta]")
    config_table.add_row("Batch Size", f"[magenta]{settings.whisper_batch_size}[/magenta]")
    config_table.add_row("Compute", f"[magenta]{settings.whisper_compute_type}[/magenta]")

    config_panel = Panel(
        config_table,
        title="[bold]Configuration[/bold]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 1),
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
            "Resolve the issues above, or run [bold]trebek --docker[/bold] to use the containerized version.\n"
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
        SpinnerColumn(spinner_name="dots2", style="cyan"),
        TextColumn("[bold white]{task.description}[/bold white]"),
        BarColumn(bar_width=40, complete_style="magenta", finished_style="bold green"),
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
        stage_label, stage_style = PIPELINE_STAGES.get(ep.get("status", "PENDING"), ("Unknown", "white"))
        table.add_row(
            ep.get("episode_id", "—"),
            f"[{stage_style}]{stage_label}[/{stage_style}]",
            ep.get("elapsed", "—"),
        )

    return table


# ──────────────────────────────────────────────────────────────
# Shutdown Summary
# ──────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────
# Per-Episode Completion Summary (inline during pipeline run)
# ──────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────
# Stats Dashboard (trebek --stats)
# ──────────────────────────────────────────────────────────────


def generate_stats_layout(db_path: str) -> Any:
    """Generates the Rich Group containing the stats dashboard UI."""
    import sqlite3 as _sqlite3

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

    conn = _sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")

    # ── Pipeline Health ──
    health_table = Table(box=None, show_header=False, padding=(0, 2))
    health_table.add_column("Metric", style="dim white", width=18)
    health_table.add_column("Value", style="bold", justify="right", width=10)

    status_counts: Dict[str, int] = {}
    try:
        for row in conn.execute("SELECT status, COUNT(*) FROM pipeline_state GROUP BY status"):
            status_counts[row[0]] = row[1]
    except _sqlite3.OperationalError:
        pass

    total = sum(status_counts.values())
    completed = status_counts.get("COMPLETED", 0)
    failed = status_counts.get("FAILED", 0)
    in_progress = total - completed - failed

    health_table.add_row("Total Episodes", f"[bold]{total}[/bold]")
    health_table.add_row("Completed", f"[green]{completed}[/green]")
    if failed > 0:
        health_table.add_row("Failed", f"[red]{failed}[/red]")
    if in_progress > 0:
        health_table.add_row("In Progress", f"[yellow]{in_progress}[/yellow]")

    success_rate = (completed / total * 100) if total > 0 else 0
    health_table.add_row("Success Rate", f"[cyan]{success_rate:.0f}%[/cyan]")

    health_panel = Panel(
        health_table,
        title="[bold]Pipeline Health[/bold]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=True,
    )

    # ── Telemetry Aggregates ──
    telemetry_table = Table(box=None, show_header=False, padding=(0, 2))
    telemetry_table.add_column("Metric", style="dim white", width=18)
    telemetry_table.add_column("Value", style="bold", justify="right", width=14)

    try:
        row = conn.execute("""
            SELECT
                SUM(gemini_total_input_tokens + gemini_total_output_tokens + gemini_total_cached_tokens),
                SUM(gemini_total_cost_usd),
                AVG(peak_vram_mb),
                AVG(stage_gpu_extraction_ms),
                AVG(stage_structured_extraction_ms),
                AVG(pydantic_retry_count),
                COUNT(*)
            FROM job_telemetry
        """).fetchone()

        if row and row[6] and row[6] > 0:
            total_tokens = int(row[0] or 0)
            total_cost = float(row[1] or 0.0)
            avg_vram = float(row[2] or 0.0)
            avg_gpu_ms = float(row[3] or 0.0)
            avg_llm_ms = float(row[4] or 0.0)
            avg_retries = float(row[5] or 0.0)

            telemetry_table.add_row("Total Tokens", f"[cyan]{total_tokens:,}[/cyan]")
            telemetry_table.add_row("Total API Cost", f"[green]${total_cost:.4f}[/green]")
            telemetry_table.add_row("Avg Peak VRAM", f"[magenta]{avg_vram / 1024:.1f} GB[/magenta]")
            telemetry_table.add_row("Avg GPU Stage", f"[blue]{avg_gpu_ms / 1000:.1f}s[/blue]")
            telemetry_table.add_row("Avg LLM Stage", f"[blue]{avg_llm_ms / 1000:.1f}s[/blue]")
            telemetry_table.add_row("Avg Retries", f"[yellow]{avg_retries:.1f}[/yellow]")
        else:
            telemetry_table.add_row("[dim]No telemetry data yet[/dim]", "")
    except _sqlite3.OperationalError:
        telemetry_table.add_row("[dim]Telemetry table not found[/dim]", "")

    telemetry_panel = Panel(
        telemetry_table,
        title="[bold]Performance[/bold]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=True,
    )

    # ── Stage Timing Breakdown ──
    timing_table = Table(
        box=box.SIMPLE,
        header_style="bold white",
        border_style="dim",
        padding=(0, 1),
        expand=True,
    )
    timing_table.add_column("Stage", style="white", min_width=22)
    timing_table.add_column("Avg (s)", justify="right", width=10)
    timing_table.add_column("Min (s)", justify="right", width=10, style="dim")
    timing_table.add_column("Max (s)", justify="right", width=10, style="dim")

    stage_columns = [
        ("Ingestion", "stage_ingestion_ms"),
        ("GPU Extraction", "stage_gpu_extraction_ms"),
        ("LLM Extraction", "stage_structured_extraction_ms"),
        ("Vectorization", "stage_vectorization_ms"),
    ]

    try:
        for label, col in stage_columns:
            srow = conn.execute(
                f"SELECT AVG({col}), MIN({col}), MAX({col}) FROM job_telemetry WHERE {col} IS NOT NULL"
            ).fetchone()
            if srow and srow[0] is not None:
                timing_table.add_row(
                    label,
                    f"[cyan]{srow[0] / 1000:.1f}[/cyan]",
                    f"{srow[1] / 1000:.1f}",
                    f"{srow[2] / 1000:.1f}",
                )
    except _sqlite3.OperationalError:
        pass

    timing_panel = Panel(
        timing_table,
        title="[bold]Stage Timing Breakdown[/bold]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 1),
    )

    # ── Recent Episodes ──
    recent_table = Table(
        box=box.SIMPLE,
        header_style="bold white",
        border_style="dim",
        padding=(0, 1),
        expand=True,
    )
    recent_table.add_column("#", style="dim", width=4, justify="right")
    recent_table.add_column("Episode", style="white", min_width=20)
    recent_table.add_column("Status", justify="center", width=14)
    recent_table.add_column("Updated", style="dim", width=20)

    try:
        recent_rows = conn.execute(
            "SELECT episode_id, status, updated_at FROM pipeline_state ORDER BY updated_at DESC LIMIT 10"
        ).fetchall()
        for i, rrow in enumerate(recent_rows, 1):
            stage_label, stage_style = PIPELINE_STAGES.get(rrow[1], (rrow[1], "white"))
            recent_table.add_row(
                str(i),
                rrow[0],
                f"[{stage_style}]{stage_label}[/{stage_style}]",
                str(rrow[2] or "—"),
            )
    except _sqlite3.OperationalError:
        pass

    recent_panel = Panel(
        recent_table,
        title="[bold]Recent Episodes[/bold]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 1),
    )

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
