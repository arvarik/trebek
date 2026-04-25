"""
Centralized Rich console, progress bars, and formatted output for the Trebek pipeline.
All visual output flows through this module to ensure a consistent, premium experience.
"""

from trebek.ui.core import console
import os
import shutil
import subprocess
import sqlite3
import sys
from typing import Any, Optional

from rich.console import Group
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


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
