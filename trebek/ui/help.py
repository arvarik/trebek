"""
Rich-rendered CLI help pages for Trebek.

Replaces argparse's plain-text help with visually striking, information-dense
panels using the Rich library. Each subcommand has a dedicated help renderer
with a consistent visual language.
"""

from rich.console import Group
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing import Callable
from rich import box

from trebek.ui.core import console
from trebek.ui.banner import TREBEK_ASCII
from trebek import __version__


# ── Design Tokens ────────────────────────────────────────────────────────────
# Curated palette for visual consistency across all help pages.
_C = "bold cyan"  # commands, primary accent
_F = "bold yellow"  # flags / options
_V = "bold magenta"  # values / arguments
_D = "dim white"  # descriptions, secondary text
_A = "bold white"  # section labels, emphasis
_G = "bold green"  # success / positive indicators
_R = "bold red"  # errors / required indicators
_BORDER = "dim cyan"  # panel borders
_BORDER_ALT = "dim yellow"  # alternate panel borders


def _header_panel() -> Panel:
    """Shared branded header used across all help pages."""
    art = Text(TREBEK_ASCII.strip(), style="bold cyan")
    tag = Text(f"  v{__version__}  •  High-fidelity J! data extraction pipeline\n", style=_D)
    return Panel(
        Group(art, tag),
        border_style="cyan",
        box=box.DOUBLE_EDGE,
        padding=(1, 4),
    )


def _subcommand_header(name: str, tagline: str) -> None:
    """Renders a compact header for subcommand help pages."""
    console.print()
    console.print(_header_panel())
    console.print(f"  [{_A}]{name}[/{_A}]  [{_D}]— {tagline}[/{_D}]\n")


def _footer() -> None:
    """Shared footer across all help pages."""
    console.print(
        f"  [{_D}]Docs & source:[/{_D}] [{_C}]https://github.com/arvarik/trebek[/{_C}]  "
        f"[{_D}]•  License: AGPL-3.0[/{_D}]\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN HELP — `trebek --help`
# ═════════════════════════════════════════════════════════════════════════════


def render_main_help() -> None:
    """Renders the main `trebek --help` page."""

    # ── Header ──
    console.print(_header_panel())

    # ── Usage ──
    console.print(f"  [{_A}]USAGE[/{_A}]    [{_C}]trebek[/{_C}] [{_D}]<command>[/{_D}] [{_D}][options][/{_D}]\n")

    # ── Commands ──
    cmd = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    cmd.add_column("Command", style=_C, min_width=14, no_wrap=True)
    cmd.add_column("Description", style="white", min_width=50, no_wrap=True)
    cmd.add_row("run", "Start the extraction pipeline (daemon or one-shot mode)")
    cmd.add_row("scan", "Preview discovered files with real-time pipeline status")
    cmd.add_row("status", "Show real-time queue health, in-flight jobs, and errors")
    cmd.add_row("inspect", "Inspect a single episode: metadata, clues, telemetry, errors")
    cmd.add_row("stats", "Live analytics dashboard — health, cost, timing, errors")
    cmd.add_row("retry", "Reset failed or specific episode(s) back to PENDING")
    cmd.add_row("clean", "Safely purge orphaned audio, tmp files, and obsolete transcripts")
    cmd.add_row("export", "Export an episode to Markdown, JSON, or CSV")
    cmd.add_row("search", "Full-text search across clues and responses with BM25 ranking")
    cmd.add_row("version", "Print version string and exit")

    console.print(
        Panel(
            cmd,
            title="[bold]Commands[/bold]",
            border_style=_BORDER,
            box=box.ROUNDED,
            padding=(1, 1),
        )
    )

    # ── Pipeline Architecture Diagram ──
    flow = Text()
    flow.append("  ┌──────────┐ → ┌────────────┐ → ┌─────────┐ → ┌─────────┐ → ┌─────────┐\n", style=_D)
    flow.append("  │", style=_D)
    flow.append(" ingest   ", style="bold green")
    flow.append("│ → │", style=_D)
    flow.append(" transcribe ", style="bold yellow")
    flow.append("│ → │", style=_D)
    flow.append(" extract ", style="bold blue")
    flow.append("│ → │", style=_D)
    flow.append(" augment ", style="bold magenta")
    flow.append("│ → │", style=_D)
    flow.append(" verify  ", style="bold cyan")
    flow.append("│\n", style=_D)
    flow.append("  │", style=_D)
    flow.append(" discover ", style="dim green")
    flow.append("│   │", style=_D)
    flow.append(" WhisperX   ", style="dim yellow")
    flow.append("│   │", style=_D)
    flow.append(" Gemini  ", style="dim blue")
    flow.append("│   │", style=_D)
    flow.append(" vision  ", style="dim magenta")
    flow.append("│   │", style=_D)
    flow.append(" state   ", style="dim cyan")
    flow.append("│\n", style=_D)
    flow.append("  └──────────┘   └────────────┘   └─────────┘   └─────────┘   └─────────┘\n", style=_D)
    flow.append("               ┈ each stage is ", style=_D)
    flow.append("idempotent", style="bold white")
    flow.append(" ┈ re-run skips completed work ┈", style=_D)

    console.print(
        Panel(
            flow,
            title="[bold]Pipeline Architecture[/bold]",
            border_style=_BORDER,
            box=box.ROUNDED,
            padding=(1, 1),
        )
    )

    # ── Quick Start + Examples side by side ──
    qs = Table(box=None, show_header=False, padding=(0, 1))
    qs.add_column("Cmd", style=_C, min_width=30)
    qs.add_column("Desc", style=_D)
    qs.add_row("trebek scan", "Preview what files will be processed")
    qs.add_row("trebek run --once", "Process everything, then exit")
    qs.add_row("trebek run", "Daemon mode — continuous polling")
    qs.add_row("trebek stats", "Monitor progress with live dashboard")

    ex = Table(box=None, show_header=False, padding=(0, 1))
    ex.add_column("Cmd", style=_C, min_width=38)
    ex.add_column("Desc", style=_D)
    ex.add_row("trebek run --stage transcribe --once", "GPU transcription only, one pass")
    ex.add_row("trebek run --stage extract --model flash", "Cheap LLM extraction pass")
    ex.add_row("trebek scan --stage transcribe", "Files still needing GPU work")
    ex.add_row("trebek retry && trebek run --once", "Re-process all failures")

    console.print(
        Columns(
            [
                Panel(
                    qs,
                    title="[bold]Quick Start[/bold]",
                    border_style="dim green",
                    box=box.ROUNDED,
                    padding=(1, 1),
                    expand=True,
                ),
                Panel(
                    ex,
                    title="[bold]Examples[/bold]",
                    border_style=_BORDER_ALT,
                    box=box.ROUNDED,
                    padding=(1, 1),
                    expand=True,
                ),
            ],
            padding=1,
            expand=True,
        )
    )

    # ── Environment Variables ──
    env = Table(box=None, show_header=True, header_style="bold white", padding=(0, 2))
    env.add_column("Variable", style=_F, min_width=28, no_wrap=True)
    env.add_column("Description", style="white", no_wrap=True)
    env.add_column("Default", style=_D, width=16, no_wrap=True)
    env.add_row("GEMINI_API_KEY", "Google Gemini API key for LLM stages", f"[{_R}]required[/{_R}]")
    env.add_row("INPUT_DIR", "Video file input directory", "input_videos")
    env.add_row("OUTPUT_DIR", "Intermediate pipeline output directory", "gpu_outputs")
    env.add_row("DB_PATH", "SQLite database file path", "trebek.db")
    env.add_row("GPU_VRAM_TARGET_GB", "VRAM budget in GB (4–24)", "16")
    env.add_row("WHISPER_BATCH_SIZE", "WhisperX batch size (tune for VRAM)", "8")
    env.add_row("WHISPER_COMPUTE_TYPE", "Compute precision: float16 | float32", "float16")

    console.print(
        Panel(
            env,
            title="[bold]Environment Variables[/bold]  [dim](also via .env file)[/dim]",
            border_style=_BORDER,
            box=box.ROUNDED,
            padding=(1, 1),
        )
    )

    # ── Supported Formats + Prerequisites side by side ──
    fmt = Table(box=None, show_header=False, padding=(0, 2))
    fmt.add_column("Ext", style=_C, width=8)
    fmt.add_column("Ext", style=_C, width=8)
    fmt.add_column("Ext", style=_C, width=8)
    fmt.add_column("Ext", style=_C, width=8)
    fmt.add_row(".mp4", ".ts", ".mkv", ".avi")
    fmt.add_row(".mov", ".webm", ".mpg", ".mpeg")
    fmt.add_row(".flv", ".wmv", ".m2ts", ".vob")

    prereq = Table(box=None, show_header=False, padding=(0, 2))
    prereq.add_column("Component", style=_A, width=16)
    prereq.add_column("Requirement", style=_D)
    prereq.add_row("Python", "≥ 3.11")
    prereq.add_row("SQLite", "≥ 3.35")
    prereq.add_row("FFmpeg", "Required — audio extraction")
    prereq.add_row("WhisperX", "Required — GPU transcription")
    prereq.add_row("Gemini API", "Required — LLM extraction")

    console.print(
        Columns(
            [
                Panel(
                    fmt,
                    title="[bold]Supported Formats[/bold]",
                    border_style=_BORDER,
                    box=box.ROUNDED,
                    padding=(1, 1),
                    expand=True,
                ),
                Panel(
                    prereq,
                    title="[bold]Prerequisites[/bold]",
                    border_style=_BORDER,
                    box=box.ROUNDED,
                    padding=(1, 1),
                    expand=True,
                ),
            ],
            padding=1,
            expand=True,
        )
    )

    # ── Footer ──
    console.print(f"  [{_D}]Run [{_A}]trebek <command> --help[/{_A}] for detailed command usage.[/{_D}]")
    _footer()


# ═════════════════════════════════════════════════════════════════════════════
#  RUN HELP — `trebek run --help`
# ═════════════════════════════════════════════════════════════════════════════


def render_run_help() -> None:
    """Renders `trebek run --help`."""
    _subcommand_header("trebek run", "Start the extraction pipeline")
    console.print(
        f"  [{_A}]USAGE[/{_A}]    [{_C}]trebek run[/{_C}] [{_D}][--once] [--stage <stage>] [--model <model>] [--input-dir <path>][/{_D}]\n"
    )

    # ── Options ──
    opt = Table(box=None, show_header=True, header_style="bold white", padding=(0, 2))
    opt.add_column("Flag", style=_F, width=18, no_wrap=True)
    opt.add_column("Value", style=_V, width=14, no_wrap=True)
    opt.add_column("Description", style="white")
    opt.add_row("--once", "", "Process all queued episodes then exit (default: daemon mode)")
    opt.add_row("--stage", "<stage>", "Run a specific pipeline stage only (default: all)")
    opt.add_row("--model", "pro | flash", "LLM model for Pass 2 extraction (default: pro)")
    opt.add_row("--input-dir", "<path>", "Override the input video directory")
    opt.add_row("--docker", "", "Delegate execution to a GPU-enabled Docker container")
    opt.add_row("--max-retries", "<n>", "Max automatic retries for failed episodes (default: 3)")

    console.print(Panel(opt, title="[bold]Options[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1)))

    # ── Stages + Models side by side ──
    stg = Table(box=None, show_header=True, header_style="bold white", padding=(0, 2))
    stg.add_column("Stage", style=_C, width=14, no_wrap=True)
    stg.add_column("Engine", style=_D, width=16, no_wrap=True)
    stg.add_column("Description", style="white")
    stg.add_row("all", "—", "Full pipeline, all stages sequentially [dim](default)[/dim]")
    stg.add_row("transcribe", "WhisperX + GPU", "Audio → diarized transcript (.json.gz)")
    stg.add_row("extract", "Gemini LLM", "Transcript → structured Pydantic clue data")
    stg.add_row("augment", "Gemini Vision", "Multimodal visual cue extraction (board, scores)")
    stg.add_row("verify", "State Machine", "Deterministic game-theory validation + DB commit")

    mdl = Table(box=None, show_header=True, header_style="bold white", padding=(0, 2))
    mdl.add_column("Alias", style=_C, width=8, no_wrap=True)
    mdl.add_column("Model ID", style=_D, width=30, no_wrap=True)
    mdl.add_column("Pricing (per M tokens)", style="white")
    mdl.add_row("pro", "gemini-3.1-pro-preview", f"[{_G}]$2.00[/{_G}] in  [{_G}]$12.00[/{_G}] out")
    mdl.add_row("flash3", "gemini-3-flash-preview", f"[{_G}]$0.50[/{_G}] in  [{_G}]$3.00[/{_G}] out")
    mdl.add_row("flash", "gemini-3.1-flash-lite-preview", f"[{_G}]$0.25[/{_G}] in  [{_G}]$1.50[/{_G}] out")

    console.print(
        Columns(
            [
                Panel(
                    stg,
                    title="[bold]Stages[/bold]  [dim]--stage <name>[/dim]",
                    border_style=_BORDER_ALT,
                    box=box.ROUNDED,
                    padding=(1, 1),
                    expand=True,
                ),
                Panel(
                    mdl,
                    title="[bold]Models[/bold]  [dim]--model <alias>[/dim]",
                    border_style="dim magenta",
                    box=box.ROUNDED,
                    padding=(1, 1),
                    expand=True,
                ),
            ],
            padding=1,
            expand=True,
        )
    )

    # ── Execution Modes ──
    modes = Table(box=None, show_header=True, header_style="bold white", padding=(0, 2))
    modes.add_column("Mode", style=_C, width=26, no_wrap=True)
    modes.add_column("Behavior", style="white")
    modes.add_row("trebek run", "Daemon — polls for new files every 5s, runs until Ctrl+C")
    modes.add_row("trebek run --once", "One-shot — processes queue then exits with code 0")
    modes.add_row("trebek run --docker", "Docker — delegates to trebek:latest with GPU passthrough")

    console.print(
        Panel(modes, title="[bold]Execution Modes[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1))
    )

    # ── Behavior Notes ──
    console.print(f"  [{_D}]Notes:[/{_D}]")
    console.print(
        f"    [{_D}]• Each stage is [bold]idempotent[/bold] — re-running only processes unfinished episodes.[/{_D}]"
    )
    console.print(f"    [{_D}]• [{_F}]--stage transcribe --once[/{_F}] auto-resets FAILED episodes to PENDING.[/{_D}]")
    console.print(
        f"    [{_D}]• [{_F}]--docker[/{_F}] mounts CWD + input dir into the container with [{_A}]--gpus all[/{_A}].[/{_D}]"
    )
    console.print(
        f"    [{_D}]• Output goes to [{_A}]$OUTPUT_DIR[/{_A}] (default: gpu_outputs/) and [{_A}]$DB_PATH[/{_A}] (default: trebek.db).[/{_D}]"
    )
    console.print()
    _footer()


# ═════════════════════════════════════════════════════════════════════════════
#  SCAN HELP — `trebek scan --help`
# ═════════════════════════════════════════════════════════════════════════════


def render_scan_help() -> None:
    """Renders `trebek scan --help`."""
    _subcommand_header("trebek scan", "Preview discovered video files with pipeline status")
    console.print(
        f"  [{_A}]USAGE[/{_A}]    [{_C}]trebek scan[/{_C}] [{_D}][--stage <stage>] [--input-dir <path>][/{_D}]\n"
    )

    # ── Options ──
    opt = Table(box=None, show_header=True, header_style="bold white", padding=(0, 2))
    opt.add_column("Flag", style=_F, width=18, no_wrap=True)
    opt.add_column("Value", style=_V, width=20, no_wrap=True)
    opt.add_column("Description", style="white")
    opt.add_row(
        "--stage", "transcribe | extract\naugment | verify", "Only show files that still need work at this stage"
    )
    opt.add_row("--input-dir", "<path>", "Override the input video directory")

    console.print(Panel(opt, title="[bold]Options[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1)))

    # ── Status Legend ──
    legend = Table(box=None, show_header=True, header_style="bold white", padding=(0, 2))
    legend.add_column("Status", width=26, no_wrap=True)
    legend.add_column("Pipeline State", style="white", width=20, no_wrap=True)
    legend.add_column("Meaning", style=_D)
    legend.add_row("[bold green]● New[/bold green]", "—", "File found on disk but not yet ingested")
    legend.add_row("[dim white]⏳ PENDING[/dim white]", "Queued", "Ingested, waiting for GPU transcription")
    legend.add_row("[yellow]🎤 TRANSCRIBING[/yellow]", "In progress", "WhisperX GPU worker is actively processing")
    legend.add_row("[cyan]📝 TRANSCRIPT_READY[/cyan]", "Stage complete", "Transcript done, awaiting LLM extraction")
    legend.add_row("[blue]🧹 CLEANED[/blue]", "In progress", "LLM structured extraction underway")
    legend.add_row(
        "[bright_magenta]🔬 MULTIMODAL[/bright_magenta]", "In progress", "Visual augmentation (board/score snapshots)"
    )
    legend.add_row("[bold green]✅ COMPLETED[/bold green]", "Done", "All stages passed, committed to database")
    legend.add_row("[bold red]❌ FAILED[/bold red]", "Error", "Processing failed — check trebek stats for details")

    console.print(
        Panel(
            legend, title="[bold]Pipeline Status Legend[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1)
        )
    )

    # ── Output Description ──
    console.print(f"  [{_D}]The scan table shows for each file:[/{_D}]")
    console.print(f"    [{_D}]• Relative path, container format, file size[/{_D}]")
    console.print(f"    [{_D}]• Current pipeline status and retry count[/{_D}]")
    console.print(f"    [{_D}]• Last error message (if FAILED)[/{_D}]")
    console.print()

    # ── Examples ──
    console.print(f"  [{_A}]Examples:[/{_A}]")
    console.print(
        f"    [{_C}]trebek scan[/{_C}]                           [{_D}]All files with current pipeline status[/{_D}]"
    )
    console.print(
        f"    [{_C}]trebek scan --stage transcribe[/{_C}]        [{_D}]Files that still need GPU transcription[/{_D}]"
    )
    console.print(
        f"    [{_C}]trebek scan --stage extract[/{_C}]           [{_D}]Files waiting for LLM extraction[/{_D}]"
    )
    console.print(f"    [{_C}]trebek scan --input-dir /mnt/media[/{_C}]    [{_D}]Scan a custom directory[/{_D}]")
    console.print()
    _footer()


# ═════════════════════════════════════════════════════════════════════════════
#  STATS HELP — `trebek stats --help`
# ═════════════════════════════════════════════════════════════════════════════


def render_stats_help() -> None:
    """Renders `trebek stats --help`."""
    _subcommand_header("trebek stats", "Live analytics dashboard")
    console.print(f"  [{_A}]USAGE[/{_A}]    [{_C}]trebek stats[/{_C}]\n")

    # ── Dashboard Panels ──
    panels = Table(box=None, show_header=True, header_style="bold white", padding=(0, 2))
    panels.add_column("Section", style=_C, width=22, no_wrap=True)
    panels.add_column("Metrics Displayed", style="white")
    panels.add_row("Pipeline Health", "Per-status episode counts, completion percentage, progress bar")
    panels.add_row("Performance", "Total tokens consumed, estimated API cost (USD), peak VRAM usage")
    panels.add_row("Stage Timing", "Average / min / max duration per stage (transcribe → verify)")
    panels.add_row("Recent Episodes", "Last 10 episodes with status, retry count, and error messages")
    panels.add_row("Cost Breakdown", "Per-model token usage with input/output cost split")

    console.print(
        Panel(panels, title="[bold]Dashboard Sections[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1))
    )

    console.print(f"  [{_D}]The dashboard auto-refreshes every 2 seconds. Press [{_A}]Ctrl+C[/{_A}] to exit.[/{_D}]")
    console.print(f"  [{_D}]Data is read from [{_A}]$DB_PATH[/{_A}] (default: trebek.db).[/{_D}]")
    console.print()
    _footer()


# ═════════════════════════════════════════════════════════════════════════════
#  RETRY HELP — `trebek retry --help`
# ═════════════════════════════════════════════════════════════════════════════


def render_retry_help() -> None:
    """Renders `trebek retry --help`."""
    _subcommand_header("trebek retry", "Reset failed or specific episodes for re-processing")
    console.print(
        f"  [{_A}]USAGE[/{_A}]    [{_C}]trebek retry[/{_C}] [{_V}]\\[episode_id][/{_V}] [{_F}][--force][/{_F}]\n"
    )

    # ── Arguments & Options ──
    opts = Table(box=None, show_header=False, padding=(0, 2))
    opts.add_column("Argument", style=_F, width=22, no_wrap=True)
    opts.add_column("Description", style="white")
    opts.add_row("episode_id", "Optional specific episode ID to reset (default: all failed)")
    opts.add_row("--force", "Reset the episode even if its status is not FAILED")
    console.print(
        Panel(opts, title="[bold]Arguments & Options[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1))
    )

    # ── Behavior ──
    what = Table(box=None, show_header=False, padding=(0, 2))
    what.add_column("Action", style=_A, width=22, no_wrap=True)
    what.add_column("Detail", style="white")
    what.add_row("Status reset", "Reset back to PENDING")
    what.add_row("Retry count", "Reset to 0 (re-enables automatic retries)")
    what.add_row("Error log", "Cleared — last_error set to NULL")

    console.print(Panel(what, title="[bold]Behavior[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1)))

    # ── Workflow ──
    console.print(f"  [{_A}]Examples:[/{_A}]")
    console.print(f"    [{_C}]trebek retry[/{_C}]                     [{_D}]Reset all FAILED episodes[/{_D}]")
    console.print(f"    [{_C}]trebek retry 2024-03-15_ep8942[/{_C}]   [{_D}]Reset a single failed episode[/{_D}]")
    console.print(
        f"    [{_C}]trebek retry 2024-03-15 --force[/{_C}]  [{_D}]Force-reset an episode from any status[/{_D}]"
    )
    console.print()
    _footer()


# ═════════════════════════════════════════════════════════════════════════════
#  STATUS HELP — `trebek status --help`
# ═════════════════════════════════════════════════════════════════════════════


def render_status_help() -> None:
    """Renders `trebek status --help`."""
    _subcommand_header("trebek status", "Real-time queue health and worker status")
    console.print(f"  [{_A}]USAGE[/{_A}]    [{_C}]trebek status[/{_C}] [{_D}][options][/{_D}]\n")

    opts = Table(box=None, show_header=False, padding=(0, 2))
    opts.add_column("Option", style=_F, width=20, no_wrap=True)
    opts.add_column("Description", style="white")
    opts.add_row("--watch", "Watch queue health in real time with continuous refresh")
    opts.add_row("--json", "Output machine-readable queue statistics as JSON")
    opts.add_row("--refresh-interval", "Seconds between refreshes in watch mode (default: 2.0)")

    console.print(Panel(opts, title="[bold]Options[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1)))
    _footer()


# ═════════════════════════════════════════════════════════════════════════════
#  INSPECT HELP — `trebek inspect --help`
# ═════════════════════════════════════════════════════════════════════════════


def render_inspect_help() -> None:
    """Renders `trebek inspect --help`."""
    _subcommand_header("trebek inspect", "Detailed inspection of an episode")
    console.print(
        f"  [{_A}]USAGE[/{_A}]    [{_C}]trebek inspect[/{_C}] [{_V}]<episode_id>[/{_V}] [{_D}][options][/{_D}]\n"
    )

    opts = Table(box=None, show_header=False, padding=(0, 2))
    opts.add_column("Argument", style=_F, width=20, no_wrap=True)
    opts.add_column("Description", style="white")
    opts.add_row("episode_id", "Required episode ID to inspect")
    opts.add_row("--json", "Dump complete episode inspection record as raw JSON")

    console.print(
        Panel(opts, title="[bold]Arguments & Options[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1))
    )

    console.print(f"  [{_A}]Details inspected:[/{_A}]")
    console.print(f"    [{_D}]• Metadata: Air date, host, tournament format, pipeline status[/{_D}]")
    console.print(f"    [{_D}]• Contestants: Podium, Coryat scores, final scores[/{_D}]")
    console.print(f"    [{_D}]• Clue counts: Daily Doubles, Triple Stumpers, verification status[/{_D}]")
    console.print(f"    [{_D}]• Telemetry: Gemini tokens, spend ($), GPU VRAM, stage latency[/{_D}]")
    console.print(f"    [{_D}]• Quality warnings & last error stack trace[/{_D}]")
    console.print()
    _footer()


# ═════════════════════════════════════════════════════════════════════════════
#  CLEAN HELP — `trebek clean --help`
# ═════════════════════════════════════════════════════════════════════════════


def render_clean_help() -> None:
    """Renders `trebek clean --help`."""
    _subcommand_header("trebek clean", "Purge orphaned and obsolete files")
    console.print(f"  [{_A}]USAGE[/{_A}]    [{_C}]trebek clean[/{_C}] [{_F}][--apply][/{_F}]\n")

    opts = Table(box=None, show_header=False, padding=(0, 2))
    opts.add_column("Option", style=_F, width=20, no_wrap=True)
    opts.add_column("Description", style="white")
    opts.add_row("--apply", "Actually delete files (default runs safe dry-run preview)")

    console.print(Panel(opts, title="[bold]Options[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1)))

    console.print(f"  [{_A}]Purgeable file categories:[/{_A}]")
    console.print(f"    [{_D}]• Orphaned audio chunks (.wav) from interrupted GPU workers[/{_D}]")
    console.print(f"    [{_D}]• Transient temporary files (.tmp, .part)[/{_D}]")
    console.print(f"    [{_D}]• Host interview slices (.mp3) from completed episodes[/{_D}]")
    console.print(f"    [{_D}]• Obsolete transcripts (.json.gz) already committed to SQLite[/{_D}]")
    console.print()
    _footer()


# ═════════════════════════════════════════════════════════════════════════════
#  EXPORT HELP — `trebek export --help`
# ═════════════════════════════════════════════════════════════════════════════


def render_export_help() -> None:
    """Renders `trebek export --help`."""
    _subcommand_header("trebek export", "Export episode data to Markdown, JSON, or CSV")
    console.print(
        f"  [{_A}]USAGE[/{_A}]    [{_C}]trebek export[/{_C}] [{_V}]<episode_id>[/{_V}] [{_D}][options][/{_D}]\n"
    )

    opts = Table(box=None, show_header=False, padding=(0, 2))
    opts.add_column("Argument", style=_F, width=20, no_wrap=True)
    opts.add_column("Description", style="white")
    opts.add_row("episode_id", "Required episode ID to export")
    opts.add_row("--format, -f", "Export format: 'md' (default), 'json', or 'csv'")
    opts.add_row("--output, -o", "Output file path (prints to stdout if omitted)")

    console.print(
        Panel(opts, title="[bold]Arguments & Options[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1))
    )

    console.print(f"  [{_A}]Formats:[/{_A}]")
    console.print(
        f"    [{_C}]md[/{_C}]     [{_D}]Formatted dossier: header, contestants, J! & DJ! boards, Final J![/{_D}]"
    )
    console.print(f"    [{_C}]json[/{_C}]   [{_D}]Full structured JSON object with clues, buzzes, and telemetry[/{_D}]")
    console.print(f"    [{_C}]csv[/{_C}]    [{_D}]Tabular clue list with buzzes, wagers, and responses[/{_D}]")
    console.print()
    _footer()


# ═════════════════════════════════════════════════════════════════════════════
#  SEARCH HELP — `trebek search --help`
# ═════════════════════════════════════════════════════════════════════════════


def render_search_help() -> None:
    """Renders `trebek search --help`."""
    _subcommand_header("trebek search", "Full-text search across extracted clues, categories, and answers")
    console.print(
        f"  [{_A}]USAGE[/{_A}]    [{_C}]trebek search[/{_C}] [{_V}]<query>[/{_V}] [{_D}][options][/{_D}]\n"
    )

    opts = Table(box=None, show_header=False, padding=(0, 2))
    opts.add_column("Argument / Option", style=_F, width=22, no_wrap=True)
    opts.add_column("Description", style="white")
    opts.add_row("query", "Search term or phrase (e.g. 'Shakespeare', 'Moon', 'Presidents')")
    opts.add_row("--round, -r", "Filter by game round: 'J!', 'Double J!', 'Final J!', 'Tiebreaker'")
    opts.add_row("--limit, -l", "Maximum number of results to return (default: 25)")
    opts.add_row("--json", "Output results as structured JSON")

    console.print(
        Panel(opts, title="[bold]Arguments & Options[/bold]", border_style=_BORDER, box=box.ROUNDED, padding=(1, 1))
    )

    console.print(f"  [{_A}]Examples:[/{_A}]")
    console.print(f"    [{_C}]trebek search \"Mount Everest\"[/{_C}]")
    console.print(f"    [{_C}]trebek search Shakespeare --round \"Double J!\"[/{_C}]")
    console.print(f"    [{_C}]trebek search \"civil war\" --limit 10 --json[/{_C}]")
    console.print()
    _footer()


# ═════════════════════════════════════════════════════════════════════════════
#  Dispatcher
# ═════════════════════════════════════════════════════════════════════════════

_HELP_RENDERERS: dict[str, Callable[[], None]] = {
    "main": render_main_help,
    "run": render_run_help,
    "scan": render_scan_help,
    "status": render_status_help,
    "inspect": render_inspect_help,
    "stats": render_stats_help,
    "retry": render_retry_help,
    "clean": render_clean_help,
    "export": render_export_help,
    "search": render_search_help,
}


def render_help(command: str = "main") -> None:
    """Dispatch to the appropriate help renderer."""
    renderer = _HELP_RENDERERS.get(command, render_main_help)
    renderer()

