"""
Trebek CLI — Entry point for the J! data extraction pipeline.

Provides a subcommand-based interface with Rich-rendered help pages:
    trebek run          Start full pipeline in daemon mode
    trebek run --once   Process queue and exit
    trebek scan         Preview discovered video files
    trebek stats        Show pipeline analytics dashboard
    trebek retry        Reset failed episodes for re-processing
    trebek version      Show version info
"""

import argparse
import os
import sys
from typing import Any

from trebek.config import settings, MODEL_ALIASES, MODEL_PRO
from trebek.pipeline.stages import VALID_STAGES
from trebek.pipeline.discovery import discover_video_files
from trebek.cli_docker import handle_docker
from trebek.ui import (
    console,
    render_startup_banner,
    render_dry_run_table,
    render_stats_dashboard,
    render_system_diagnostics,
)

from trebek.config import SUPPORTED_VIDEO_EXTENSIONS


def handle_scan(input_dir: str, stage_filter: str | None = None) -> None:
    """Scans for video files and renders a preview table."""
    render_startup_banner(mode="scan")
    render_system_diagnostics(settings)

    console.print(f"\n  [dim]Scanning (recursive):[/dim] [bold]{os.path.abspath(input_dir)}[/bold]")
    exts = ", ".join(e.lstrip(".").upper() for e in SUPPORTED_VIDEO_EXTENSIONS[:6])
    console.print(f"  [dim]Formats:[/dim] [bold]{exts}[/bold] [dim]+ 6 more[/dim]")
    if stage_filter:
        console.print(
            f"  [dim]Stage filter:[/dim] [bold yellow]{stage_filter}[/bold yellow] [dim](showing files that still need this stage)[/dim]"
        )
    console.print()

    files = discover_video_files(input_dir, stage_filter=stage_filter)
    render_dry_run_table(files)


# ─────────────────────────────────────────────────────────────────────────────
#  Custom ArgumentParser that uses Rich help rendering
# ─────────────────────────────────────────────────────────────────────────────


class TrebekArgumentParser(argparse.ArgumentParser):
    """ArgumentParser subclass that renders help with Rich instead of plain text."""

    def __init__(self, *args, help_command: str = "main", **kwargs) -> None:  # type: ignore[no-untyped-def]
        self._help_command = help_command
        super().__init__(*args, **kwargs)

    def print_help(self, file: Any = None) -> None:
        from trebek.ui.help import render_help

        render_help(self._help_command)

    def error(self, message: str) -> None:  # type: ignore[override]
        """Override error to show Rich help instead of plain argparse error."""
        console.print(f"\n  [bold red]Error:[/bold red] {message}\n")
        self.print_help()
        sys.exit(2)


def build_parser() -> TrebekArgumentParser:
    """Builds the CLI argument parser with subcommands."""
    parser = TrebekArgumentParser(
        prog="trebek",
        help_command="main",
        add_help=True,
    )

    subparsers = parser.add_subparsers(dest="command", parser_class=TrebekArgumentParser)

    # ── trebek run ───────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Start the pipeline",
        help_command="run",
    )
    run_parser.add_argument(
        "--once",
        action="store_true",
        help="Process all currently queued episodes then exit (no continuous polling)",
    )
    run_parser.add_argument(
        "--stage",
        type=str,
        choices=list(VALID_STAGES),
        default="all",
        help="Which pipeline stage(s) to run (default: all)",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_ALIASES.keys()),
        default="pro",
        help="LLM model for Pass 2 extraction: 'pro' (default), 'flash' (cheapest), or 'flash3' (balanced)",
    )
    run_parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override the input directory (default: from .env or 'input_videos')",
    )
    run_parser.add_argument(
        "--docker",
        action="store_true",
        help="Run the pipeline inside a GPU-enabled Docker container",
    )
    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed episodes (default: 3)",
    )

    # ── trebek scan ──────────────────────────────────────────────────
    scan_parser = subparsers.add_parser(
        "scan",
        help="Preview discovered video files without processing",
        help_command="scan",
    )
    scan_parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override the input directory (default: from .env or 'input_videos')",
    )
    scan_parser.add_argument(
        "--stage",
        type=str,
        choices=["transcribe", "extract", "augment", "verify"],
        default=None,
        help="Only show files that still need work at this stage",
    )

    # ── trebek stats ─────────────────────────────────────────────────
    subparsers.add_parser(
        "stats",
        help="Show pipeline analytics dashboard",
        help_command="stats",
    )

    # ── trebek retry ─────────────────────────────────────────────────
    subparsers.add_parser(
        "retry",
        help="Reset all FAILED episodes back to PENDING for re-processing",
        help_command="retry",
    )

    # ── trebek version ───────────────────────────────────────────────
    subparsers.add_parser(
        "version",
        help="Show version and exit",
        help_command="main",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Default to 'run' if no subcommand is given (preserves `trebek` with no args behavior)
    command = args.command or "run"

    # ── trebek version ───────────────────────────────────────────────
    if command == "version":
        from trebek import __version__

        console.print(f"  [bold cyan]trebek[/bold cyan] [dim]v{__version__}[/dim]")
        return

    # ── trebek scan ──────────────────────────────────────────────────
    if command == "scan":
        input_dir = getattr(args, "input_dir", None) or settings.input_dir
        stage_filter = getattr(args, "stage", None)
        handle_scan(input_dir, stage_filter=stage_filter)
        return

    # ── trebek stats ─────────────────────────────────────────────────
    if command == "stats":
        render_stats_dashboard(settings.db_path)
        return

    # ── trebek retry ───────────────────────────────────────────────────────
    if command == "retry":
        import asyncio
        from trebek.database import DatabaseWriter

        async def _retry() -> None:
            writer = DatabaseWriter(settings.db_path)
            await writer.start()
            try:
                count = await writer.reset_failed_episodes()
                if count > 0:
                    console.print(f"  [green]✔[/green] Reset [bold]{count}[/bold] failed episode(s) back to PENDING")
                    console.print("  [dim]Run [bold]trebek run --once[/bold] to re-process them.[/dim]")
                else:
                    console.print("  [dim]No FAILED episodes found.[/dim]")
            finally:
                await writer.stop()

        asyncio.run(_retry())
        return

    # ── trebek run ───────────────────────────────────────────────────
    input_dir = getattr(args, "input_dir", None) or settings.input_dir

    # Docker delegation
    if getattr(args, "docker", False):
        handle_docker(args, input_dir)
        return

    # Import here to avoid loading heavy modules for scan/stats
    import asyncio
    from trebek.pipeline import run_pipeline

    mode = "once" if getattr(args, "once", False) else "daemon"
    stage = getattr(args, "stage", "all")
    llm_model = MODEL_ALIASES.get(getattr(args, "model", "pro"), MODEL_PRO)
    max_retries = getattr(args, "max_retries", 3)

    asyncio.run(
        run_pipeline(
            mode=mode,
            input_dir_override=input_dir,
            stage=stage,
            llm_model=llm_model,
            max_retries=max_retries,
        )
    )


if __name__ == "__main__":
    main()
