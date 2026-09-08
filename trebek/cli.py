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

    hf_token = os.environ.get("HF_TOKEN", "") or getattr(settings, "hf_token", "")
    if not hf_token:
        console.print(
            "  [bold yellow]⚠️  HF_TOKEN is not configured:[/bold yellow] Speaker diarization will be skipped,\n"
            "  reducing clue speaker attribution accuracy by ~50%. Set [bold]HF_TOKEN[/bold] in .env to enable.\n"
        )

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
    run_parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=None,
        help="Number of concurrent episodes processed in LLM extraction (default: from .env or 2)",
    )
    run_parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback for WhisperX transcription if CUDA GPU is not detected",
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

    # ── trebek status ────────────────────────────────────────────────
    status_parser = subparsers.add_parser(
        "status",
        help="Show real-time queue health and in-flight workers",
        help_command="status",
    )
    status_parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch queue health in real time with continuous refresh",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable queue status summary as JSON",
    )
    status_parser.add_argument(
        "--refresh-interval",
        type=float,
        default=2.0,
        help="Seconds between refreshes in watch mode (default: 2.0)",
    )

    # ── trebek inspect ───────────────────────────────────────────────
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Detailed inspection of a single episode",
        help_command="inspect",
    )
    inspect_parser.add_argument(
        "episode_id",
        type=str,
        help="Episode ID to inspect",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Dump complete episode inspection record as raw JSON",
    )

    # ── trebek stats ─────────────────────────────────────────────────
    subparsers.add_parser(
        "stats",
        help="Show pipeline analytics dashboard",
        help_command="stats",
    )

    # ── trebek retry ─────────────────────────────────────────────────
    retry_parser = subparsers.add_parser(
        "retry",
        help="Reset failed or specific episodes back to PENDING for re-processing",
        help_command="retry",
    )
    retry_parser.add_argument(
        "episode_id",
        type=str,
        nargs="?",
        default=None,
        help="Optional specific episode ID to reset (default: all failed episodes)",
    )
    retry_parser.add_argument(
        "--force",
        action="store_true",
        help="Reset the episode even if its status is not FAILED",
    )

    # ── trebek clean ─────────────────────────────────────────────────
    clean_parser = subparsers.add_parser(
        "clean",
        help="Safely purge orphaned audio, tmp files, and obsolete transcripts",
        help_command="clean",
    )
    clean_parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files (default: dry run)",
    )

    # ── trebek export ────────────────────────────────────────────────
    export_parser = subparsers.add_parser(
        "export",
        help="Export an episode to Markdown, JSON, or CSV",
        help_command="export",
    )
    export_parser.add_argument(
        "episode_id",
        type=str,
        help="Episode ID to export",
    )
    export_parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["md", "markdown", "json", "csv"],
        default="md",
        help="Export format: 'md' (default), 'json', or 'csv'",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (prints to stdout if omitted)",
    )

    # ── trebek search ────────────────────────────────────────────────
    search_parser = subparsers.add_parser(
        "search",
        help="Full-text search across all extracted clues and responses using FTS5",
        help_command="search",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query term or phrase (e.g. 'Shakespeare' or 'Mount Everest')",
    )
    search_parser.add_argument(
        "--round",
        "-r",
        type=str,
        choices=["J!", "Double J!", "Final J!", "Tiebreaker"],
        default=None,
        help="Filter results by game round",
    )
    search_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=25,
        help="Maximum results to return (default: 25)",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Output search results as raw JSON",
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

    # ── trebek status ────────────────────────────────────────────────
    if command == "status":
        import asyncio
        from trebek.ui.status import run_status_display

        asyncio.run(
            run_status_display(
                settings.db_path,
                watch=getattr(args, "watch", False),
                as_json=getattr(args, "json", False),
                refresh_interval=getattr(args, "refresh_interval", 2.0),
            )
        )
        return

    # ── trebek inspect ───────────────────────────────────────────────
    if command == "inspect":
        from trebek.ui.inspect import handle_inspect_command

        handle_inspect_command(
            settings.db_path,
            args.episode_id,
            output_dir=settings.output_dir,
            as_json=getattr(args, "json", False),
        )
        return

    # ── trebek stats ─────────────────────────────────────────────────
    if command == "stats":
        import asyncio

        asyncio.run(render_stats_dashboard(settings.db_path))
        return

    # ── trebek retry ───────────────────────────────────────────────────────
    if command == "retry":
        import asyncio
        from trebek.database import DatabaseWriter

        async def _retry() -> None:
            writer = DatabaseWriter(settings.db_path)
            await writer.start()
            try:
                ep_id = getattr(args, "episode_id", None)
                force = getattr(args, "force", False)
                count = await writer.reset_episode(ep_id, force=force)
                if count > 0:
                    target = f"episode '{ep_id}'" if ep_id else f"{count} failed episode(s)"
                    console.print(f"  [green]✔[/green] Reset [bold]{target}[/bold] back to PENDING")
                    console.print("  [dim]Run [bold]trebek run --once[/bold] to re-process them.[/dim]")
                else:
                    if ep_id:
                        console.print(
                            f"  [dim]Episode '{ep_id}' not found in FAILED state (pass --force to override).[/dim]"
                        )
                    else:
                        console.print("  [dim]No FAILED episodes found.[/dim]")
            finally:
                await writer.stop()

        asyncio.run(_retry())
        return

    # ── trebek clean ─────────────────────────────────────────────────
    if command == "clean":
        from trebek.ui.cleanup import handle_clean_command

        handle_clean_command(
            output_dir=settings.output_dir,
            db_path=settings.db_path,
            apply=getattr(args, "apply", False),
        )
        return

    # ── trebek export ────────────────────────────────────────────────
    if command == "export":
        from trebek.analysis.export import export_episode

        try:
            content = export_episode(
                settings.db_path,
                args.episode_id,
                format=getattr(args, "format", "md"),
                output_path=getattr(args, "output", None),
                output_dir=settings.output_dir,
            )
            if getattr(args, "output", None):
                console.print(
                    f"  [green]✔[/green] Exported episode [bold]{args.episode_id}[/bold] to [cyan]{args.output}[/cyan]"
                )
            else:
                print(content)
        except Exception as e:
            console.print(f"\n  [bold red]Export error:[/bold red] {e}\n")
            sys.exit(1)
        return

    # ── trebek search ────────────────────────────────────────────────
    if command == "search":
        import asyncio
        from trebek.database import DatabaseWriter

        async def _search() -> None:
            writer = DatabaseWriter(settings.db_path)
            await writer.start()
            try:
                query = args.query
                round_filter = getattr(args, "round", None)
                limit = getattr(args, "limit", 25)
                as_json = getattr(args, "json", False)

                results = await writer.search_clues(query, limit=limit, round_filter=round_filter)

                if as_json:
                    import json

                    print(json.dumps(results, indent=2))
                    return

                if not results:
                    console.print(f"\n  [yellow]No clues found matching:[/yellow] [bold]{query}[/bold]\n")
                    return

                from rich.table import Table

                table = Table(
                    title=f'Search Results for "{query}" ({len(results)} match{"es" if len(results) != 1 else ""})',
                    border_style="cyan",
                    header_style="bold cyan",
                )
                table.add_column("Episode", style="dim", width=18)
                table.add_column("Round", width=12)
                table.add_column("Category", style="bold yellow", width=22)
                table.add_column("Clue Text", width=40)
                table.add_column("Correct Response", style="bold green", width=22)

                for r in results:
                    table.add_row(
                        str(r.get("episode_id", "")),
                        str(r.get("round", "")),
                        str(r.get("category", "")),
                        str(r.get("clue_text", "")),
                        str(r.get("correct_response", "")),
                    )

                console.print()
                console.print(table)
                console.print()
            finally:
                await writer.stop()

        asyncio.run(_search())
        return

    # ── trebek run ───────────────────────────────────────────────────
    input_dir = getattr(args, "input_dir", None) or settings.input_dir

    # Docker delegation
    if getattr(args, "docker", False):
        handle_docker(args, input_dir)
        return

    stage = getattr(args, "stage", "all")
    allow_cpu = getattr(args, "allow_cpu", False) or settings.allow_cpu

    # Hardware check for transcription stages
    if stage in ("all", "transcribe") and not allow_cpu:
        from trebek.gpu.hardware import detect_hardware

        hw = detect_hardware()
        if not hw.is_cuda:
            console.print(
                f"\n  [bold red]Error: CUDA GPU not detected ({hw.device_name}).[/bold red]\n"
                "  WhisperX transcription requires an NVIDIA GPU for performant execution.\n\n"
                "  [dim]Options to proceed:[/dim]\n"
                "  • Pass [bold]--allow-cpu[/bold] to run transcription on CPU (slower fallback).\n"
                "  • Run downstream stages only: [bold]trebek run --stage extract[/bold]\n"
                "  • Run containerized with GPU pass-through: [bold]trebek run --docker[/bold]\n"
            )
            sys.exit(1)

    # Import here to avoid loading heavy modules for scan/stats
    import asyncio
    from trebek.pipeline import run_pipeline

    mode = "once" if getattr(args, "once", False) else "daemon"
    llm_model = MODEL_ALIASES.get(getattr(args, "model", "pro"), MODEL_PRO)
    max_retries = getattr(args, "max_retries", 3)
    llm_concurrency = getattr(args, "llm_concurrency", None)

    asyncio.run(
        run_pipeline(
            mode=mode,
            input_dir_override=input_dir,
            stage=stage,
            llm_model=llm_model,
            max_retries=max_retries,
            llm_concurrency=llm_concurrency,
        )
    )


if __name__ == "__main__":
    main()
