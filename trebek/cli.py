"""
Trebek CLI — Entry point for the pipeline daemon.

Usage:
    trebek                    # Start continuous daemon mode
    trebek --once             # Process current queue then exit
    trebek --dry-run          # Preview discovered files, no processing
    trebek --input-dir /path  # Override input directory
"""

import argparse
import os
import sqlite3

from trebek.config import settings, SUPPORTED_VIDEO_EXTENSIONS
from trebek.ui import (
    console,
    render_startup_banner,
    render_dry_run_table,
    render_stats_dashboard,
    render_system_diagnostics,
)


def discover_video_files(input_dir: str) -> list[dict[str, object]]:
    """Recursively scans input_dir for all supported video files."""
    files: list[dict[str, object]] = []

    if not os.path.exists(input_dir):
        return files

    # Check which episodes are already queued in the database
    queued_ids: set[str] = set()
    db_path = settings.db_path
    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute("SELECT episode_id FROM pipeline_state").fetchall()
                queued_ids = {row[0] for row in rows}
        except sqlite3.OperationalError:
            pass  # DB may not have the table yet

    for dirpath, _dirnames, filenames in os.walk(input_dir):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_VIDEO_EXTENSIONS:
                continue

            filepath = os.path.join(dirpath, fname)
            rel = os.path.relpath(filepath, input_dir)
            episode_id = os.path.splitext(rel)[0].replace(os.sep, "_").replace(" ", "_")
            status = "Already Queued" if episode_id in queued_ids else "New"

            files.append(
                {
                    "filename": rel,
                    "filepath": filepath,
                    "format": ext,
                    "size_bytes": os.path.getsize(filepath),
                    "status": status,
                }
            )

    return files


def handle_dry_run(input_dir: str) -> None:
    """Scans for video files and renders a preview table."""
    render_startup_banner(mode="dry-run")
    render_system_diagnostics(settings)

    console.print(f"\n  [dim]Scanning (recursive):[/dim] [bold]{os.path.abspath(input_dir)}[/bold]")
    exts = ", ".join(e.lstrip(".").upper() for e in SUPPORTED_VIDEO_EXTENSIONS[:6])
    console.print(f"  [dim]Formats:[/dim] [bold]{exts}[/bold] [dim]+ 6 more[/dim]\n")

    files = discover_video_files(input_dir)
    render_dry_run_table(files)


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="trebek",
        description="🎙️  Trebek — High-fidelity Jeopardy! data extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  trebek                    Start continuous daemon\n"
            "  trebek --dry-run           Preview discovered files\n"
            "  trebek --once              Process queue then exit\n"
            "  trebek --input-dir ./vids  Override input directory\n"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan input directory and display discovered files without processing",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process all currently queued files then exit (no continuous polling)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override the input directory (default: from .env or 'input_videos')",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run the pipeline inside a GPU-enabled Docker container",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database analytics dashboard (no processing)",
    )
    parser.add_argument(
        "--nollm",
        action="store_true",
        help="GPU-only mode: transcribe episodes but skip LLM extraction, multimodal, and state machine",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Override input_dir if provided via CLI
    input_dir = args.input_dir or settings.input_dir

    if args.docker:
        import subprocess
        import sys

        cwd_abs = os.path.abspath(os.getcwd())
        input_abs = os.path.abspath(input_dir)

        cmd = [
            "docker",
            "run",
            "--rm",
            "-it",
            "--gpus",
            "all",
            "--shm-size=8gb",
            "-v",
            f"{cwd_abs}:/app",
        ]

        # If input_dir is outside CWD, mount it separately
        if not input_abs.startswith(cwd_abs):
            cmd.extend(["-v", f"{input_abs}:{input_abs}:ro"])

        # Pass .env file if it exists
        env_file = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_file):
            cmd.extend(["--env-file", env_file])
        if "GEMINI_API_KEY" in os.environ:
            cmd.extend(["-e", f"GEMINI_API_KEY={os.environ['GEMINI_API_KEY']}"])

        cmd.append("trebek:latest")

        if args.dry_run:
            cmd.append("--dry-run")
        if args.once:
            cmd.append("--once")
        if args.nollm:
            cmd.append("--nollm")
        if args.input_dir:
            cmd.extend(["--input-dir", input_abs])

        console.print("[dim cyan]Orchestrating Docker container...[/dim cyan]")
        try:
            sys.exit(subprocess.run(cmd).returncode)
        except FileNotFoundError:
            console.print("[bold red]❌ Docker is not installed or not in PATH.[/bold red]")
            sys.exit(1)

    if args.dry_run:
        handle_dry_run(input_dir)
        return

    if args.stats:
        render_stats_dashboard(settings.db_path)
        return

    # Import here to avoid loading heavy modules for --dry-run
    import asyncio
    from trebek.pipeline import run_pipeline

    mode = "once" if args.once else "daemon"

    asyncio.run(run_pipeline(mode=mode, input_dir_override=input_dir, nollm=args.nollm))


if __name__ == "__main__":
    main()
