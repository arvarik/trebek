"""
Trebek CLI — Entry point for the pipeline daemon.

Usage:
    python src/cli.py                    # Start continuous daemon mode
    python src/cli.py --once             # Process current queue then exit
    python src/cli.py --dry-run          # Preview discovered files, no processing
    python src/cli.py --input-dir /path  # Override input directory
"""
import argparse
import os
import sqlite3

from config import settings, SUPPORTED_VIDEO_EXTENSIONS
from console import (
    console,
    render_startup_banner,
    render_dry_run_table,
    render_system_diagnostics,
)


def discover_video_files(input_dir: str) -> list[dict[str, object]]:
    """Scans input_dir for all supported video files and returns metadata."""
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

    for entry in sorted(os.scandir(input_dir), key=lambda e: e.name):
        if not entry.is_file():
            continue
        ext = os.path.splitext(entry.name)[1].lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            continue

        episode_id = os.path.splitext(entry.name)[0]
        status = "Already Queued" if episode_id in queued_ids else "New"

        files.append({
            "filename": entry.name,
            "filepath": entry.path,
            "format": ext,
            "size_bytes": entry.stat().st_size,
            "status": status,
        })

    return files


def handle_dry_run(input_dir: str) -> None:
    """Scans for video files and renders a preview table."""
    render_startup_banner(mode="dry-run")
    render_system_diagnostics(settings)

    console.print(f"\n  [dim]Scanning:[/dim] [bold]{os.path.abspath(input_dir)}[/bold]")
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
            "  python src/cli.py                    Start continuous daemon\n"
            "  python src/cli.py --dry-run           Preview discovered files\n"
            "  python src/cli.py --once              Process queue then exit\n"
            "  python src/cli.py --input-dir ./vids  Override input directory\n"
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Override input_dir if provided via CLI
    input_dir = args.input_dir or settings.input_dir

    if args.dry_run:
        handle_dry_run(input_dir)
        return

    # Import here to avoid loading heavy modules for --dry-run
    import asyncio
    from main import run_pipeline

    mode = "once" if args.once else "daemon"

    asyncio.run(run_pipeline(mode=mode, input_dir_override=input_dir))


if __name__ == "__main__":
    main()
