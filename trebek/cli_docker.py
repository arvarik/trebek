"""
Docker container delegation for the ``trebek run --docker`` flag.

Constructs and executes a ``docker run`` command that passes through
GPU access, volume mounts, environment variables, and CLI flags to
the ``trebek:latest`` container image.
"""

import os
import sys
import argparse
import subprocess

from trebek.ui.core import console


def handle_docker(args: argparse.Namespace, input_dir: str) -> None:
    """Orchestrates a Docker container run, passing through CLI flags."""
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

    # Forward subcommand and flags
    cmd.append("run")
    if args.once:
        cmd.append("--once")
    if args.stage != "all":
        cmd.extend(["--stage", args.stage])
    if args.model != "pro":
        cmd.extend(["--model", args.model])
    if args.input_dir:
        cmd.extend(["--input-dir", input_abs])

    console.print("[dim cyan]Orchestrating Docker container...[/dim cyan]")
    try:
        sys.exit(subprocess.run(cmd).returncode)
    except FileNotFoundError:
        console.print("[bold red]❌ Docker is not installed or not in PATH.[/bold red]")
        sys.exit(1)
