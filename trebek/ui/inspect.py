"""
Episode inspection UI — Rich-rendered detail views for `trebek inspect <ep_id>`.
"""

import json
from typing import Any, Dict
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich import box

from trebek.ui.core import console
from trebek.ui.banner import render_startup_banner
from trebek.analysis.inspect import inspect_episode


def render_episode_inspection(data: Dict[str, Any]) -> Group:
    """Renders comprehensive episode details using Rich."""
    ep_id = data["episode_id"]
    status = data["status"]

    status_color = {
        "COMPLETED": "bold green",
        "FAILED": "bold red",
        "PENDING": "dim white",
        "TRANSCRIBING": "bold yellow",
        "TRANSCRIPT_READY": "cyan",
        "CLEANED": "bold blue",
        "SAVING": "magenta",
        "MULTIMODAL_PROCESSING": "bold magenta",
        "MULTIMODAL_DONE": "cyan",
        "VECTORIZING": "bold green",
    }.get(status, "white")

    # ── Header Metadata ──
    meta_table = Table(box=None, show_header=False, padding=(0, 2), expand=True)
    meta_table.add_column("Key", style="dim white", width=18)
    meta_table.add_column("Value", style="bold white")

    meta_table.add_row("Status", f"[{status_color}]{status}[/{status_color}]")
    if data.get("air_date"):
        meta_table.add_row("Air Date", f"[cyan]{data['air_date']}[/cyan]")
    if data.get("host_name"):
        meta_table.add_row("Host", f"[white]{data['host_name']}[/white]")
    if data.get("is_tournament"):
        meta_table.add_row("Format", "[yellow]Tournament Special[/yellow]")
    if data.get("source_filename"):
        meta_table.add_row("Source Video", f"[dim]{data['source_filename']}[/dim]")
    meta_table.add_row("Retries", f"[magenta]{data.get('retry_count', 0)}[/magenta]")
    meta_table.add_row("Updated", f"[dim]{data.get('updated_at', '')}[/dim]")

    header_panel = Panel(
        meta_table,
        title=f"[bold cyan]Episode Inspection — {ep_id}[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    renderables = [header_panel]

    # ── Contestants Table ──
    contestants = data.get("contestants", [])
    if contestants:
        c_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), expand=True)
        c_table.add_column("Podium", justify="center", width=8)
        c_table.add_column("Contestant Name", style="bold white", width=22)
        c_table.add_column("Occupation", style="dim", width=20)
        c_table.add_column("Coryat ($)", justify="right", style="cyan", width=12)
        c_table.add_column("Final Score ($)", justify="right", style="green", width=14)

        for c in contestants:
            coryat_str = f"${c['coryat_score']:,}" if c.get("coryat_score") is not None else "-"
            final_str = f"${c['final_score']:,}" if c.get("final_score") is not None else "-"
            c_table.add_row(
                str(c.get("podium_position") or "-"),
                c.get("name", "Unknown"),
                c.get("occupational_category") or "-",
                coryat_str,
                final_str,
            )

        c_panel = Panel(
            c_table,
            title="[bold yellow]Contestants & Performance[/bold yellow]",
            border_style="dim yellow",
            box=box.ROUNDED,
            padding=(0, 1),
        )
        renderables.append(c_panel)

    # ── Clue Statistics Table ──
    clues_sum = data.get("clues_summary", {})
    if clues_sum and clues_sum.get("total_clues", 0) > 0:
        clue_table = Table(box=None, show_header=False, padding=(0, 2), expand=True)
        clue_table.add_column("Metric", style="dim white", width=22)
        clue_table.add_column("Value", style="bold", justify="right")

        clue_table.add_row("Total Clues", f"[bold green]{clues_sum.get('total_clues')}[/bold green]")
        clue_table.add_row("Distinct Categories", str(clues_sum.get("distinct_categories", 0)))
        clue_table.add_row("Daily Doubles", f"[yellow]{clues_sum.get('daily_doubles', 0)}[/yellow]")
        clue_table.add_row("Triple Stumpers", f"[red]{clues_sum.get('triple_stumpers', 0)}[/red]")
        clue_table.add_row("Verified Clues", f"[cyan]{clues_sum.get('verified_clues', 0)}[/cyan]")

        clue_panel = Panel(
            clue_table,
            title="[bold green]Clue Inventory[/bold green]",
            border_style="dim green",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        renderables.append(clue_panel)

    # ── Telemetry & Cost Panel ──
    telemetry = data.get("telemetry")
    if telemetry:
        tel_table = Table(box=None, show_header=False, padding=(0, 2), expand=True)
        tel_table.add_column("Metric", style="dim white", width=22)
        tel_table.add_column("Value", style="bold", justify="right")

        tokens = telemetry.get("tokens", {})
        total_tokens = tokens.get("total", 0)
        cost_usd = telemetry.get("cost_usd", 0.0)

        tel_table.add_row("Gemini API Spend", f"[bold green]${cost_usd:.4f}[/bold green]")
        tel_table.add_row(
            "Tokens (Total)",
            f"{total_tokens:,} [dim]({tokens.get('input', 0):,} in · {tokens.get('output', 0):,} out)[/dim]",
        )
        if tokens.get("cached", 0) > 0:
            tel_table.add_row("Tokens Cached", f"[cyan]{tokens.get('cached', 0):,}[/cyan]")

        if telemetry.get("peak_vram_mb"):
            tel_table.add_row("Peak GPU VRAM", f"{telemetry.get('peak_vram_mb', 0):.0f} MB")
        if telemetry.get("avg_gpu_utilization_pct"):
            tel_table.add_row("Avg GPU Util", f"{telemetry.get('avg_gpu_utilization_pct', 0):.1f}%")

        latencies = telemetry.get("stage_latencies_ms", {})
        for stage_name, lat in latencies.items():
            if lat:
                tel_table.add_row(f"Latency ({stage_name})", f"{lat / 1000:.1f}s")

        tel_panel = Panel(
            tel_table,
            title="[bold magenta]Hardware & API Telemetry[/bold magenta]",
            border_style="dim magenta",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        renderables.append(tel_panel)

    # ── Quality Gate Warnings ──
    quality_warnings = data.get("quality_warnings", [])
    if quality_warnings:
        warn_table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        warn_table.add_column("Warning", style="yellow")
        for w in quality_warnings:
            warn_table.add_row(f"⚠️  {w}")

        warn_panel = Panel(
            warn_table,
            title=f"[bold yellow]Quality Gate Warnings ({len(quality_warnings)})[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        renderables.append(warn_panel)

    # ── Last Error Trace ──
    if data.get("last_error"):
        err_panel = Panel(
            f"[bold red]{data['last_error']}[/bold red]",
            title="[bold red]Last Error Details[/bold red]",
            border_style="red",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        renderables.append(err_panel)

    return Group(*renderables)


def handle_inspect_command(
    db_path: str,
    episode_id: str,
    output_dir: str,
    as_json: bool = False,
) -> None:
    """Executes the `trebek inspect` CLI subcommand."""
    data = inspect_episode(db_path, episode_id, output_dir=output_dir)
    if not data:
        console.print(f"\n  [bold red]Error:[/bold red] Episode [bold]{episode_id}[/bold] not found in database.\n")
        return

    if as_json:
        print(json.dumps(data, indent=2, default=str))
        return

    render_startup_banner(mode=f"inspect {episode_id}")
    console.print(render_episode_inspection(data))
