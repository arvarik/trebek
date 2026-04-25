import sqlite3
from typing import Dict
from rich.panel import Panel
from rich.table import Table
from rich import box
from trebek.ui.progress import PIPELINE_STAGES


def generate_health_panel(conn: sqlite3.Connection) -> Panel:
    health_table = Table(box=None, show_header=False, padding=(0, 2))
    health_table.add_column("Metric", style="dim white", width=18)
    health_table.add_column("Value", style="bold", justify="right", width=10)

    status_counts: Dict[str, int] = {}
    try:
        for row in conn.execute("SELECT status, COUNT(*) FROM pipeline_state GROUP BY status"):
            status_counts[row[0]] = row[1]
    except sqlite3.OperationalError:
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

    return Panel(
        health_table,
        title="[bold]Pipeline Health[/bold]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=True,
    )


def generate_telemetry_panel(conn: sqlite3.Connection) -> Panel:
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
    except sqlite3.OperationalError:
        telemetry_table.add_row("[dim]Telemetry table not found[/dim]", "")

    return Panel(
        telemetry_table,
        title="[bold]Performance[/bold]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=True,
    )


def generate_timing_panel(conn: sqlite3.Connection) -> Panel:
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
    except sqlite3.OperationalError:
        pass

    return Panel(
        timing_table,
        title="[bold]Stage Timing Breakdown[/bold]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 1),
    )


def generate_recent_panel(conn: sqlite3.Connection) -> Panel:
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
    except sqlite3.OperationalError:
        pass

    return Panel(
        recent_table,
        title="[bold]Recent Episodes[/bold]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 1),
    )
