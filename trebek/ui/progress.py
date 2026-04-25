"""
Centralized Rich console, progress bars, and formatted output for the Trebek pipeline.
All visual output flows through this module to ensure a consistent, premium experience.
"""

from trebek.ui.core import console

from typing import Dict

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


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
