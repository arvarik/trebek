"""
Centralized Rich console, progress bars, and formatted output for the Trebek pipeline.
All visual output flows through this module to ensure a consistent, premium experience.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional, Set

from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from trebek.ui.core import console


# Stage display names and their visual style
PIPELINE_STAGES: Dict[str, tuple[str, str]] = {
    "PENDING": ("⏳ Queued", "dim white"),
    "TRANSCRIBING": ("🎤 GPU Transcription", "yellow"),
    "TRANSCRIPT_READY": ("📝 Transcript Ready", "cyan"),
    "CLEANED": ("🧹 LLM Extraction", "blue"),
    "SAVING": ("💾 State Verification", "magenta"),
    "MULTIMODAL_PROCESSING": ("🔬 Multimodal Augmentation", "bright_magenta"),
    "MULTIMODAL_DONE": ("🔬 Multimodal Complete", "cyan"),
    "VECTORIZING": ("🧠 Relational Commit", "green"),
    "COMPLETED": ("✅ Done", "bold green"),
    "FAILED": ("❌ Failed", "bold red"),
}


def get_stage_display(status: str) -> str:
    """Returns a Rich-formatted stage display string."""
    label, style = PIPELINE_STAGES.get(status, (status, "white"))
    return f"[{style}]{label}[/{style}]"


# ── Conditional Column Classes for Mixed-Task Multi-Bar Displays ──────────────


class StageSpinnerColumn(SpinnerColumn):
    """Shows spinner for active work, pause icon when idle, checkmark when finished."""

    def render(self, task: Task) -> RenderableType:
        if task.fields.get("idle", False):
            return Text("⏸ ", style="dim")
        if task.finished:
            return Text("✓ ", style="bold green")
        return super().render(task)


class ConditionalBarColumn(BarColumn):
    """Renders a progress bar only if task.total is set (e.g. overall verified task)."""

    def render(self, task: Task) -> Any:
        if task.total is None:
            return Text("")
        return super().render(task)


class ConditionalMofNColumn(MofNCompleteColumn):
    """Renders M/N complete only if task.total is set."""

    def render(self, task: Task) -> Text:
        if task.total is None:
            return Text("")
        return super().render(task)


class ConditionalTimeRemainingColumn(TimeRemainingColumn):
    """Renders ETA only if task.total is set."""

    def render(self, task: Task) -> Text:
        if task.total is None:
            return Text("")
        return super().render(task)


class ConditionalSeparatorColumn(ProgressColumn):
    """Renders bullet separator only if task.total is set."""

    def render(self, task: Task) -> RenderableType:
        if task.total is None:
            return Text("")
        return Text("•", style="dim")


# ── Live Session Telemetry Ticker ─────────────────────────────────────────────


class SessionTelemetry:
    """Thread-safe session accumulator for live Gemini API spend and token usage."""

    def __init__(self) -> None:
        self.total_cost_usd: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.thinking_tokens: int = 0
        self.cached_tokens: int = 0
        self.api_calls: int = 0
        self._lock = threading.Lock()

    def record_usage(self, usage: Dict[str, Any]) -> None:
        with self._lock:
            self.total_cost_usd += float(usage.get("cost_usd", 0.0) or 0.0)
            self.input_tokens += int(usage.get("input_tokens", 0) or 0)
            self.output_tokens += int(usage.get("output_tokens", 0) or 0)
            self.thinking_tokens += int(usage.get("thinking_tokens", 0) or 0)
            self.cached_tokens += int(usage.get("cached_tokens", 0) or 0)
            self.api_calls += 1

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self.output_tokens

    @property
    def total_cached_tokens(self) -> int:
        return self.cached_tokens

    @property
    def total_calls(self) -> int:
        return self.api_calls

    def render_ticker(self) -> RenderableType:
        with self._lock:
            cost = self.total_cost_usd
            total_tok = self.input_tokens + self.output_tokens + self.thinking_tokens
            inp = self.input_tokens
            out = self.output_tokens
            cached = self.cached_tokens
            calls = self.api_calls

        def _fmt(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1_000_000:.2f}M"
            if n >= 1_000:
                return f"{n / 1_000:.1f}k"
            return str(n)

        grid = Table.grid(padding=(0, 2), expand=False)
        grid.add_column(style="bold cyan")
        grid.add_column()
        grid.add_column()
        grid.add_column()

        tok_details = f"[dim]({_fmt(inp)} in · {_fmt(out)} out"
        if cached > 0:
            tok_details += f" · {_fmt(cached)} cached"
        tok_details += ")[/dim]"

        grid.add_row(
            "⚡ Gemini API Live:",
            f"[bold green]${cost:.4f}[/bold green]",
            f"[cyan]{_fmt(total_tok)}[/cyan] tok {tok_details}",
            f"[dim]{calls} call{'s' if calls != 1 else ''}[/dim]",
        )
        return Panel(grid, box=box.HORIZONTALS, border_style="dim cyan", padding=(0, 1))


class TrebekPipelineProgress(Progress):
    """Extended Rich Progress that injects the live telemetry ticker above tasks."""

    def __init__(
        self, *columns: ProgressColumn, session_telemetry: Optional[SessionTelemetry] = None, **kwargs: Any
    ) -> None:
        self.session_telemetry = session_telemetry
        super().__init__(*columns, **kwargs)

    def get_renderables(self) -> Generator[RenderableType, None, None]:
        if self.session_telemetry is not None:
            yield self.session_telemetry.render_ticker()
        yield self.make_tasks_table(self.tasks)


# ── Pipeline Progress Coordinator ─────────────────────────────────────────────


@dataclass
class StageTaskSlots:
    verified: TaskID
    gpu: Optional[TaskID] = None
    llm: Dict[int, TaskID] = field(default_factory=dict)
    augment: Optional[TaskID] = None
    commit: Optional[TaskID] = None


class PipelineProgressCoordinator:
    """Coordinates multi-task Rich progress updates across concurrent workers."""

    def __init__(self, progress: TrebekPipelineProgress, telemetry: SessionTelemetry) -> None:
        self.progress = progress
        self.telemetry = telemetry
        self.slots: Optional[StageTaskSlots] = None

    def initialize_slots(self, active_stages: Set[str], llm_concurrency: int, total_episodes: int) -> None:
        gpu_id = None
        if "transcribe" in active_stages:
            gpu_id = self.progress.add_task("[bold yellow][GPU][/bold yellow] [dim]Idle[/dim]", total=None, idle=True)

        llm_ids: Dict[int, TaskID] = {}
        if "extract" in active_stages:
            for i in range(1, llm_concurrency + 1):
                label = f"[bold blue][LLM-{i}][/bold blue]" if llm_concurrency > 1 else "[bold blue][LLM][/bold blue]"
                llm_ids[i] = self.progress.add_task(f"{label} [dim]Idle[/dim]", total=None, idle=True)

        augment_id = None
        if "augment" in active_stages:
            augment_id = self.progress.add_task(
                "[bold magenta][Augment][/bold magenta] [dim]Idle[/dim]", total=None, idle=True
            )

        commit_id = None
        if "verify" in active_stages:
            commit_id = self.progress.add_task("[bold cyan][Commit][/bold cyan] [dim]Idle[/dim]", total=None, idle=True)

        verified_id = self.progress.add_task(
            "[bold green][Verified][/bold green] Completed episodes",
            total=total_episodes,
            completed=0,
            idle=False,
        )

        self.slots = StageTaskSlots(
            verified=verified_id,
            gpu=gpu_id,
            llm=llm_ids,
            augment=augment_id,
            commit=commit_id,
        )

    def update_gpu(self, episode_id: Optional[str]) -> None:
        if not self.slots or self.slots.gpu is None:
            return
        if episode_id:
            desc = f"[bold yellow][GPU][/bold yellow] Currently transcribing [white]{episode_id[:28]}[/white]"
            self.progress.update(self.slots.gpu, description=desc, idle=False)
        else:
            self.progress.update(
                self.slots.gpu, description="[bold yellow][GPU][/bold yellow] [dim]Idle (waiting)[/dim]", idle=True
            )

    def update_llm(self, worker_id: int, episode_id: Optional[str], pass_name: str = "") -> None:
        if not self.slots or worker_id not in self.slots.llm:
            return
        task_id = self.slots.llm[worker_id]
        label = (
            f"[bold blue][LLM-{worker_id}][/bold blue]" if len(self.slots.llm) > 1 else "[bold blue][LLM][/bold blue]"
        )
        if episode_id:
            action = f" {pass_name}" if pass_name else ""
            desc = f"{label}{action} for [white]{episode_id[:24]}[/white]"
            self.progress.update(task_id, description=desc, idle=False)
        else:
            self.progress.update(task_id, description=f"{label} [dim]Idle (waiting)[/dim]", idle=True)

    def update_augment(self, episode_id: Optional[str]) -> None:
        if not self.slots or self.slots.augment is None:
            return
        if episode_id:
            desc = f"[bold magenta][Augment][/bold magenta] Sniping visual frames for [white]{episode_id[:24]}[/white]"
            self.progress.update(self.slots.augment, description=desc, idle=False)
        else:
            self.progress.update(
                self.slots.augment,
                description="[bold magenta][Augment][/bold magenta] [dim]Idle (waiting)[/dim]",
                idle=True,
            )

    def update_commit(self, episode_id: Optional[str]) -> None:
        if not self.slots or self.slots.commit is None:
            return
        if episode_id:
            desc = f"[bold cyan][Commit][/bold cyan] Verifying state machine for [white]{episode_id[:24]}[/white]"
            self.progress.update(self.slots.commit, description=desc, idle=False)
        else:
            self.progress.update(
                self.slots.commit, description="[bold cyan][Commit][/bold cyan] [dim]Idle (waiting)[/dim]", idle=True
            )

    def advance_verified(self) -> None:
        if self.slots and self.slots.verified is not None:
            self.progress.advance(self.slots.verified)

    def update_total_episodes(self, total: int) -> None:
        if self.slots and self.slots.verified is not None:
            self.progress.update(self.slots.verified, total=total)


def create_pipeline_progress(session_telemetry: Optional[SessionTelemetry] = None) -> TrebekPipelineProgress:
    """Creates a Rich Progress instance configured for pipeline tracking."""
    return TrebekPipelineProgress(
        StageSpinnerColumn(spinner_name="dots2", style="cyan"),
        TextColumn("[bold white]{task.description}[/bold white]"),
        ConditionalBarColumn(bar_width=36, complete_style="magenta", finished_style="bold green"),
        ConditionalMofNColumn(),
        ConditionalSeparatorColumn(),
        TimeElapsedColumn(),
        ConditionalSeparatorColumn(),
        ConditionalTimeRemainingColumn(),
        console=console,
        transient=False,
        session_telemetry=session_telemetry,
    )
