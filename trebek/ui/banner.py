"""
Branded startup banner for the Trebek pipeline.

Renders the ASCII art logo and mode indicator at application startup.
System diagnostics are handled separately in ``trebek.ui.diagnostics``.
"""

from trebek.ui.core import console

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from rich import box


TREBEK_ASCII = r"""
 ╺┳╸┏━┓┏━╸┏┓ ┏━╸┃┏╸
  ┃ ┣┳┛┣╸ ┣┻┓┣╸ ┣┻┓
  ╹ ╹┗╸┗━╸┗━┛┗━╸╹ ╹
"""


def render_startup_banner(mode: str = "daemon") -> None:
    """Renders the Trebek branded startup panel."""
    mode_label = {
        "daemon": "[bold green]▶ Daemon Mode[/bold green] — Continuous polling",
        "once": "[bold yellow]▶ One-Shot Mode[/bold yellow] — Process queue then exit",
        "dry-run": "[bold blue]▶ Dry Run[/bold blue] — Preview only, no processing",
        "stats": "[bold magenta]▶ Stats[/bold magenta] — Database analytics dashboard",
    }.get(mode, f"[white]▶ {mode}[/white]")

    ascii_art = Text(TREBEK_ASCII.strip("\n"), style="bold cyan")
    tagline = Text("  High-fidelity J! data extraction pipeline\n", style="dim white")

    panel = Panel(
        Group(ascii_art, tagline),
        subtitle=f"  {mode_label}  ",
        border_style="cyan",
        box=box.DOUBLE_EDGE,
        padding=(1, 4),
    )
    console.print(panel)
