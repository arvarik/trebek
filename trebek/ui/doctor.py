"""
Trebek Doctor — Pre-flight environment diagnostics and system readiness verification.

Validates:
1. Python environment (version >= 3.11, architecture)
2. External binaries (ffmpeg, ffprobe)
3. Hardware acceleration & ML runtimes (CUDA, MPS, CPU, PyTorch, WhisperX, PyAnnote)
4. API keys & authentication (GEMINI_API_KEY, HF_TOKEN, optional API ping)
5. Database & POSIX locks (SQLite version >= 3.35, WAL mode advisory locks, disk space)
6. Pipeline directory permissions (input_dir, output_dir)

Provides formatted Rich output with actionable remediation suggestions,
and supports structured JSON output via `--json`.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trebek.config import Settings
from trebek.gpu.hardware import detect_hardware
from trebek.ui.core import console


CheckStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass
class DiagnosticCheck:
    category: str
    component: str
    status: CheckStatus
    detail: str
    remediation: str = ""


@dataclass
class DoctorReport:
    success: bool
    summary: dict[str, int]
    checks: list[DiagnosticCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
        }


def _check_binary(name: str) -> tuple[bool, str]:
    """Checks if a binary exists in PATH and retrieves its first version line."""
    path = shutil.which(name)
    if not path:
        return False, "not found in PATH"
    try:
        flag = "-version" if name == "ffmpeg" else "--version"
        res = subprocess.run([name, flag], capture_output=True, text=True, timeout=5)
        first_line = res.stdout.strip().split("\n")[0] if res.stdout else ""
        return True, first_line[:65] if first_line else f"found at {path}"
    except Exception:
        return True, f"found at {path}"


def _test_sqlite_wal_locks(db_dir: str) -> tuple[bool, str]:
    """Verifies that the target directory supports SQLite WAL mode POSIX advisory locks."""
    test_db_path = os.path.join(db_dir, ".trebek_doctor_wal_test.db")
    try:
        conn = sqlite3.connect(test_db_path, timeout=3.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("CREATE TABLE IF NOT EXISTS _test (id INTEGER PRIMARY KEY);")
        conn.execute("INSERT INTO _test DEFAULT VALUES;")
        conn.commit()
        conn.close()
        # Clean up test database and its aux files
        for ext in ("", "-wal", "-shm"):
            p = test_db_path + ext
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return True, "WAL mode & POSIX advisory locks verified"
    except Exception as e:
        return False, f"WAL lock test failed: {e}"


def run_diagnostics(settings: Settings, check_api: bool = False) -> DoctorReport:
    """Runs all environment and preflight diagnostic checks and returns a DoctorReport."""
    checks: list[DiagnosticCheck] = []

    # ── 1. Python Environment ──
    py_ver = f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    if sys.version_info >= (3, 11):
        checks.append(
            DiagnosticCheck(
                category="System & Runtime",
                component="Python Version",
                status="PASS",
                detail=f"{py_ver} ({platform.system()} {platform.machine()})",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                category="System & Runtime",
                component="Python Version",
                status="FAIL",
                detail=f"{py_ver} (Python >= 3.11 required)",
                remediation="Upgrade Python to 3.11 or higher.",
            )
        )

    # ── 2. External Binaries ──
    ffmpeg_ok, ffmpeg_detail = _check_binary("ffmpeg")
    if ffmpeg_ok:
        checks.append(
            DiagnosticCheck(
                category="External Binaries",
                component="FFmpeg",
                status="PASS",
                detail=ffmpeg_detail,
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                category="External Binaries",
                component="FFmpeg",
                status="FAIL",
                detail="ffmpeg not found in PATH",
                remediation="Install ffmpeg (brew install ffmpeg, apt-get install ffmpeg, or use --docker).",
            )
        )

    ffprobe_ok, ffprobe_detail = _check_binary("ffprobe")
    if ffprobe_ok:
        checks.append(
            DiagnosticCheck(
                category="External Binaries",
                component="FFprobe",
                status="PASS",
                detail=ffprobe_detail,
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                category="External Binaries",
                component="FFprobe",
                status="WARN",
                detail="ffprobe not found in PATH",
                remediation="Install ffprobe for enhanced media stream inspection.",
            )
        )

    # ── 3. Hardware & ML Runtimes ──
    hw = detect_hardware()
    if hw.is_cuda:
        vram_info = f" ({hw.vram_gb:.1f} GB VRAM)" if hw.vram_gb else ""
        cuda_ver = f", CUDA {hw.cuda_version}" if hw.cuda_version else ""
        checks.append(
            DiagnosticCheck(
                category="Hardware & ML",
                component="GPU Acceleration",
                status="PASS",
                detail=f"{hw.device_name}{vram_info}{cuda_ver}",
            )
        )
    elif hw.is_mps:
        checks.append(
            DiagnosticCheck(
                category="Hardware & ML",
                component="GPU Acceleration",
                status="WARN",
                detail=f"{hw.device_name} detected (WhisperX uses CPU fallback)",
                remediation="CTranslate2 lacks MPS support. Run with --allow-cpu or use --docker on a Linux/CUDA machine.",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                category="Hardware & ML",
                component="GPU Acceleration",
                status="WARN",
                detail="CPU only (no CUDA GPU detected)",
                remediation="Transcription will run on CPU (--allow-cpu). Use an NVIDIA GPU for 10x-20x speedup.",
            )
        )

    # PyTorch importability
    try:
        import torch

        checks.append(
            DiagnosticCheck(
                category="Hardware & ML",
                component="PyTorch",
                status="PASS",
                detail=f"v{torch.__version__}",
            )
        )
    except ImportError:
        checks.append(
            DiagnosticCheck(
                category="Hardware & ML",
                component="PyTorch",
                status="WARN",
                detail="PyTorch is not installed in current Python environment",
                remediation="Install PyTorch via 'pip install torch' or run inside Docker.",
            )
        )

    # WhisperX importability
    try:
        import whisperx  # noqa: F401  # type: ignore

        checks.append(
            DiagnosticCheck(
                category="Hardware & ML",
                component="WhisperX",
                status="PASS",
                detail="Installed and importable",
            )
        )
    except ImportError:
        checks.append(
            DiagnosticCheck(
                category="Hardware & ML",
                component="WhisperX",
                status="WARN",
                detail="whisperx package is not installed",
                remediation="Install whisperx or use hybrid Docker execution: trebek run --docker",
            )
        )

    # PyAnnote importability
    try:
        import pyannote.audio  # noqa: F401  # type: ignore

        checks.append(
            DiagnosticCheck(
                category="Hardware & ML",
                component="PyAnnote Audio",
                status="PASS",
                detail="Installed and importable",
            )
        )
    except ImportError:
        checks.append(
            DiagnosticCheck(
                category="Hardware & ML",
                component="PyAnnote Audio",
                status="WARN",
                detail="pyannote.audio package is not installed",
                remediation="Install pyannote.audio for speaker diarization or use --docker.",
            )
        )

    # ── 4. API Keys & Authentication ──
    api_key = os.environ.get("GEMINI_API_KEY", settings.gemini_api_key)
    if api_key:
        masked_key = api_key[:4] + "•" * 12 + api_key[-4:] if len(api_key) > 8 else "•" * len(api_key)
        if check_api:
            try:
                from google import genai
                from trebek.config import MODEL_FLASH

                client = genai.Client(api_key=api_key)
                client.models.get(model=MODEL_FLASH)
                checks.append(
                    DiagnosticCheck(
                        category="API & Authentication",
                        component="Gemini API Key",
                        status="PASS",
                        detail=f"{masked_key} (API ping succeeded)",
                    )
                )
            except Exception as e:
                checks.append(
                    DiagnosticCheck(
                        category="API & Authentication",
                        component="Gemini API Key",
                        status="FAIL",
                        detail=f"{masked_key} (Ping failed: {str(e)[:80]})",
                        remediation="Verify GEMINI_API_KEY in .env or at https://aistudio.google.com/apikey",
                    )
                )
        else:
            checks.append(
                DiagnosticCheck(
                    category="API & Authentication",
                    component="Gemini API Key",
                    status="PASS",
                    detail=f"{masked_key} (configured)",
                )
            )
    else:
        if getattr(settings, "mock_llm", False):
            checks.append(
                DiagnosticCheck(
                    category="API & Authentication",
                    component="Gemini API Key",
                    status="PASS",
                    detail="Not set (mock mode enabled via --mock-llm)",
                )
            )
        else:
            checks.append(
                DiagnosticCheck(
                    category="API & Authentication",
                    component="Gemini API Key",
                    status="FAIL",
                    detail="Not configured in .env or environment",
                    remediation="Set GEMINI_API_KEY in .env (get free key at https://aistudio.google.com/apikey) or use --mock-llm.",
                )
            )

    hf_token = os.environ.get("HF_TOKEN", "") or getattr(settings, "hf_token", "")
    if hf_token:
        masked_hf = hf_token[:4] + "•" * 8 + hf_token[-4:] if len(hf_token) > 8 else "•" * len(hf_token)
        if check_api:
            try:
                import urllib.request

                req = urllib.request.Request(
                    "https://huggingface.co/api/models/pyannote/speaker-diarization-3.1",
                    headers={"Authorization": f"Bearer {hf_token}"},
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        checks.append(
                            DiagnosticCheck(
                                category="API & Authentication",
                                component="Hugging Face Token",
                                status="PASS",
                                detail=f"{masked_hf} (pyannote access verified)",
                            )
                        )
                    else:
                        checks.append(
                            DiagnosticCheck(
                                category="API & Authentication",
                                component="Hugging Face Token",
                                status="WARN",
                                detail=f"{masked_hf} (HTTP {resp.status})",
                                remediation="Accept licenses at https://huggingface.co/pyannote/speaker-diarization-3.1",
                            )
                        )
            except Exception as e:
                checks.append(
                    DiagnosticCheck(
                        category="API & Authentication",
                        component="Hugging Face Token",
                        status="WARN",
                        detail=f"{masked_hf} (gated access check failed: {e})",
                        remediation="Accept licenses at https://huggingface.co/pyannote/speaker-diarization-3.1",
                    )
                )
        else:
            checks.append(
                DiagnosticCheck(
                    category="API & Authentication",
                    component="Hugging Face Token",
                    status="PASS",
                    detail=f"{masked_hf} (configured)",
                )
            )
    else:
        checks.append(
            DiagnosticCheck(
                category="API & Authentication",
                component="Hugging Face Token",
                status="WARN",
                detail="Not configured (HF_TOKEN is unset)",
                remediation="Set HF_TOKEN in .env to enable speaker identification (pyannote gated model).",
            )
        )

    # ── 5. Database & Storage ──
    sqlite_ver = sqlite3.sqlite_version
    sqlite_tuple = tuple(int(x) for x in sqlite_ver.split(".")[:3])
    if sqlite_tuple >= (3, 35, 0):
        checks.append(
            DiagnosticCheck(
                category="Storage & Database",
                component="SQLite Version",
                status="PASS",
                detail=f"{sqlite_ver} (>= 3.35 required)",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                category="Storage & Database",
                component="SQLite Version",
                status="FAIL",
                detail=f"{sqlite_ver} is too old (< 3.35)",
                remediation="Upgrade SQLite to >= 3.35 for RETURNING clause and FTS5 support.",
            )
        )

    db_path = settings.db_path
    if os.path.isdir(db_path):
        checks.append(
            DiagnosticCheck(
                category="Storage & Database",
                component="Database Path",
                status="FAIL",
                detail=f"'{db_path}' is a directory, not a database file",
                remediation="Remove the directory and specify a file path (e.g. ./data:/app/data with DB_PATH=/app/data/trebek.db).",
            )
        )
    else:
        db_dir = os.path.dirname(os.path.abspath(db_path)) or "."
        try:
            os.makedirs(db_dir, exist_ok=True)
            wal_ok, wal_detail = _test_sqlite_wal_locks(db_dir)
            checks.append(
                DiagnosticCheck(
                    category="Storage & Database",
                    component="SQLite WAL Locks",
                    status="PASS" if wal_ok else "FAIL",
                    detail=wal_detail,
                    remediation=""
                    if wal_ok
                    else "Ensure database path is on a local filesystem (ext4, NTFS, APFS) supporting POSIX locks.",
                )
            )
        except Exception as e:
            checks.append(
                DiagnosticCheck(
                    category="Storage & Database",
                    component="SQLite WAL Locks",
                    status="FAIL",
                    detail=f"Cannot create or access database directory '{db_dir}': {e}",
                    remediation=f"Ensure write permissions for database directory: {db_dir}",
                )
            )

        try:
            usage = shutil.disk_usage(db_dir)
            free_gb = usage.free / (1024**3)
            if free_gb >= 10.0:
                disk_status: CheckStatus = "PASS"
                remed = ""
            elif free_gb >= 2.0:
                disk_status = "WARN"
                remed = "Disk space is somewhat low (<10 GB free). Video processing requires buffer space."
            else:
                disk_status = "FAIL"
                remed = "Critically low disk space (<2 GB free). Free up disk space before processing videos."
            checks.append(
                DiagnosticCheck(
                    category="Storage & Database",
                    component="Disk Space",
                    status=disk_status,
                    detail=f"{free_gb:.1f} GB free on {db_dir}",
                    remediation=remed,
                )
            )
        except Exception as e:
            checks.append(
                DiagnosticCheck(
                    category="Storage & Database",
                    component="Disk Space",
                    status="WARN",
                    detail=f"Could not determine disk usage: {e}",
                )
            )

    # ── 6. Pipeline Directories ──
    in_dir = settings.input_dir
    if os.path.exists(in_dir):
        if os.path.isdir(in_dir):
            try:
                from trebek.pipeline.discovery import discover_video_files

                vids = discover_video_files(in_dir)
                checks.append(
                    DiagnosticCheck(
                        category="Pipeline Directories",
                        component="Input Directory",
                        status="PASS",
                        detail=f"'{in_dir}' exists ({len(vids)} video file(s) found)",
                    )
                )
            except Exception:
                checks.append(
                    DiagnosticCheck(
                        category="Pipeline Directories",
                        component="Input Directory",
                        status="PASS",
                        detail=f"'{in_dir}' exists",
                    )
                )
        else:
            checks.append(
                DiagnosticCheck(
                    category="Pipeline Directories",
                    component="Input Directory",
                    status="FAIL",
                    detail=f"'{in_dir}' is a file, expected a directory",
                    remediation=f"Remove '{in_dir}' and create a directory for input videos.",
                )
            )
    else:
        checks.append(
            DiagnosticCheck(
                category="Pipeline Directories",
                component="Input Directory",
                status="WARN",
                detail=f"'{in_dir}' does not exist yet",
                remediation=f"Create the directory: mkdir -p {in_dir}",
            )
        )

    out_dir = settings.output_dir
    try:
        os.makedirs(out_dir, exist_ok=True)
        checks.append(
            DiagnosticCheck(
                category="Pipeline Directories",
                component="Output Directory",
                status="PASS",
                detail=f"'{out_dir}' is writable",
            )
        )
    except Exception as e:
        checks.append(
            DiagnosticCheck(
                category="Pipeline Directories",
                component="Output Directory",
                status="FAIL",
                detail=f"Cannot write to '{out_dir}': {e}",
                remediation=f"Ensure write permissions for output directory: {out_dir}",
            )
        )

    # ── Summary counts ──
    pass_count = sum(1 for c in checks if c.status == "PASS")
    warn_count = sum(1 for c in checks if c.status == "WARN")
    fail_count = sum(1 for c in checks if c.status == "FAIL")

    return DoctorReport(
        success=(fail_count == 0),
        summary={"pass": pass_count, "warn": warn_count, "fail": fail_count, "total": len(checks)},
        checks=checks,
    )


def render_doctor_results(report: DoctorReport, as_json: bool = False) -> int:
    """Renders the doctor report and returns an appropriate exit code (0 = success, 1 = failure)."""
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.success else 1

    from trebek import __version__
    from trebek.ui.banner import TREBEK_ASCII

    art = Text(TREBEK_ASCII.strip(), style="bold cyan")
    tag = Text(f"  v{__version__}  •  Pre-flight Environment Diagnostics", style="dim white")
    console.print(
        Panel(
            Text.assemble(art, "\n", tag),
            border_style="dim cyan",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )

    # Group by category
    current_cat = ""
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim cyan",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Status", width=8, justify="center")
    table.add_column("Category", width=22, style="dim white")
    table.add_column("Component", width=22, style="bold white")
    table.add_column("Details", style="white")

    for c in report.checks:
        if c.status == "PASS":
            status_text = "[bold green]✓ PASS[/bold green]"
        elif c.status == "WARN":
            status_text = "[bold yellow]! WARN[/bold yellow]"
        else:
            status_text = "[bold red]✗ FAIL[/bold red]"

        cat_display = c.category if c.category != current_cat else ""
        current_cat = c.category
        table.add_row(status_text, cat_display, c.component, c.detail)

    console.print(table)

    # Render Remediation panel if warnings or failures exist
    issues = [c for c in report.checks if c.status in ("WARN", "FAIL") and c.remediation]
    if issues:
        rem_lines: list[Text] = []
        for issue in issues:
            icon = "✗" if issue.status == "FAIL" else "!"
            style = "bold red" if issue.status == "FAIL" else "bold yellow"
            rem_lines.append(
                Text.assemble(
                    Text(f" • [{icon}] {issue.component}: ", style=style),
                    Text(issue.remediation, style="white"),
                )
            )

        rem_group = Text("\n").join(rem_lines)
        console.print(
            Panel(
                rem_group,
                title="[bold yellow]Actionable Recommendations[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    # Final summary banner
    summary = report.summary
    if report.success:
        if summary["warn"] == 0:
            console.print(
                f"\n  [bold green]✔ All {summary['pass']} pre-flight checks passed![/bold green] "
                "System is ready to run the extraction pipeline.\n"
            )
        else:
            console.print(
                f"\n  [bold green]✔ Pre-flight passed with {summary['warn']} warning(s).[/bold green] "
                "Pipeline can run, but check warnings above for optimal throughput.\n"
            )
        return 0
    else:
        console.print(
            f"\n  [bold red]⛔ Pre-flight failed with {summary['fail']} critical blocker(s).[/bold red] "
            "Please resolve the blockers above before running the pipeline.\n"
        )
        return 1
