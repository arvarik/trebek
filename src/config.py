"""Application configuration.

This module defines configurable paths used by the application. For development,
the defaults are set relative to the project root. When deploying to another
machine (e.g., a Mac mini), CHANGE THESE to absolute paths that match the
target environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _project_root() -> str:
    """Return the absolute filesystem path to the project root directory.

    Computed relative to this file, so it works out-of-the-box during development.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


PROJECT_ROOT: str = _project_root()

# Default development paths (relative to project root). For deployment, set these
# to ABSOLUTE paths on the target machine. See README for examples.
RECORDINGS_DIR: str = os.path.join(PROJECT_ROOT, "recordings")
TRANSCRIPTS_DIR: str = os.path.join(PROJECT_ROOT, "transcripts")
LOGS_DIR: str = os.path.join(PROJECT_ROOT, "logs")


def ensure_directories_exist() -> None:
    """Create required directories if they don't already exist."""
    for directory_path in (RECORDINGS_DIR, TRANSCRIPTS_DIR, LOGS_DIR):
        os.makedirs(directory_path, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration values for the transcription model."""

    # Whisper model size or local directory path
    model_size: str = "large-v3"
    # Attempt to use Apple GPU via Metal when available; fall back to CPU.
    preferred_device: str = "metal"  # options: "auto", "metal", "cpu"
    # Compute type for inference; float16 on GPU/Metal, int8_float16 or int8 on CPU.
    compute_type_metal: str = "float16"
    compute_type_cpu: str = "int8_float16"



