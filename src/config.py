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
RECORDINGS_DIR: str = "/Volumes/Arvind SSD/Media/TV Shows/Jeopardy (1984)"
TRANSCRIPTS_DIR: str = os.path.join(PROJECT_ROOT, "transcripts")
LOGS_DIR: str = os.path.join(PROJECT_ROOT, "logs")


def ensure_directories_exist() -> None:
    """Create required directories if they don't already exist."""
    for directory_path in (RECORDINGS_DIR, TRANSCRIPTS_DIR, LOGS_DIR):
        os.makedirs(directory_path, exist_ok=True)


@dataclass(frozen=True)
class WhisperCppConfig:
    """Configuration for the whisper.cpp backend.

    IMPORTANT: Set absolute paths on the target machine for reliability.
    See README for build and model download instructions.
    """

    # Path to whisper.cpp binary (e.g., "/Users/you/Projects/whisper.cpp/main" or build output)
    binary_path: str = "/Users/arvarik/Documents/github/whisper.cpp/build/bin/whisper-cli"
    # Path to the GGML/GGUF model file. Default to medium.en for better memory/performance balance.
    # Example filenames: ggml-medium.en.bin, ggml-medium.bin, ggml-large-v3-q5_0.bin (quantized)
    model_path: str = "/Users/arvarik/Documents/github/whisper.cpp/models/ggml-medium.en.bin"

    # Optional runtime settings
    language: str = "en"  # ISO 639-1 code; set "auto" to autodetect
    threads: int = max(1, os.cpu_count() or 4)
    enable_word_timestamps: bool = True
    print_progress: bool = True  # request progress output from whisper.cpp


