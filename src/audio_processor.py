from __future__ import annotations

import os
import tempfile
from typing import Tuple

import ffmpeg  # type: ignore


def extract_audio(video_path: str) -> Tuple[str, float]:
    """Extract mono 16 kHz WAV audio from the given video file.

    Returns a tuple of (wav_path, duration_seconds).
    The caller is responsible for deleting the returned temporary file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Create a named temporary file path for the WAV output
    fd, wav_path = tempfile.mkstemp(prefix="audio_", suffix=".wav")
    os.close(fd)  # We only need the path; ffmpeg will write to it

    # Probe duration using ffmpeg to include in metadata
    try:
        probe = ffmpeg.probe(video_path)
        format_info = probe.get("format", {})
        duration_seconds = float(format_info.get("duration", 0.0))
    except Exception:
        duration_seconds = 0.0

    # Build and run the ffmpeg command: 16kHz mono PCM S16LE WAV
    (
        ffmpeg
        .input(video_path)
        .output(
            wav_path,
            acodec="pcm_s16le",
            ac=1,
            ar=16000,
            vn=None,
            loglevel="error",
        )
        .overwrite_output()
        .run()
    )

    return wav_path, duration_seconds


