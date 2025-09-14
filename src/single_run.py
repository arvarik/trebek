#!/usr/bin/env python3
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone

# Ensure local src imports resolve when executed directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from audio_processor import extract_audio
from transcriber import Transcriber
from config import WhisperCppConfig
import config
from file_manager import save_transcription


def _setup_logging() -> None:
    # Ensure log directory exists
    try:
        config.ensure_directories_exist()
    except Exception:
        os.makedirs(config.LOGS_DIR, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = os.path.join(config.LOGS_DIR, f"single_run.{ts}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    root.handlers.clear()
    root.addHandler(ch)
    root.addHandler(fh)
    logging.info("Log file: %s", log_path)


def main() -> None:
    _setup_logging()

    # Customize these variables for your run
    video_path = "/Volumes/Arvind SSD/Media/TV Shows/Jeopardy (1984)/Season 41/Jeopardy (1984) - S41E230 - Jeopardy .ts"
    cpp = WhisperCppConfig(
        binary_path="/Users/arvarik/Documents/github/whisper.cpp/build/bin/whisper-cli",
        model_path="/Users/arvarik/Documents/github/whisper.cpp/models/ggml-medium.en.bin",
        threads=max(1, os.cpu_count() or 4),
        language="en",
        print_progress=True,
    )

    logging.info("Starting single-run transcription")
    logging.info("Video: %s", video_path)
    logging.info("Model: %s", cpp.model_path)

    # Step 1: Extract audio
    logging.info("[1/3] Extracting audio...")
    wav_path, duration_seconds = extract_audio(video_path)
    logging.info("Audio extracted: %s (%.1fs)", wav_path, duration_seconds)

    # Step 2: Transcribe (progress is streamed by whisper.cpp when print_progress=True)
    logging.info("[2/3] Transcribing with whisper.cpp (progress will stream below)...")
    transcriber = Transcriber(cpp)
    result = transcriber.transcribe(wav_path)
    logging.info("Transcription complete. Language: %s, Duration: %.1fs", result.get("language", "unknown"), result.get("duration", 0.0))

    # Step 3: Save JSON
    logging.info("[3/3] Saving transcript JSON...")
    output_path = save_transcription(video_path, result)
    logging.info("Saved transcript to: %s", output_path)

    # Cleanup
    try:
        if os.path.exists(wav_path):
            os.remove(wav_path)
            logging.info("Removed temporary audio: %s", wav_path)
    except Exception:
        logging.warning("Failed to remove temporary audio: %s", wav_path)


if __name__ == "__main__":
    main()
