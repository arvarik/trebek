from __future__ import annotations

import logging
import logging.handlers
import os
import queue
import threading
import time
from typing import Optional

import config
from audio_processor import extract_audio
from directory_watcher import DirectoryWatcher, DirectoryWatcherConfig
from file_manager import save_transcription
from transcriber import Transcriber


def _setup_logging() -> None:
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(config.LOGS_DIR, "transcriber.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
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


def main() -> None:
    # Ensure directories exist
    config.ensure_directories_exist()

    # Initialize logging
    _setup_logging()
    logging.info("Starting Jeopardy! Transcription Service")

    # Shared queue for file paths to process
    file_queue: "queue.Queue[str]" = queue.Queue()

    # Initialize components
    from config import WhisperCppConfig
    transcriber = Transcriber(WhisperCppConfig())
    watcher = DirectoryWatcher(file_queue, DirectoryWatcherConfig(config.RECORDINGS_DIR, recursive=True))

    # Start the watcher in a background thread. The observer manages its own threads,
    # but we call start/stop from the main thread for lifecycle control.
    watcher_thread = threading.Thread(target=watcher.start, name="WatcherThread", daemon=True)
    watcher_thread.start()
    logging.info("Watching for new recordings in: %s", config.RECORDINGS_DIR)

    try:
        while True:
            try:
                # Wait for next file path (blocking with timeout to allow graceful loop)
                video_path = file_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                logging.info("Processing started: %s", video_path)

                wav_path, duration_seconds = extract_audio(video_path)
                logging.info("Audio extracted: %s (%.1fs)", wav_path, duration_seconds)

                result = transcriber.transcribe(wav_path)
                logging.info("Transcription complete (%s)", result.get("language", "unknown"))

                output_path = save_transcription(video_path, result)
                logging.info("Saved transcription: %s", output_path)

            except Exception as e:
                logging.exception("Failed to process %s: %s", video_path, e)
            finally:
                # Clean up temporary audio file if present
                try:
                    if 'wav_path' in locals() and os.path.exists(wav_path):
                        os.remove(wav_path)
                        logging.info("Cleaned up temporary audio: %s", wav_path)
                except Exception:
                    logging.warning("Failed to remove temporary audio: %s", locals().get('wav_path'))

            # Mark the queue task as done
            file_queue.task_done()

    except KeyboardInterrupt:
        logging.info("Shutting down (KeyboardInterrupt)")
    finally:
        try:
            watcher.stop()
        except Exception:
            logging.warning("Watcher stop encountered an issue during shutdown.")


if __name__ == "__main__":
    main()


