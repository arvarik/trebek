from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import Iterable

from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from watchdog.observers import Observer

import config


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".ts",
    ".m4v",
}


def _has_video_extension(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in VIDEO_EXTENSIONS


@dataclass
class DirectoryWatcherConfig:
    directory: str
    recursive: bool = False


class _EnqueueOnCreateHandler(FileSystemEventHandler):
    def __init__(self, file_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self._file_queue = file_queue

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        # Ignore directories; only handle files with video extensions
        if event.is_directory:
            return
        if _has_video_extension(event.src_path):
            self._file_queue.put(event.src_path)


class DirectoryWatcher:
    """Watches a directory and enqueues new video files as they appear."""

    def __init__(self, file_queue: "queue.Queue[str]", 
                 watch_config: DirectoryWatcherConfig | None = None) -> None:
        self._file_queue = file_queue
        self._observer = Observer()
        directory = (watch_config.directory if watch_config else config.RECORDINGS_DIR)
        recursive = (watch_config.recursive if watch_config else False)
        self._handler = _EnqueueOnCreateHandler(file_queue)
        self._observer.schedule(self._handler, directory, recursive=recursive)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._observer.start()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join(timeout=5)


