"""
Pipeline file discovery — scans input directories for video files and enriches
each entry with its current pipeline status from the database.

Used by both ``trebek scan`` (preview mode) and ingestion workers.
"""

import os
import sqlite3
from typing import Any

from trebek.config import settings, SUPPORTED_VIDEO_EXTENSIONS


# Status thresholds: episodes at or beyond this status have "passed" the stage
_STAGE_COMPLETED_STATUSES: dict[str, set[str]] = {
    "transcribe": {"TRANSCRIPT_READY", "CLEANED", "SAVING", "MULTIMODAL_PROCESSING", "MULTIMODAL_DONE", "VECTORIZING", "COMPLETED"},
    "extract": {"SAVING", "MULTIMODAL_PROCESSING", "MULTIMODAL_DONE", "VECTORIZING", "COMPLETED"},
    "augment": {"MULTIMODAL_DONE", "VECTORIZING", "COMPLETED"},
    "verify": {"COMPLETED"},
}


def discover_video_files(input_dir: str, stage_filter: str | None = None) -> list[dict[str, Any]]:
    """Recursively scans input_dir for all supported video files.

    If stage_filter is provided, only returns files that still need work
    at that stage (i.e., haven't passed through it yet).
    """
    files: list[dict[str, Any]] = []

    if not os.path.exists(input_dir):
        return files

    # Fetch full pipeline state for each episode
    episode_states: dict[str, tuple[str, int, str | None]] = {}  # episode_id → (status, retry_count, last_error)
    db_path = settings.db_path
    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT episode_id, status, retry_count, last_error FROM pipeline_state"
                ).fetchall()
                for row in rows:
                    episode_states[row[0]] = (row[1], row[2] or 0, row[3])
        except sqlite3.OperationalError:
            pass  # DB may not have the table yet

    completed_statuses = _STAGE_COMPLETED_STATUSES.get(stage_filter or "", set()) if stage_filter else set()

    for dirpath, _dirnames, filenames in os.walk(input_dir):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_VIDEO_EXTENSIONS:
                continue

            filepath = os.path.join(dirpath, fname)
            rel = os.path.relpath(filepath, input_dir)
            episode_id = os.path.splitext(rel)[0].replace(os.sep, "_").replace(" ", "_")

            if episode_id in episode_states:
                ep_status, retry_count, last_error = episode_states[episode_id]
                pipeline_status = ep_status
            else:
                pipeline_status = "New"
                retry_count = 0
                last_error = None

            # Stage filtering: skip files that have already passed through the target stage
            if stage_filter and pipeline_status in completed_statuses:
                continue

            files.append(
                {
                    "filename": rel,
                    "filepath": filepath,
                    "format": ext,
                    "size_bytes": os.path.getsize(filepath),
                    "status": pipeline_status,
                    "retry_count": retry_count,
                    "last_error": last_error,
                }
            )

    return files
