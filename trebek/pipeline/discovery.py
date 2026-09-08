"""
Pipeline file discovery — scans input directories for video files and enriches
each entry with its current pipeline status from the database.

Used by both ``trebek scan`` (preview mode) and ingestion workers.
"""

import hashlib
import os
import re
import time
import asyncio
import sqlite3
import structlog
from typing import Any

from trebek.config import settings, SUPPORTED_VIDEO_EXTENSIONS, IGNORED_EXTENSIONS
from trebek.status import PipelineStatus as S

logger = structlog.get_logger()


def compute_file_fingerprint(filepath: str, sample_size: int = 65536) -> str:
    """Computes a fast, deterministic SHA-256 fingerprint for a video file.

    Hashes the file size (8 bytes big-endian) + the first sample_size bytes (container header)
    + the last sample_size bytes (tail metadata like MP4 moov atoms). For files smaller than
    or equal to 2 * sample_size (128 KB), hashes the entire file.

    Runs in <1ms even for multi-gigabyte files.
    """
    hasher = hashlib.sha256()
    try:
        size = os.path.getsize(filepath)
    except OSError:
        return ""

    hasher.update(size.to_bytes(8, byteorder="big"))

    if size <= sample_size * 2:
        try:
            with open(filepath, "rb") as f:
                hasher.update(f.read())
        except OSError:
            return ""
        return hasher.hexdigest()

    try:
        with open(filepath, "rb") as f:
            # First chunk (container header)
            hasher.update(f.read(sample_size))
            # Last chunk (tail metadata / trailing atoms)
            f.seek(size - sample_size)
            hasher.update(f.read(sample_size))
    except OSError:
        return ""

    return hasher.hexdigest()


def derive_episode_id(rel_path: str) -> str:
    """Derives a clean, normalized, unique episode_id from a relative file path.

    Transforms relative paths like:
        'Season 41/S41E01.mp4' -> 'Season_41_S41E01'
        'episode 1.mp4' -> 'episode_1'
        'Jeopardy! [1080p] 2024-01-15.mp4' -> 'Jeopardy_1080p_2024-01-15'
    """
    dirname, basename = os.path.split(rel_path)
    if basename.startswith(".") and "." in basename[1:]:
        basename_no_ext = os.path.splitext(basename)[0]
    elif basename.startswith("."):
        basename_no_ext = ""
    else:
        basename_no_ext = os.path.splitext(basename)[0]

    rel_no_ext = os.path.join(dirname, basename_no_ext) if dirname else basename_no_ext
    # Replace path separators (both POSIX and Windows)
    s = rel_no_ext.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    # Replace non-alphanumeric, non-hyphen, non-underscore characters with underscore
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", s)
    # Collapse multiple consecutive underscores into one
    s = re.sub(r"_+", "_", s)
    # Strip leading and trailing underscores
    s = s.strip("_")
    return s or "episode"


def is_candidate_video_file(filename: str) -> bool:
    """Check whether a filename looks like a valid, ready video file.

    Excludes hidden files (starting with .), temporary/download extensions
    (.part, .crdownload, .tmp, etc.), backup files ending with ~, and ensures
    the extension is in SUPPORTED_VIDEO_EXTENSIONS.
    """
    if filename.startswith(".") or filename.endswith("~"):
        return False
    lower_fname = filename.lower()
    for ign in IGNORED_EXTENSIONS:
        if lower_fname.endswith(ign):
            return False
    ext = os.path.splitext(lower_fname)[1]
    return ext in SUPPORTED_VIDEO_EXTENSIONS


def is_file_stable(filepath: str, min_age_seconds: float = 0.0) -> bool:
    """Checks if a file is non-empty, readable, and not actively being written.

    If min_age_seconds > 0, checks if the file was modified at least min_age_seconds ago.
    Also verifies the file can be opened for reading and has > 0 bytes.
    """
    try:
        stat = os.stat(filepath)
        if stat.st_size <= 0:
            return False
        if min_age_seconds > 0:
            age = time.time() - stat.st_mtime
            if age < min_age_seconds:
                return False
        with open(filepath, "rb") as f:
            f.read(1)
        return True
    except (OSError, PermissionError):
        return False


def scan_video_files(
    input_dir: str,
    check_stability: bool = False,
    min_age_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Recursively scans input_dir for all supported video files.

    Performs deterministic directory traversal, filters invalid and temporary files,
    resolves extension collisions, and derives normalized episode IDs.

    Returns a list of dicts with:
        - filename: relative path from input_dir
        - filepath: absolute path
        - format: lowercase extension (e.g. '.mp4')
        - size_bytes: integer file size
        - episode_id: normalized unique episode ID
        - mtime: modification timestamp
    """
    files: list[dict[str, Any]] = []
    if not os.path.exists(input_dir):
        return files

    seen_ids: dict[str, str] = {}  # episode_id -> rel_path for collision detection

    for dirpath, dirnames, filenames in os.walk(input_dir):
        # Sort in-place for deterministic traversal order
        dirnames.sort()
        for fname in sorted(filenames):
            if not is_candidate_video_file(fname):
                continue

            filepath = os.path.join(dirpath, fname)

            if check_stability and not is_file_stable(filepath, min_age_seconds=min_age_seconds):
                logger.debug("Skipping non-stable / in-progress file", file=filepath)
                continue

            try:
                size_bytes = os.path.getsize(filepath)
                if size_bytes <= 0:
                    continue
                mtime = os.path.getmtime(filepath)
            except OSError:
                continue

            rel = os.path.relpath(filepath, input_dir)
            ext = os.path.splitext(fname)[1].lower()
            base_id = derive_episode_id(rel)

            # Extension collision resolution: if another file in the same scan
            # produced the exact same episode_id (e.g., S41E01.mp4 vs S41E01.mkv)
            episode_id = base_id
            if episode_id in seen_ids and seen_ids[episode_id] != rel:
                disambiguated = f"{base_id}_{ext.lstrip('.')}"
                logger.warning(
                    "Extension collision detected — disambiguating episode ID",
                    base_id=base_id,
                    first_file=seen_ids[base_id],
                    second_file=rel,
                    assigned_id=disambiguated,
                )
                episode_id = disambiguated

            seen_ids[episode_id] = rel

            fingerprint = compute_file_fingerprint(filepath)

            files.append(
                {
                    "filename": rel,
                    "filepath": os.path.abspath(filepath),
                    "format": ext,
                    "size_bytes": size_bytes,
                    "episode_id": episode_id,
                    "mtime": mtime,
                    "fingerprint": fingerprint,
                }
            )

    return files


async def async_scan_video_files(
    input_dir: str,
    check_stability: bool = False,
    min_age_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Offloads synchronous filesystem traversal to a thread to protect the event loop."""
    return await asyncio.to_thread(
        scan_video_files,
        input_dir,
        check_stability=check_stability,
        min_age_seconds=min_age_seconds,
    )


# Status thresholds: episodes at or beyond this status have "passed" the stage
_STAGE_COMPLETED_STATUSES: dict[str, set[str]] = {
    "transcribe": {
        S.TRANSCRIPT_READY,
        S.CLEANED,
        S.SAVING,
        S.MULTIMODAL_PROCESSING,
        S.MULTIMODAL_DONE,
        S.VECTORIZING,
        S.COMPLETED,
    },
    "extract": {S.SAVING, S.MULTIMODAL_PROCESSING, S.MULTIMODAL_DONE, S.VECTORIZING, S.COMPLETED},
    "augment": {S.MULTIMODAL_DONE, S.VECTORIZING, S.COMPLETED},
    "verify": {S.COMPLETED},
}


def discover_video_files(input_dir: str, stage_filter: str | None = None) -> list[dict[str, Any]]:
    """Recursively scans input_dir for all supported video files.

    If stage_filter is provided, only returns files that still need work
    at that stage (i.e., haven't passed through it yet).
    """
    scanned = scan_video_files(input_dir)
    if not scanned:
        return []

    # Fetch full pipeline state for each episode
    episode_states: dict[str, tuple[str, int, str | None]] = {}  # episode_id → (status, retry_count, last_error)
    db_path = settings.db_path
    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute("SELECT episode_id, status, retry_count, last_error FROM pipeline_state").fetchall()
                for row in rows:
                    episode_states[row[0]] = (row[1], row[2] or 0, row[3])
        except sqlite3.OperationalError:
            pass  # DB may not have the table yet

    completed_statuses = _STAGE_COMPLETED_STATUSES.get(stage_filter or "", set()) if stage_filter else set()

    files: list[dict[str, Any]] = []
    for f in scanned:
        ep_id = f["episode_id"]
        if ep_id in episode_states:
            ep_status, retry_count, last_error = episode_states[ep_id]
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
                "filename": f["filename"],
                "filepath": f["filepath"],
                "format": f["format"],
                "size_bytes": f["size_bytes"],
                "status": pipeline_status,
                "retry_count": retry_count,
                "last_error": last_error,
                "episode_id": ep_id,
                "fingerprint": f.get("fingerprint", ""),
            }
        )

    return files
