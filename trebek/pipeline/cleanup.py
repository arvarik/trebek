"""
Orphaned and obsolete file cleanup — safely identifies and purges
abandoned temporary files, unmapped audio chunks, and completed transcripts.
"""

import os
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Set

from trebek.status import PipelineStatus


@dataclass
class CleanupItem:
    filepath: str
    filename: str
    category: str
    size_bytes: int
    reason: str


def scan_cleanup_candidates(output_dir: str, db_path: str) -> List[CleanupItem]:
    """
    Scans output_dir for orphaned audio chunks, temporary files, and
    obsolete transcripts that are safe to purge.
    """
    if not os.path.exists(output_dir):
        return []

    # Query active / completed transcripts from DB if it exists
    active_transcripts: Set[str] = set()
    completed_episodes: Set[str] = set()
    all_known_transcripts: Set[str] = set()

    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                for row in conn.execute("SELECT episode_id, status, transcript_path FROM pipeline_state"):
                    ep_id, status, tx_path = row[0], row[1], row[2]
                    if tx_path:
                        all_known_transcripts.add(os.path.abspath(tx_path))
                        if status in (
                            PipelineStatus.PENDING,
                            PipelineStatus.TRANSCRIBING,
                            PipelineStatus.TRANSCRIPT_READY,
                            PipelineStatus.CLEANED,
                        ):
                            active_transcripts.add(os.path.abspath(tx_path))
                    if status == PipelineStatus.COMPLETED:
                        completed_episodes.add(ep_id)
        except sqlite3.OperationalError:
            pass

    candidates: List[CleanupItem] = []

    try:
        for entry in os.scandir(output_dir):
            if not entry.is_file():
                continue

            fname = entry.name
            fpath = os.path.abspath(entry.path)
            fsize = entry.stat().st_size

            # 1. Orphaned .wav audio chunks (crashed GPU workers or ffmpeg slices)
            if fname.endswith(".wav"):
                candidates.append(
                    CleanupItem(
                        filepath=fpath,
                        filename=fname,
                        category="Orphaned Audio (.wav)",
                        size_bytes=fsize,
                        reason="Transient audio chunk from previous worker run",
                    )
                )

            # 2. Temporary files (.tmp, .part)
            elif fname.endswith(".tmp") or ".tmp." in fname or fname.endswith(".part"):
                candidates.append(
                    CleanupItem(
                        filepath=fpath,
                        filename=fname,
                        category="Temporary File (.tmp)",
                        size_bytes=fsize,
                        reason="Interrupted or temporary processing file",
                    )
                )

            # 3. Interview audio slices
            elif fname.endswith("_interview_slice.mp3") or fname.endswith("_slice.mp3"):
                ep_prefix = fname.replace("_interview_slice.mp3", "").replace("_slice.mp3", "")
                reason = "Host interview slice no longer needed"
                if ep_prefix in completed_episodes:
                    reason = "Completed episode audio slice"
                candidates.append(
                    CleanupItem(
                        filepath=fpath,
                        filename=fname,
                        category="Interview Slice (.mp3)",
                        size_bytes=fsize,
                        reason=reason,
                    )
                )

            # 4. Obsolete / orphaned .json.gz transcripts
            elif fname.endswith(".json.gz"):
                if fpath not in active_transcripts:
                    if fpath not in all_known_transcripts:
                        reason = "Orphaned transcript not referenced in database"
                    else:
                        reason = "Transcript for completed / downstream-verified episode"
                    candidates.append(
                        CleanupItem(
                            filepath=fpath,
                            filename=fname,
                            category="Obsolete Transcript (.json.gz)",
                            size_bytes=fsize,
                            reason=reason,
                        )
                    )
    except OSError:
        pass

    # Sort by size descending
    candidates.sort(key=lambda x: x.size_bytes, reverse=True)
    return candidates


def execute_cleanup(candidates: List[CleanupItem]) -> Tuple[int, int]:
    """
    Deletes the candidate files. Returns (count_deleted, total_bytes_freed).
    """
    deleted_count = 0
    bytes_freed = 0

    for item in candidates:
        try:
            os.remove(item.filepath)
            deleted_count += 1
            bytes_freed += item.size_bytes
        except OSError:
            pass

    return deleted_count, bytes_freed
