"""
Queue status analysis — real-time queue health queries without starting the pipeline.
"""

import os
import sqlite3
from typing import Any, Dict, List

from trebek.status import PipelineStatus


IN_FLIGHT_STATUSES = (
    PipelineStatus.TRANSCRIBING,
    PipelineStatus.CLEANED,
    PipelineStatus.SAVING,
    PipelineStatus.MULTIMODAL_PROCESSING,
    PipelineStatus.VECTORIZING,
)


def get_queue_status(db_path: str) -> Dict[str, Any]:
    """
    Queries real-time queue statistics from the SQLite database.
    Does not require spinning up the pipeline or worker queues.
    """
    if not os.path.exists(db_path):
        return {
            "database_found": False,
            "db_path": db_path,
            "total": 0,
            "status_counts": {},
            "in_flight": [],
            "recent_errors": [],
        }

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")

            # Per-status counts
            status_counts: Dict[str, int] = {}
            for row in conn.execute("SELECT status, COUNT(*) FROM pipeline_state GROUP BY status"):
                status_counts[row[0]] = row[1]

            total = sum(status_counts.values())

            # In-flight jobs
            in_flight_placeholders = ",".join(["?"] * len(IN_FLIGHT_STATUSES))
            in_flight_query = f"""
                SELECT episode_id, status, updated_at, retry_count
                FROM pipeline_state
                WHERE status IN ({in_flight_placeholders})
                ORDER BY updated_at DESC
            """
            in_flight_rows = conn.execute(in_flight_query, IN_FLIGHT_STATUSES).fetchall()
            in_flight: List[Dict[str, Any]] = [
                {
                    "episode_id": r[0],
                    "status": r[1],
                    "updated_at": r[2],
                    "retry_count": r[3],
                }
                for r in in_flight_rows
            ]

            # Recent errors
            error_query = """
                SELECT episode_id, last_error, updated_at, retry_count
                FROM pipeline_state
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT 5
            """
            error_rows = conn.execute(error_query, (PipelineStatus.FAILED,)).fetchall()
            recent_errors: List[Dict[str, Any]] = [
                {
                    "episode_id": r[0],
                    "last_error": r[1] or "",
                    "updated_at": r[2],
                    "retry_count": r[3],
                }
                for r in error_rows
            ]

            return {
                "database_found": True,
                "db_path": db_path,
                "total": total,
                "status_counts": status_counts,
                "in_flight": in_flight,
                "recent_errors": recent_errors,
            }
    except sqlite3.OperationalError as e:
        return {
            "database_found": True,
            "db_path": db_path,
            "error": str(e),
            "total": 0,
            "status_counts": {},
            "in_flight": [],
            "recent_errors": [],
        }
