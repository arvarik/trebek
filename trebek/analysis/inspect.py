"""
Episode inspection analysis — gathers comprehensive episode metadata,
contestant performance, telemetry, clue statistics, and quality warnings.
"""

import os
import sqlite3
from typing import Any, Dict, List, Optional
from pathlib import Path


def inspect_episode(
    db_path: str,
    episode_id: str,
    output_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Inspects a single episode across pipeline_state, relational tables,
    job_telemetry, and cached intermediate JSONs.
    Returns None if the episode is not found.
    """
    if not os.path.exists(db_path):
        return None

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")

            # 1. pipeline_state
            ps_row = conn.execute(
                """
                SELECT episode_id, status, source_filename, transcript_path,
                       retry_count, last_error, created_at, updated_at
                FROM pipeline_state
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()

            if not ps_row:
                return None

            data: Dict[str, Any] = {
                "episode_id": ps_row[0],
                "status": ps_row[1],
                "source_filename": ps_row[2],
                "transcript_path": ps_row[3],
                "retry_count": ps_row[4],
                "last_error": ps_row[5],
                "created_at": ps_row[6],
                "updated_at": ps_row[7],
            }

            # 2. episodes table
            ep_row = conn.execute(
                "SELECT air_date, host_name, is_tournament FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if ep_row:
                data["air_date"] = ep_row[0]
                data["host_name"] = ep_row[1]
                data["is_tournament"] = bool(ep_row[2])
            else:
                data["air_date"] = None
                data["host_name"] = None
                data["is_tournament"] = False

            # 3. contestants and performances
            perf_rows = conn.execute(
                """
                SELECT c.name, c.occupational_category, c.is_returning_champion,
                       p.podium_position, p.coryat_score, p.final_score, p.forrest_bounce_index
                FROM episode_performances p
                JOIN contestants c ON p.contestant_id = c.contestant_id
                WHERE p.episode_id = ?
                ORDER BY p.podium_position ASC
                """,
                (episode_id,),
            ).fetchall()

            contestants: List[Dict[str, Any]] = []
            for r in perf_rows:
                contestants.append(
                    {
                        "name": r[0],
                        "occupational_category": r[1],
                        "is_returning_champion": bool(r[2]),
                        "podium_position": r[3],
                        "coryat_score": r[4],
                        "final_score": r[5],
                        "forrest_bounce_index": r[6],
                    }
                )
            data["contestants"] = contestants

            # 4. clues summary
            clue_row = conn.execute(
                """
                SELECT COUNT(*),
                       SUM(CASE WHEN is_daily_double = 1 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN is_triple_stumper = 1 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN is_verified = 1 THEN 1 ELSE 0 END),
                       COUNT(DISTINCT category)
                FROM clues
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()

            data["clues_summary"] = {
                "total_clues": clue_row[0] if clue_row else 0,
                "daily_doubles": clue_row[1] if clue_row and clue_row[1] else 0,
                "triple_stumpers": clue_row[2] if clue_row and clue_row[2] else 0,
                "verified_clues": clue_row[3] if clue_row and clue_row[3] else 0,
                "distinct_categories": clue_row[4] if clue_row else 0,
            }

            # 5. job_telemetry
            tel_row = conn.execute(
                """
                SELECT peak_vram_mb, avg_gpu_utilization_pct,
                       stage_ingestion_ms, stage_gpu_extraction_ms,
                       stage_commercial_filtering_ms, stage_structured_extraction_ms,
                       stage_multimodal_ms, stage_vectorization_ms,
                       gemini_total_input_tokens, gemini_total_output_tokens,
                       gemini_total_cached_tokens, gemini_total_cost_usd,
                       gemini_api_latency_ms, pydantic_retry_count
                FROM job_telemetry
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()

            if tel_row:
                data["telemetry"] = {
                    "peak_vram_mb": tel_row[0],
                    "avg_gpu_utilization_pct": tel_row[1],
                    "stage_latencies_ms": {
                        "ingestion": tel_row[2],
                        "gpu_extraction": tel_row[3],
                        "commercial_filtering": tel_row[4],
                        "structured_extraction": tel_row[5],
                        "multimodal": tel_row[6],
                        "vectorization": tel_row[7],
                    },
                    "tokens": {
                        "input": tel_row[8] or 0,
                        "output": tel_row[9] or 0,
                        "cached": tel_row[10] or 0,
                        "total": (tel_row[8] or 0) + (tel_row[9] or 0) + (tel_row[10] or 0),
                    },
                    "cost_usd": tel_row[11] or 0.0,
                    "gemini_api_latency_ms": tel_row[12] or 0.0,
                    "pydantic_retry_count": tel_row[13] or 0,
                }
            else:
                data["telemetry"] = None

            # 6. Quality gate warnings from intermediate JSON
            quality_warnings: List[str] = []
            if output_dir:
                json_path = Path(output_dir) / f"episode_{episode_id}.json"
                if json_path.exists():
                    try:
                        from trebek.schemas import Episode
                        from trebek.llm.validation import _validate_extraction_integrity

                        content = json_path.read_text(encoding="utf-8")
                        episode_obj = Episode.model_validate_json(content)
                        quality_warnings = _validate_extraction_integrity(episode_obj)
                    except Exception as e:
                        quality_warnings.append(f"Could not validate intermediate JSON: {e}")

            data["quality_warnings"] = quality_warnings

            return data
    except sqlite3.OperationalError:
        return None
