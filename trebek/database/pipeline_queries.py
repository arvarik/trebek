"""
Pipeline-specific database query operations implemented as a mixin.

Separates domain-specific pipeline queries (polling, retries, telemetry)
from the core ``DatabaseWriter`` actor pattern (queue, execute, transaction).
The mixin is inherited by ``DatabaseWriter`` so all call sites remain unchanged.
"""

import structlog
from typing import Any, Optional

from trebek.status import PipelineStatus

logger = structlog.get_logger()

# Whitelist of valid job_telemetry column names to prevent SQL injection.
# update_job_telemetry() uses f-string interpolation for column names;
# this frozenset ensures only known columns can be written.
_TELEMETRY_COLUMNS: frozenset[str] = frozenset(
    {
        "peak_vram_mb",
        "avg_gpu_utilization_pct",
        "stage_ingestion_ms",
        "stage_gpu_extraction_ms",
        "stage_commercial_filtering_ms",
        "stage_structured_extraction_ms",
        "stage_multimodal_ms",
        "stage_vectorization_ms",
        "gemini_total_input_tokens",
        "gemini_total_output_tokens",
        "gemini_total_cached_tokens",
        "gemini_total_cost_usd",
        "gemini_api_latency_ms",
        "pydantic_retry_count",
    }
)


class PipelineQueryMixin:
    """Mixin providing pipeline-specific database operations.

    Requires the host class to implement ``execute()`` and ``executemany()``
    async methods (provided by ``DatabaseWriter``).
    """

    async def poll_for_work(self, from_status: str, to_status: str) -> Optional[str]:
        """
        Atomic polling query to avoid race conditions between workers.
        Requires SQLite 3.35+ for RETURNING.
        """
        query = """
        UPDATE pipeline_state
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE episode_id = (
            SELECT episode_id
            FROM pipeline_state
            WHERE status = ?
            ORDER BY created_at ASC
            LIMIT 1
        )
        RETURNING episode_id;
        """
        try:
            # Execute will return the lastrowid or the fetchall output depending on RETURNING clause support
            result = await self.execute(query, (to_status, from_status), timeout=5.0)  # type: ignore[attr-defined]
            if result and isinstance(result, list) and len(result) > 0:
                return str(result[0][0])  # extract RETURNING clause
            return None
        except Exception as e:
            logger.error("Error polling for work", error=str(e))
            return None

    async def update_job_telemetry(self, episode_id: str, **kwargs: Any) -> None:
        """
        Upserts job telemetry fields for a given episode.

        Column names are validated against ``_TELEMETRY_COLUMNS`` to prevent
        SQL injection — kwargs keys are interpolated into the UPDATE statement.
        """
        if not kwargs:
            return

        # Validate column names against whitelist to prevent SQL injection
        invalid_columns = set(kwargs.keys()) - _TELEMETRY_COLUMNS
        if invalid_columns:
            raise ValueError(
                f"Invalid telemetry column(s): {sorted(invalid_columns)}. Valid: {sorted(_TELEMETRY_COLUMNS)}"
            )

        # First ensure a row exists
        await self.execute("INSERT OR IGNORE INTO job_telemetry (episode_id) VALUES (?)", (episode_id,))  # type: ignore[attr-defined]

        # Then update the provided fields
        set_clauses = []
        params = []
        for k, v in kwargs.items():
            set_clauses.append(f"{k} = ?")
            params.append(v)

        params.append(episode_id)

        query = f"UPDATE job_telemetry SET {', '.join(set_clauses)} WHERE episode_id = ?"
        await self.execute(query, tuple(params))  # type: ignore[attr-defined]

    async def fail_episode_with_retry(
        self, episode_id: str, previous_status: str, error: str, max_retries: int = 3
    ) -> bool:
        """
        Implements retry-with-backoff for failed episodes.
        Returns True if the episode was permanently failed (retries exhausted),
        False if it was reset for retry.
        """
        rows = await self.execute(  # type: ignore[attr-defined]
            "SELECT retry_count FROM pipeline_state WHERE episode_id = ?", (episode_id,)
        )
        current_retries = rows[0][0] if rows else 0

        if current_retries >= max_retries:
            # Exhausted retries — permanently fail
            await self.execute(  # type: ignore[attr-defined]
                "UPDATE pipeline_state SET status = ?, last_error = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                (PipelineStatus.FAILED, error[:500], episode_id),
            )
            logger.warning(
                "Episode permanently failed (retries exhausted)",
                episode_id=episode_id,
                retries=current_retries,
                max_retries=max_retries,
                error=error[:200],
            )
            return True
        else:
            # Reset to previous status for retry, increment counter
            await self.execute(  # type: ignore[attr-defined]
                "UPDATE pipeline_state SET status = ?, retry_count = retry_count + 1, "
                "last_error = ?, updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                (previous_status, error[:500], episode_id),
            )
            logger.info(
                "Episode queued for retry",
                episode_id=episode_id,
                retry=current_retries + 1,
                max_retries=max_retries,
                reset_to=previous_status,
            )
            return False

    async def reset_episode(
        self,
        episode_id: Optional[str] = None,
        force: bool = False,
        reset_to: str = PipelineStatus.PENDING,
    ) -> int:
        """
        Resets an episode or all failed episodes back to a specified status (default PENDING).
        If episode_id is provided, only that episode is reset.
        If force is True, the episode is reset even if it's not currently FAILED.
        Returns the number of rows reset.
        """
        if episode_id:
            if force:
                query = (
                    "UPDATE pipeline_state SET status = ?, retry_count = 0, last_error = NULL, "
                    "updated_at = CURRENT_TIMESTAMP WHERE episode_id = ? RETURNING episode_id"
                )
                params: tuple[Any, ...] = (reset_to, episode_id)
            else:
                query = (
                    "UPDATE pipeline_state SET status = ?, retry_count = 0, last_error = NULL, "
                    "updated_at = CURRENT_TIMESTAMP WHERE episode_id = ? AND status = ? RETURNING episode_id"
                )
                params = (reset_to, episode_id, PipelineStatus.FAILED)
        else:
            query = (
                "UPDATE pipeline_state SET status = ?, retry_count = 0, last_error = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE status = ? RETURNING episode_id"
            )
            params = (reset_to, PipelineStatus.FAILED)

        result = await self.execute(query, params)  # type: ignore[attr-defined]
        count = len(result) if isinstance(result, list) else 0
        if count > 0:
            logger.info("Reset episode(s) for retry", count=count, episode_id=episode_id, force=force)
        return count

    async def reset_failed_episodes(self) -> int:
        """
        Resets all FAILED episodes back to PENDING for re-processing.
        Returns the count of episodes reset.
        """
        return await self.reset_episode()

    async def insert_job_telemetry(self, telemetry: Any) -> None:
        """
        Inserts a job telemetry record into the database.
        """
        query = """
        INSERT INTO job_telemetry (
            episode_id, peak_vram_mb, avg_gpu_utilization_pct,
            stage_ingestion_ms, stage_gpu_extraction_ms,
            stage_commercial_filtering_ms, stage_structured_extraction_ms,
            stage_multimodal_ms, stage_vectorization_ms,
            gemini_total_input_tokens, gemini_total_output_tokens,
            gemini_total_cached_tokens, gemini_total_cost_usd,
            gemini_api_latency_ms, pydantic_retry_count
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """
        params = (
            telemetry.episode_id,
            telemetry.peak_vram_mb,
            telemetry.avg_gpu_utilization_pct,
            telemetry.stage_ingestion_ms,
            telemetry.stage_gpu_extraction_ms,
            telemetry.stage_commercial_filtering_ms,
            telemetry.stage_structured_extraction_ms,
            telemetry.stage_multimodal_ms,
            telemetry.stage_vectorization_ms,
            telemetry.gemini_total_input_tokens,
            telemetry.gemini_total_output_tokens,
            telemetry.gemini_total_cached_tokens,
            telemetry.gemini_total_cost_usd,
            telemetry.gemini_api_latency_ms,
            telemetry.pydantic_retry_count,
        )
        await self.execute(query, params)  # type: ignore[attr-defined]

    async def search_clues(
        self,
        query: str,
        limit: int = 50,
        round_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Full-text search on clues using SQLite FTS5 with BM25 ranking.
        Searches across category, clue_text, and correct_response.
        Falls back to LIKE query if FTS5 syntax fails or table is unavailable.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        # Check if clues table exists
        has_clues_res = await self.execute(  # type: ignore[attr-defined]
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='clues'"
        )
        if not (has_clues_res and isinstance(has_clues_res, list) and len(has_clues_res) > 0):
            return []

        # If user did not specify exact quotes or operators, tokenize with prefix matching
        words = cleaned_query.split()
        safe_fts_query = " ".join(f'"{w.replace(chr(34), "")}"*' for w in words if w)

        # Check if episodes table exists to join air_date
        has_episodes_res = await self.execute(  # type: ignore[attr-defined]
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='episodes'"
        )
        has_episodes = bool(has_episodes_res and isinstance(has_episodes_res, list) and len(has_episodes_res) > 0)

        ep_join = "LEFT JOIN episodes e ON e.episode_id = c.episode_id" if has_episodes else ""
        air_date_col = "e.air_date" if has_episodes else "NULL AS air_date"

        sql = f"""
        SELECT
            c.clue_id,
            c.episode_id,
            c.round,
            c.category,
            c.board_row,
            c.board_col,
            c.selection_order,
            c.clue_text,
            c.correct_response,
            c.is_daily_double,
            c.is_triple_stumper,
            c.is_verified,
            {air_date_col},
            fts.rank
        FROM clues_fts fts
        JOIN clues c ON c.clue_id = fts.clue_id
        {ep_join}
        WHERE clues_fts MATCH ?
        """
        params: list[Any] = [safe_fts_query]
        if round_filter:
            sql += " AND c.round = ?"
            params.append(round_filter)

        sql += " ORDER BY fts.rank ASC LIMIT ?"
        params.append(limit)

        rows = None
        try:
            rows = await self.execute(sql, tuple(params))  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning("FTS5 match failed, falling back to LIKE search", query=query, error=str(e))
            like_sql = f"""
            SELECT
                c.clue_id,
                c.episode_id,
                c.round,
                c.category,
                c.board_row,
                c.board_col,
                c.selection_order,
                c.clue_text,
                c.correct_response,
                c.is_daily_double,
                c.is_triple_stumper,
                c.is_verified,
                {air_date_col},
                0.0 AS rank
            FROM clues c
            {ep_join}
            WHERE (c.clue_text LIKE ? OR c.category LIKE ? OR c.correct_response LIKE ?)
            """
            like_pat = f"%{cleaned_query}%"
            like_params: list[Any] = [like_pat, like_pat, like_pat]
            if round_filter:
                like_sql += " AND c.round = ?"
                like_params.append(round_filter)
            like_sql += " ORDER BY c.episode_id DESC, c.selection_order ASC LIMIT ?"
            like_params.append(limit)
            rows = await self.execute(like_sql, tuple(like_params))  # type: ignore[attr-defined]

        results: list[dict[str, Any]] = []
        if rows and isinstance(rows, list):
            for r in rows:
                results.append(
                    {
                        "clue_id": r[0],
                        "episode_id": r[1],
                        "round": r[2],
                        "category": r[3],
                        "board_row": r[4],
                        "board_col": r[5],
                        "selection_order": r[6],
                        "clue_text": r[7],
                        "correct_response": r[8],
                        "is_daily_double": bool(r[9]),
                        "is_triple_stumper": bool(r[10]),
                        "is_verified": bool(r[11]),
                        "air_date": r[12],
                        "rank": float(r[13]) if r[13] is not None else 0.0,
                    }
                )
        return results
