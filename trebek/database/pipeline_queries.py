"""
Pipeline-specific database query operations implemented as a mixin.

Separates domain-specific pipeline queries (polling, retries, telemetry)
from the core ``DatabaseWriter`` actor pattern (queue, execute, transaction).
The mixin is inherited by ``DatabaseWriter`` so all call sites remain unchanged.
"""

import structlog
from typing import Any, Optional

logger = structlog.get_logger()


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
        """
        if not kwargs:
            return

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
                "UPDATE pipeline_state SET status = 'FAILED', last_error = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE episode_id = ?",
                (error[:500], episode_id),
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

    async def reset_failed_episodes(self) -> int:
        """
        Resets all FAILED episodes back to PENDING for re-processing.
        Returns the count of episodes reset.
        """
        result = await self.execute(  # type: ignore[attr-defined]
            "UPDATE pipeline_state SET status = 'PENDING', retry_count = 0, last_error = NULL, "
            "updated_at = CURRENT_TIMESTAMP WHERE status = 'FAILED' RETURNING episode_id"
        )
        count = len(result) if isinstance(result, list) else 0
        if count > 0:
            logger.info("Reset failed episodes for retry", count=count)
        return count

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
