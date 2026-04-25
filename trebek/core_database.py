import asyncio
import structlog
import sqlite3
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from trebek.schemas import Episode
    from trebek.state_machine import TrebekStateMachine

logger = structlog.get_logger()


class DatabaseWriter:
    """
    Actor Pattern for SQLite write-operations, serializing concurrent requests
    while guaranteeing that waiting coroutines never deadlock.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.queue: asyncio.Queue[Tuple[str, Any, asyncio.Future[Any]]] = asyncio.Queue()
        self.task: Optional[asyncio.Task[Any]] = None
        self.vacuum_task: Optional[asyncio.Task[Any]] = None
        self.conn: Optional[sqlite3.Connection] = None

    async def start(self) -> None:
        self.conn = sqlite3.connect(self.db_path)
        # Apply strict pragma configurations at startup
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA busy_timeout=5000;")
        self.conn.execute("PRAGMA auto_vacuum=INCREMENTAL;")
        self.task = asyncio.create_task(self._process_queue())
        self.vacuum_task = asyncio.create_task(self._background_incremental_vacuum())

    async def stop(self) -> None:
        if self.vacuum_task:
            self.vacuum_task.cancel()
            try:
                await self.vacuum_task
            except asyncio.CancelledError:
                pass
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self.conn:
            self.conn.close()

    async def _process_queue(self) -> None:
        """
        Internal transaction loop. Wrapped in a try/finally block to guarantee
        future resolution even if the SQLite driver or event loop crashes.
        """
        try:
            while True:
                query, params, future = await self.queue.get()
                try:
                    assert self.conn is not None
                    cursor = self.conn.cursor()
                    if isinstance(params, list):
                        cursor.executemany(query, params)
                    else:
                        cursor.execute(query, params)
                    # Fetch results BEFORE commit — RETURNING clause results are only
                    # available while the transaction is still active
                    result: Any = cursor.fetchall() if cursor.description else None
                    self.conn.commit()
                    if result is None:
                        result = cursor.lastrowid
                    if not future.done():
                        future.set_result(result)
                except Exception as db_err:
                    if self.conn is not None:
                        self.conn.rollback()
                    logger.error("Transaction failed", error=str(db_err))
                    if not future.done():
                        future.set_exception(db_err)
                finally:
                    self.queue.task_done()

        except asyncio.CancelledError:
            logger.info("DatabaseWriter actor shutting down via cancellation.")
        except Exception as critical_err:
            logger.critical("DatabaseWriter fatal crash", error=str(critical_err))
        finally:
            # DEADLOCK PROTECTION: Forcefully resolve all pending futures in queue
            while not self.queue.empty():
                try:
                    _, _, pending_future = self.queue.get_nowait()
                    if not pending_future.done():
                        pending_future.set_exception(RuntimeError("DatabaseWriter crashed unexpectedly"))
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    break

    async def execute(self, query: str, params: Tuple[Any, ...] = (), timeout: float = 10.0) -> Any:
        """
        Enqueues query for actor processing and protects the caller with wait_for().
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((query, params, future))

        # Deadlock safeguard: Ensure we don't wait forever if the actor crashes silently
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as te:
            logger.error("Query execution timed out waiting for DatabaseWriter", query=query)
            raise te

    async def executemany(self, query: str, params_list: List[Tuple[Any, ...]], timeout: float = 10.0) -> Any:
        """
        Enqueues an executemany query for actor processing and protects the caller with wait_for().
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((query, params_list, future))

        # Deadlock safeguard: Ensure we don't wait forever if the actor crashes silently
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as te:
            logger.error("Query executemany timed out waiting for DatabaseWriter", query=query)
            raise te

    async def _background_incremental_vacuum(self, interval_seconds: int = 300) -> None:
        """
        Background asyncio task that triggers PRAGMA incremental_vacuum periodically
        to free pages without blocking pipeline throughput.
        """
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                logger.info("Triggering background SQLite PRAGMA incremental_vacuum...")
                await self.execute("PRAGMA incremental_vacuum;", (), timeout=30.0)
            except asyncio.CancelledError:
                logger.info("Vacuum background task cancelled.")
                break
            except Exception as e:
                logger.error("Incremental vacuum background task failed", error=str(e))

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
            result = await self.execute(query, (to_status, from_status), timeout=5.0)
            if result and isinstance(result, list) and len(result) > 0:
                return result[0][0]  # extract RETURNING clause
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
        await self.execute("INSERT OR IGNORE INTO job_telemetry (episode_id) VALUES (?)", (episode_id,))

        # Then update the provided fields
        set_clauses = []
        params = []
        for k, v in kwargs.items():
            set_clauses.append(f"{k} = ?")
            params.append(v)

        params.append(episode_id)

        query = f"UPDATE job_telemetry SET {', '.join(set_clauses)} WHERE episode_id = ?"
        await self.execute(query, tuple(params))

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
        await self.execute(query, params)


async def commit_episode_to_relational_tables(
    db_writer: DatabaseWriter,
    episode_id: str,
    episode_data: "Episode",
    state_machine: "TrebekStateMachine",
) -> None:
    """
    Commits verified episode data into the normalized relational tables.
    This is the Stage 8 relational commit — writing to episodes, contestants,
    clues, buzz_attempts, wagers, score_adjustments, and episode_performances.
    """
    import uuid

    # 1. Insert episode metadata
    await db_writer.execute(
        "INSERT OR IGNORE INTO episodes (episode_id, air_date, host_name, is_tournament) VALUES (?, ?, ?, ?)",
        (episode_id, episode_data.episode_date, episode_data.host_name, episode_data.is_tournament),
    )

    # 2. Insert contestants
    for contestant in episode_data.contestants:
        contestant_id = f"{episode_id}_{contestant.name.replace(' ', '_').lower()}"
        await db_writer.execute(
            "INSERT OR IGNORE INTO contestants "
            "(contestant_id, name, occupational_category, is_returning_champion) VALUES (?, ?, ?, ?)",
            (contestant_id, contestant.name, contestant.occupational_category, contestant.is_returning_champion),
        )

        # 3. Insert episode_performances with final scores from state machine
        final_score = state_machine.scores.get(contestant.name, 0)
        await db_writer.execute(
            "INSERT OR IGNORE INTO episode_performances "
            "(episode_id, contestant_id, podium_position, final_score) VALUES (?, ?, ?, ?)",
            (episode_id, contestant_id, contestant.podium_position, final_score),
        )

    # 4. Insert clues and buzz_attempts
    for clue in episode_data.clues:
        clue_id = f"{episode_id}_c{clue.selection_order}"
        is_triple_stumper = len(clue.attempts) == 0 or all(not a.is_correct for a in clue.attempts)

        dd_wager_str = str(clue.daily_double_wager) if clue.daily_double_wager is not None else None

        await db_writer.execute(
            "INSERT OR IGNORE INTO clues "
            "(clue_id, episode_id, round, category, board_row, board_col, selection_order, "
            "clue_text, correct_response, is_daily_double, is_triple_stumper, "
            "daily_double_wager, wagerer_name, requires_visual_context, "
            "host_start_timestamp_ms, host_finish_timestamp_ms, clue_syllable_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                clue_id,
                episode_id,
                clue.round,
                clue.category,
                clue.board_row,
                clue.board_col,
                clue.selection_order,
                clue.clue_text,
                clue.correct_response,
                clue.is_daily_double,
                is_triple_stumper,
                dd_wager_str,
                clue.wagerer_name,
                clue.requires_visual_context,
                clue.host_start_timestamp_ms,
                clue.host_finish_timestamp_ms,
                clue.clue_syllable_count,
            ),
        )

        # Insert buzz_attempts for this clue
        for attempt in clue.attempts:
            attempt_id = f"{clue_id}_a{attempt.attempt_order}_{uuid.uuid4().hex[:8]}"
            contestant_id = f"{episode_id}_{attempt.speaker.replace(' ', '_').lower()}"
            await db_writer.execute(
                "INSERT OR IGNORE INTO buzz_attempts "
                "(attempt_id, clue_id, contestant_id, attempt_order, buzz_timestamp_ms, "
                "response_given, is_correct, response_start_timestamp_ms, is_lockout_inferred) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    attempt_id,
                    clue_id,
                    contestant_id,
                    attempt.attempt_order,
                    attempt.buzz_timestamp_ms,
                    attempt.response_given,
                    attempt.is_correct,
                    attempt.response_start_timestamp_ms,
                    attempt.is_lockout_inferred,
                ),
            )

    # 5. Insert score_adjustments
    for adj in episode_data.score_adjustments:
        adjustment_id = f"{episode_id}_adj_{adj.effective_after_clue_selection_order}_{uuid.uuid4().hex[:8]}"
        contestant_id = f"{episode_id}_{adj.contestant.replace(' ', '_').lower()}"
        await db_writer.execute(
            "INSERT OR IGNORE INTO score_adjustments "
            "(adjustment_id, episode_id, contestant_id, points_adjusted, reason, "
            "effective_after_clue_selection_order) VALUES (?, ?, ?, ?, ?, ?)",
            (
                adjustment_id,
                episode_id,
                contestant_id,
                adj.points_adjusted,
                adj.reason,
                adj.effective_after_clue_selection_order,
            ),
        )

    logger.info(
        "Relational commit complete",
        episode_id=episode_id,
        clues=len(episode_data.clues),
        contestants=len(episode_data.contestants),
    )
