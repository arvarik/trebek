import asyncio
import structlog
import sqlite3
from typing import Any, List, Optional, Tuple

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

    async def executemany(
        self, query: str, params_list: List[Tuple[Any, ...]], timeout: float = 10.0
    ) -> Any:
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
