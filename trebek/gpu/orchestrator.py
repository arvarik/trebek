import asyncio
import structlog
import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor

logger = structlog.get_logger()


class GPUOrchestrator:
    """
    Manages the lifecycle of subprocesses ensuring VRAM reclamation and
    strict cleanup on crashes via SIGKILL.
    """

    def __init__(self, output_dir: str, batch_size: int = 8, compute_type: str = "float16", max_workers: int = 1):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.max_workers = max_workers
        os.makedirs(output_dir, exist_ok=True)
        self._start_pool()

    def _start_pool(self) -> None:
        """Starts or restarts the worker pool."""
        # 'spawn' is required for CUDA to prevent driver initialization collisions
        context = multiprocessing.get_context("spawn")

        # Keeping max_tasks_per_child unset allows warm worker architecture
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=context)

    def _restart_pool(self) -> None:
        """Restarts the worker pool (used to recover from CUDA OOM)."""
        self.shutdown()
        self._start_pool()

    async def execute_gpu_work(self, video_filepath: str) -> tuple[str, float, float]:
        """
        Dispatches video filepath to the worker pool and awaits the resulting filepath.
        """
        loop = asyncio.get_running_loop()
        import functools
        from trebek.gpu.worker import gpu_worker_task

        fn = functools.partial(gpu_worker_task, video_filepath, self.output_dir, self.batch_size, self.compute_type)
        try:
            filepath, peak_vram, avg_util = await loop.run_in_executor(self.executor, fn)
            return filepath, peak_vram, avg_util
        except MemoryError as e:
            if "CUDA OOM" in str(e):
                logger.error("Caught CUDA OOM error, restarting GPU worker pool", error=str(e))
                self._restart_pool()
            raise

    def shutdown(self) -> None:
        """
        Gracefully terminate subprocesses using SIGTERM, falling back to SIGKILL.
        """
        try:
            import psutil

            pids = list(self.executor._processes.keys())
            if not pids:
                return

            procs = []
            # 1. Send SIGTERM politely
            for pid in pids:
                try:
                    p = psutil.Process(pid)
                    p.terminate()  # SIGTERM
                    procs.append(p)
                    logger.info("Sent SIGTERM to worker PID", pid=pid)
                except psutil.NoSuchProcess:
                    pass

            # 2. Wait up to 5 seconds
            gone, alive = psutil.wait_procs(procs, timeout=5.0)

            # 3. Escalate to SIGKILL for the stubborn ones
            for p in alive:
                try:
                    p.kill()  # SIGKILL
                    logger.warning("Force killed worker PID after timeout (SIGKILL)", pid=p.pid)
                except psutil.NoSuchProcess:
                    pass
        except ImportError:
            # Fallback if psutil isn't available
            for pid in list(self.executor._processes.keys()):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except Exception:
            pass  # executor may already be cleaned up

        self.executor.shutdown(wait=True, cancel_futures=True)
