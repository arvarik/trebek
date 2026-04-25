import asyncio
import gzip
import json
import structlog
import multiprocessing
import os
import signal
import subprocess
import uuid
import threading
import time
from concurrent.futures import ProcessPoolExecutor


logger = structlog.get_logger()


logger = structlog.get_logger()

def gpu_worker_task(video_filepath: str, output_dir: str) -> tuple[str, float, float]:
    """
    Executes the GPU processing task (Stage 3) and writes results to disk to avoid
    IPC serialization bottleneck of massive JSON structures.
    """

    file_id = uuid.uuid4().hex
    audio_path = os.path.join(output_dir, f"audio_{file_id}.wav")
    whisperx_output_json = os.path.join(output_dir, f"audio_{file_id}.json")

    peak_vram_mb = 0.0
    avg_gpu_utilization_pct = 0.0

    stop_event = threading.Event()
    metrics = {"peak_vram": 0.0, "util_sum": 0.0, "util_count": 0}

    def monitor_gpu():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not stop_event.is_set():
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_mb = float(info.used) / (1024 * 1024)
                    if vram_mb > metrics["peak_vram"]:
                        metrics["peak_vram"] = vram_mb
                        
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics["util_sum"] += float(util.gpu)
                    metrics["util_count"] += 1
                except Exception:
                    pass
                time.sleep(0.5)
            pynvml.nvmlShutdown()
        except Exception:
            pass

    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()

    # 1. FFmpeg extraction
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_filepath, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 2. WhisperX / Pyannote
    subprocess.run(
        ["whisperx", audio_path, "--output_dir", output_dir, "--output_format", "json", "--compute_type", "float16"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Read the JSON from WhisperX
    with open(whisperx_output_json, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    processed_result = {
        "status": "success",
        "video_filepath": video_filepath,
        "transcript": transcript_data,
    }

    output_path = os.path.join(output_dir, f"gpu_output_{file_id}.json.gz")

    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(processed_result, f)

    # Cleanup intermediate files
    try:
        os.remove(audio_path)
        os.remove(whisperx_output_json)
    except OSError:
        pass

    stop_event.set()
    monitor_thread.join(timeout=1.0)
    
    if metrics["util_count"] > 0:
        avg_gpu_utilization_pct = metrics["util_sum"] / metrics["util_count"]
    peak_vram_mb = metrics["peak_vram"]

    return output_path, peak_vram_mb, avg_gpu_utilization_pct


class GPUOrchestrator:
    """
    Manages the lifecycle of subprocesses ensuring VRAM reclamation and
    strict cleanup on crashes via SIGKILL.
    """

    def __init__(self, output_dir: str, max_workers: int = 1):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 'spawn' is required for CUDA to prevent driver initialization collisions
        context = multiprocessing.get_context("spawn")

        # max_tasks_per_child=1 ensures worker death and VRAM flush per task (Python 3.11+)
        # Setting max_workers=1 to limit memory usage specifically for Stage 3 constraint.
        self.executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=context, max_tasks_per_child=1)
        self._child_pids: list[int] = []

    async def execute_gpu_work(self, video_filepath: str) -> tuple[str, float, float]:
        """
        Dispatches video filepath to the worker pool and awaits the resulting filepath.
        """
        loop = asyncio.get_running_loop()
        filepath, peak_vram, avg_util = await loop.run_in_executor(self.executor, gpu_worker_task, video_filepath, self.output_dir)
        return filepath, peak_vram, avg_util

    def shutdown(self) -> None:
        """
        Forcefully SIGKILL any zombie workers, then shut down the executor cleanly.
        This is called from the orchestrator's shutdown path, which is triggered
        by the asyncio signal handlers registered in main.py.
        """
        # Forcefully kill child processes to reclaim GPU memory
        try:
            for pid in list(self.executor._processes.keys()):
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.info("Force killed worker PID during shutdown", pid=pid)
                except ProcessLookupError:
                    pass
        except Exception:
            pass  # executor may already be cleaned up
        self.executor.shutdown(wait=True, cancel_futures=True)
