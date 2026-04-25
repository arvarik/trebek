import asyncio
import gzip
import json
import structlog
import multiprocessing
import os
import signal
import sys
import uuid
import subprocess
from concurrent.futures import ProcessPoolExecutor
from typing import Any

logger = structlog.get_logger()


def gpu_worker_task(video_filepath: str, output_dir: str) -> str:
    """
    Executes the GPU processing task (Stage 3) and writes results to disk to avoid
    IPC serialization bottleneck of massive JSON structures.
    """
    
    file_id = uuid.uuid4().hex
    audio_path = os.path.join(output_dir, f"audio_{file_id}.wav")
    whisperx_output_json = os.path.join(output_dir, f"audio_{file_id}.json")
    
    # 1. FFmpeg extraction
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_filepath, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # 2. WhisperX / Pyannote
    subprocess.run(
        ["whisperx", audio_path, "--output_dir", output_dir, "--output_format", "json", "--compute_type", "float16"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
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

    return output_path


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
        self._setup_safety_handlers()

    def _setup_safety_handlers(self) -> None:
        """
        Registers signal handlers to forcefully SIGKILL zombie workers if the
        main event loop panics or receives an interrupt.
        """

        def kill_children(signum: int, frame: Any) -> None:
            logger.critical(f"Orchestrator received signal {signum}. Sending SIGKILL to children...")
            for pid in self.executor._processes.keys():
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"Force killed worker PID: {pid}")
                except ProcessLookupError:
                    pass
            sys.exit(signum)

        signal.signal(signal.SIGINT, kill_children)
        signal.signal(signal.SIGTERM, kill_children)

    async def execute_gpu_work(self, video_filepath: str) -> str:
        """
        Dispatches video filepath to the worker pool and awaits the resulting filepath.
        """
        loop = asyncio.get_running_loop()
        filepath = await loop.run_in_executor(self.executor, gpu_worker_task, video_filepath, self.output_dir)
        return filepath

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
