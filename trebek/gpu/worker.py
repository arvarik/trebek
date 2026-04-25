import gzip
import json
import structlog
import os
import subprocess
import uuid
import threading
import time
from typing import Any

logger = structlog.get_logger()
_whisperx_model: Any = None


def gpu_worker_task(
    video_filepath: str, output_dir: str, batch_size: int = 8, compute_type: str = "float16"
) -> tuple[str, float, float]:
    """
    Executes the GPU processing task (Stage 3) and writes results to disk to avoid
    IPC serialization bottleneck of massive JSON structures.
    Uses Warm Worker architecture to keep model weights in VRAM.
    """
    import gc
    import torch
    import whisperx

    file_id = uuid.uuid4().hex
    audio_path = os.path.join(output_dir, f"audio_{file_id}.wav")

    peak_vram_mb = 0.0
    avg_gpu_utilization_pct = 0.0

    stop_event = threading.Event()
    metrics = {"peak_vram": 0.0, "util_sum": 0.0, "util_count": 0}

    def monitor_gpu() -> None:
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
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_filepath, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        err_msg = (result.stderr or "").strip().splitlines()
        # Last 3 lines of ffmpeg stderr are usually the most informative
        detail = "\n".join(err_msg[-3:]) if err_msg else f"exit code {result.returncode}"
        raise RuntimeError(f"ffmpeg failed for {video_filepath}: {detail}")

    # 2. WhisperX - Native API with Warm Worker
    global _whisperx_model
    if "_whisperx_model" not in globals() or _whisperx_model is None:
        logger.info("Loading WhisperX model into VRAM (Cold Start)...")
        _whisperx_model = whisperx.load_model("large-v3", device="cuda", compute_type=compute_type, language="en")
    else:
        logger.info("Using cached WhisperX model (Warm Start)...")

    try:
        audio = whisperx.load_audio(audio_path)
        transcript_data = _whisperx_model.transcribe(audio, batch_size=batch_size, language="en")

        # Explicit Memory Management
        del audio
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        # Check for OOM specifically
        if "OutOfMemoryError" in str(type(e).__name__) or "CUDA out of memory" in str(e):
            raise MemoryError("CUDA OOM") from e
        raise RuntimeError(f"whisperx failed: {str(e)}")

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
    except OSError:
        pass

    stop_event.set()
    monitor_thread.join(timeout=1.0)

    if metrics["util_count"] > 0:
        avg_gpu_utilization_pct = metrics["util_sum"] / metrics["util_count"]
    peak_vram_mb = metrics["peak_vram"]

    return output_path, peak_vram_mb, avg_gpu_utilization_pct
