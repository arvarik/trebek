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
_whisperx_align_model: Any = None
_whisperx_align_metadata: Any = None


def gpu_worker_task(
    video_filepath: str, output_dir: str, batch_size: int = 8, compute_type: str = "float16"
) -> tuple[str, float, float]:
    """
    Executes the GPU processing task (Stage 3) and writes results to disk to avoid
    IPC serialization bottleneck of massive JSON structures.
    Uses Warm Worker architecture to keep model weights in VRAM.

    Pipeline:
    1. FFmpeg audio extraction (video → WAV)
    2. WhisperX transcription (large-v3, paragraph-level segments)
    3. WhisperX forced alignment (wav2vec2, word-level timestamps)
       — This step is CRITICAL: without it, segments are ~25s paragraphs
       with zero word-level tokens, causing downstream LLM extraction
       to miss 50%+ of clues.
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
                except Exception as e:
                    logger.debug("pynvml loop error", error=str(e))
                time.sleep(0.5)
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning("pynvml failed, falling back to torch.cuda", error=str(e))
            # Fallback to PyTorch VRAM tracking if pynvml is not installed or fails
            try:
                import torch

                while not stop_event.is_set():
                    try:
                        vram_mb = float(torch.cuda.max_memory_allocated(0)) / (1024 * 1024)
                        if vram_mb > metrics["peak_vram"]:
                            metrics["peak_vram"] = vram_mb
                        # Torch cannot easily provide utilization %
                    except Exception:
                        pass
                    time.sleep(0.5)
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

    # 2. WhisperX Transcription — Warm Worker
    global _whisperx_model, _whisperx_align_model, _whisperx_align_metadata
    if "_whisperx_model" not in globals() or _whisperx_model is None:
        logger.info("Loading WhisperX model into VRAM (Cold Start)...")
        _whisperx_model = whisperx.load_model("large-v3", device="cuda", compute_type=compute_type, language="en")
    else:
        logger.info("Using cached WhisperX model (Warm Start)...")

    try:
        audio = whisperx.load_audio(audio_path)
        transcript_data = _whisperx_model.transcribe(audio, batch_size=batch_size, language="en")

        raw_segment_count = len(transcript_data.get("segments", []))
        logger.info(
            "WhisperX transcription complete (pre-alignment)",
            segments=raw_segment_count,
        )

        # 3. Forced Alignment — word-level timestamps via wav2vec2
        # Without this step, segments are ~25s paragraph chunks with zero
        # word-level tokens, making Line ID-based extraction nearly impossible.
        if "_whisperx_align_model" not in globals() or _whisperx_align_model is None:
            logger.info("Loading WhisperX alignment model (wav2vec2)...")
            _whisperx_align_model, _whisperx_align_metadata = whisperx.load_align_model(
                language_code="en", device="cuda"
            )

        try:
            aligned_result = whisperx.align(
                transcript_data["segments"],
                _whisperx_align_model,
                _whisperx_align_metadata,
                audio,
                device="cuda",
                return_char_alignments=False,
            )
            aligned_segment_count = len(aligned_result.get("segments", []))
            word_count = sum(len(s.get("words", [])) for s in aligned_result.get("segments", []))
            logger.info(
                "WhisperX alignment complete",
                pre_alignment_segments=raw_segment_count,
                post_alignment_segments=aligned_segment_count,
                word_level_tokens=word_count,
            )
            transcript_data = aligned_result
        except Exception as align_err:
            logger.warning(
                "WhisperX alignment failed, using unaligned transcript",
                error=str(align_err)[:200],
            )
            # Fall through with the unaligned transcript_data

        # 4. Speaker Diarization — assign SPEAKER_XX labels to each segment
        # Requires HF_TOKEN env var for pyannote/speaker-diarization-3.1 (gated model).
        # Without this step, segments lack the 'speaker' field and downstream
        # LLM extraction cannot determine who said what.
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            try:
                from whisperx.diarize import DiarizationPipeline

                diarize_model = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-3.1",
                    token=hf_token,
                    device="cuda",
                )
                diarize_segments = diarize_model(
                    audio_path,
                    min_speakers=3,  # At least host + 2 contestants
                )
                transcript_data = whisperx.assign_word_speakers(diarize_segments, transcript_data)
                speaker_set = {s.get("speaker", "?") for s in transcript_data.get("segments", [])}
                logger.info(
                    "WhisperX diarization complete",
                    speakers_found=len(speaker_set),
                    speaker_ids=sorted(speaker_set),
                )
                del diarize_model
            except Exception as diarize_err:
                logger.warning(
                    "WhisperX diarization failed, segments will lack speaker labels",
                    error=str(diarize_err)[:200],
                )
        else:
            logger.warning(
                "HF_TOKEN not set — skipping speaker diarization. "
                "Segments will lack 'speaker' field. Set HF_TOKEN env var "
                "with a HuggingFace token that has access to pyannote/speaker-diarization-3.1"
            )

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
