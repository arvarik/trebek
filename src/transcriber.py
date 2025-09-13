from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from faster_whisper import WhisperModel  # type: ignore

from config import ModelConfig


class Transcriber:
    """Wraps faster-whisper model initialization and transcription settings."""

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        cfg = model_config or ModelConfig()
        device, compute_type = self._select_device_and_compute_type(cfg)
        self._model = WhisperModel(
            cfg.model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
            num_workers=1,
        )

    def _select_device_and_compute_type(self, cfg: ModelConfig) -> Tuple[str, str]:
        # Prefer Metal on Apple Silicon; faster-whisper uses CoreML via Metal when device="metal".
        if cfg.preferred_device == "metal":
            return "metal", cfg.compute_type_metal
        if cfg.preferred_device == "cpu":
            return "cpu", cfg.compute_type_cpu
        # Auto strategy: default to metal, fall back to cpu
        try:
            return "metal", cfg.compute_type_metal
        except Exception:
            return "cpu", cfg.compute_type_cpu

    def transcribe(self, audio_path: str) -> Dict:
        """Transcribe the given WAV file with settings favoring accuracy.

        Returns a JSON-serializable dictionary with segments and text.
        """
        segments_iter, info = self._model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=True,
            word_timestamps=True,
        )

        segments: List[Dict] = []
        for seg in segments_iter:
            segments.append(
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "words": [
                        {
                            "start": w.start,
                            "end": w.end,
                            "word": w.word,
                            "prob": getattr(w, "probability", None),
                        }
                        for w in (seg.words or [])
                    ],
                }
            )

        result = {
            "language": info.language,
            "duration": info.duration,
            "text": " ".join(s["text"].strip() for s in segments).strip(),
            "segments": segments,
            "model": "faster-whisper",
            "model_size": self._model.model_size,
        }
        return result


