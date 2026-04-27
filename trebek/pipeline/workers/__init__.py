"""
Pipeline workers — async coroutines for each processing stage.

Each worker polls for episodes at a specific pipeline status,
processes them, and advances the status:
    - ``ingestion`` — Scans for video files and registers them
    - ``gpu`` — Dispatches to WhisperX for transcription
    - ``llm`` — Runs Gemini extraction passes (1 + 2)
    - ``multimodal`` — Temporal sniping for visual cues (Pass 3)
    - ``state_machine`` — Game state verification and relational commit
"""

from .ingestion import ingestion_worker, run_ingestion_pass
from .gpu import extractor_worker
from .llm import llm_worker
from .multimodal import multimodal_worker
from .state_machine import state_machine_worker

__all__ = [
    "ingestion_worker",
    "run_ingestion_pass",
    "extractor_worker",
    "llm_worker",
    "multimodal_worker",
    "state_machine_worker",
]
