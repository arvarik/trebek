"""
Trebek — High-fidelity multimodal AI pipeline for Jeopardy! data extraction.

Processes video recordings through a 5-stage pipeline:
    ingest → transcribe (WhisperX) → extract (Gemini) → augment → verify

Architecture:
    - ``llm/``        — Gemini API passes (speaker anchoring, extraction, multimodal)
    - ``pipeline/``   — Orchestrator, workers, stage definitions
    - ``database/``   — SQLite actor-pattern writer with pipeline operations
    - ``gpu/``        — WhisperX GPU worker pool with warm start
    - ``ui/``         — Rich console rendering, progress, dashboard
    - ``analysis/``   — Post-extraction analytics (buzzer physics, embeddings)
"""

__version__ = "0.1.6"
