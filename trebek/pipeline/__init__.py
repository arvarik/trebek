"""
Pipeline package — orchestrator, workers, stage definitions, and file discovery.

The pipeline processes episodes through 5 stages:
ingest → transcribe → extract → augment → verify
"""

from trebek.pipeline.orchestrator import TrebekPipelineOrchestrator, run_pipeline
from trebek.pipeline.stages import VALID_STAGES

__all__ = ["TrebekPipelineOrchestrator", "run_pipeline", "VALID_STAGES"]
