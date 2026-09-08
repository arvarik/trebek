"""
GPU package — WhisperX transcription worker pool with warm start architecture.

Manages subprocess-based GPU workers via ``ProcessPoolExecutor`` with
CUDA-safe ``spawn`` context. Model weights stay resident in VRAM across
tasks to eliminate cold-start latency.
"""

from .orchestrator import GPUOrchestrator
from .worker import reset_gpu_models, reset_warm_models

clear_gpu_cache = reset_gpu_models

__all__ = ["GPUOrchestrator", "reset_gpu_models", "reset_warm_models", "clear_gpu_cache"]
