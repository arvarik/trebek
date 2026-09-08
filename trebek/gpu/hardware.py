"""
Hardware acceleration detection for CUDA, Apple Silicon (MPS), and CPU fallback.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareInfo:
    device: str  # "cuda", "mps", "cpu"
    device_name: str  # e.g. "NVIDIA GeForce RTX 4090", "Apple Silicon (MPS)", "CPU"
    is_cuda: bool
    is_mps: bool
    is_cpu: bool
    vram_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    whisper_device: str = "cuda"  # Device WhisperX/CTranslate2 can use ("cuda" or "cpu")
    recommended_compute: str = "float16"


def detect_hardware() -> HardwareInfo:
    """Safely inspects PyTorch and CUDA availability without throwing if torch is missing."""
    try:
        import torch

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram = None
            try:
                props = torch.cuda.get_device_properties(0)
                vram = round(props.total_memory / (1024**3), 1)
            except Exception:
                pass
            cuda_ver = getattr(torch.version, "cuda", None)
            return HardwareInfo(
                device="cuda",
                device_name=dev_name,
                is_cuda=True,
                is_mps=False,
                is_cpu=False,
                vram_gb=vram,
                cuda_version=cuda_ver,
                whisper_device="cuda",
                recommended_compute="float16",
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Note: CTranslate2 (used by WhisperX) does NOT support MPS; it only supports cuda and cpu.
            return HardwareInfo(
                device="mps",
                device_name="Apple Silicon (MPS)",
                is_cuda=False,
                is_mps=True,
                is_cpu=False,
                whisper_device="cpu",
                recommended_compute="int8",
            )
        else:
            return HardwareInfo(
                device="cpu",
                device_name="CPU",
                is_cuda=False,
                is_mps=False,
                is_cpu=True,
                whisper_device="cpu",
                recommended_compute="int8",
            )
    except ImportError:
        return HardwareInfo(
            device="cpu",
            device_name="CPU (PyTorch not installed)",
            is_cuda=False,
            is_mps=False,
            is_cpu=True,
            whisper_device="cpu",
            recommended_compute="int8",
        )
