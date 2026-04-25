import pytest
from pydantic import ValidationError
from src.config import Settings


def test_config_gpu_vram_bounds() -> None:
    # Valid bound
    settings = Settings(gpu_vram_target_gb=16)
    assert settings.gpu_vram_target_gb == 16

    # Lower bound failure (< 4)
    with pytest.raises(ValidationError, match="gpu_vram_target_gb"):
        Settings(gpu_vram_target_gb=2)

    # Upper bound failure (> 24)
    with pytest.raises(ValidationError, match="gpu_vram_target_gb"):
        Settings(gpu_vram_target_gb=32)


def test_config_whisper_compute_type() -> None:
    # Valid types
    assert Settings(whisper_compute_type="float16").whisper_compute_type == "float16"
    assert Settings(whisper_compute_type="float32").whisper_compute_type == "float32"

    # Invalid type
    with pytest.raises(ValidationError, match="whisper_compute_type"):
        Settings(whisper_compute_type="float64")
    
    with pytest.raises(ValidationError, match="whisper_compute_type"):
        Settings(whisper_compute_type="int8")


def test_config_whisper_batch_size() -> None:
    # Valid sizes
    assert Settings(whisper_batch_size=8).whisper_batch_size == 8

    # Invalid sizes (<= 0)
    with pytest.raises(ValidationError, match="whisper_batch_size"):
        Settings(whisper_batch_size=0)
        
    with pytest.raises(ValidationError, match="whisper_batch_size"):
        Settings(whisper_batch_size=-4)
