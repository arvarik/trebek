"""
Application configuration — environment-driven settings with Pydantic validation.

Reads from ``.env`` files and environment variables. Provides model constants,
pricing data, and supported video format definitions.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Tuple


# All container formats natively supported by FFmpeg's libavformat
SUPPORTED_VIDEO_EXTENSIONS: Tuple[str, ...] = (
    ".mp4",
    ".ts",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".mpg",
    ".mpeg",
    ".flv",
    ".wmv",
    ".m2ts",
    ".vob",
)


# ── Model Constants ──────────────────────────────────────────────
MODEL_FLASH = "gemini-3-flash-preview"
MODEL_PRO = "gemini-3.1-pro-preview"

# CLI alias → canonical model name
MODEL_ALIASES: dict[str, str] = {
    "flash": MODEL_FLASH,
    "pro": MODEL_PRO,
}

# Per-million-token pricing (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    MODEL_FLASH: {"input": 0.075, "output": 0.30},
    MODEL_PRO: {"input": 1.25, "output": 5.00},
}


class Settings(BaseSettings):
    db_path: str = Field(default="trebek.db", description="Path to the SQLite database")
    output_dir: str = Field(default="gpu_outputs", description="Directory to store intermediate pipeline outputs")
    input_dir: str = Field(default="input_videos", description="Directory to poll for new video files")
    gemini_api_key: str = Field(default="", description="GCP / Gemini API Key")
    log_level: str = Field(default="INFO", description="Logging level")

    # GPU constraints
    gpu_vram_target_gb: int = Field(
        default=16, description="Target VRAM ceiling for safety limits (e.g. 16 for 4060/5060 Ti)"
    )
    whisper_batch_size: int = Field(
        default=8, description="WhisperX batch size tuned for 16GB VRAM (safe default; max ~16)"
    )
    whisper_compute_type: str = Field(default="float16", description="Compute type for WhisperX to prevent OOM")

    @field_validator("gpu_vram_target_gb")
    @classmethod
    def validate_gpu_vram(cls, v: int) -> int:
        if v < 4 or v > 24:
            raise ValueError("gpu_vram_target_gb must be >= 4 and <= 24")
        return v

    @field_validator("whisper_compute_type")
    @classmethod
    def validate_whisper_compute_type(cls, v: str) -> str:
        if v not in ("float16", "float32"):
            raise ValueError("whisper_compute_type must be 'float16' or 'float32'")
        return v

    @field_validator("whisper_batch_size")
    @classmethod
    def validate_whisper_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("whisper_batch_size must be > 0")
        return v

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
