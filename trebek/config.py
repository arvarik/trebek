"""
Application configuration — environment-driven settings with Pydantic validation.

Reads from ``.env`` files and environment variables. Provides model constants,
pricing data, and supported video format definitions.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field, field_validator
from typing import Any, Tuple


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

# Incomplete, temporary, or active download extensions to ignore during scanning
IGNORED_EXTENSIONS: Tuple[str, ...] = (
    ".part",
    ".crdownload",
    ".tmp",
    ".temp",
    ".partial",
    ".aria2",
    ".ytdl",
)


# ── Model Constants ──────────────────────────────────────────────
MODEL_FLASH = "gemini-3.8-flash"
MODEL_FLASH38 = "gemini-3.8-flash"
MODEL_FLASH_LITE = "gemini-3.1-flash-lite-preview"
MODEL_FLASH3 = "gemini-3-flash-preview"
MODEL_PRO = "gemini-3.1-pro-preview"
MODEL_EMBEDDING = "gemini-embedding-001"

# CLI alias / provider string → canonical model name
MODEL_ALIASES: dict[str, str] = {
    "flash": MODEL_FLASH,
    "gemini-flash": MODEL_FLASH,
    "flash38": MODEL_FLASH38,
    "gemini-3.8-flash": MODEL_FLASH38,
    "flash-lite": MODEL_FLASH_LITE,
    "gemini-flash-lite": MODEL_FLASH_LITE,
    "flash3": MODEL_FLASH3,
    "gemini-flash-3": MODEL_FLASH3,
    "pro": MODEL_PRO,
    "gemini-pro": MODEL_PRO,
    "gemini-3.1-pro": MODEL_PRO,
}


def resolve_model_name(name: str | None) -> str:
    """Resolves a user-provided model alias, provider-prefixed string,
    or explicit model ID to its canonical Gemini model identifier.

    Examples:
        - "gemini-flash", "flash" -> "gemini-3.8-flash"
        - "gemini-pro", "pro" -> "gemini-3.1-pro-preview"
        - "flash-lite", "gemini-flash-lite" -> "gemini-3.1-flash-lite-preview"
        - "gemini-3.8-flash" -> "gemini-3.8-flash"
        - "models/gemini-3.8-flash" -> "gemini-3.8-flash"
        - "gemini/gemini-flash" -> "gemini-3.8-flash"
        - "google/gemini-flash" -> "gemini-3.8-flash"
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return MODEL_PRO
    # Strip leading 'models/' or provider prefixes like 'gemini/' or 'google/'
    if cleaned.startswith("models/"):
        cleaned = cleaned[len("models/") :]
    if cleaned.startswith("gemini/"):
        cleaned = cleaned[len("gemini/") :]
    elif cleaned.startswith("google/"):
        cleaned = cleaned[len("google/") :]

    key = cleaned.lower().replace("_", "-")

    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]

    return cleaned


# Per-million-token pricing (USD) — Standard tier, prompts ≤200k tokens
# Source: https://ai.google.dev/gemini-api/docs/pricing#standard (2026-04-26)
MODEL_PRICING: dict[str, dict[str, float]] = {
    MODEL_FLASH: {"input": 0.50, "output": 3.00},
    MODEL_FLASH38: {"input": 0.50, "output": 3.00},
    MODEL_FLASH_LITE: {"input": 0.25, "output": 1.50},
    MODEL_FLASH3: {"input": 0.50, "output": 3.00},
    MODEL_PRO: {"input": 2.00, "output": 12.00},
}

# Canonical J! host names — used for host validation and speaker reconciliation.
# The LLM sometimes misidentifies contestants as hosts; this list provides a
# ground-truth allowlist for override logic.
KNOWN_HOSTS: frozenset[str] = frozenset(
    {
        "Ken Jennings",
        "Ryan Seacrest",
        "Mayim Bialik",
        "Alex Trebek",
        "Buzzy Cohen",
    }
)


class Settings(BaseSettings):
    db_path: str = Field(
        default="trebek.db",
        validation_alias=AliasChoices("db_path", "database_path", "DB_PATH", "DATABASE_PATH"),
        description="Path to the SQLite database",
    )
    output_dir: str = Field(default="gpu_outputs", description="Directory to store intermediate pipeline outputs")
    input_dir: str = Field(default="input_videos", description="Directory to poll for new video files")
    gemini_api_key: str = Field(default="", description="GCP / Gemini API Key")
    log_level: str = Field(default="INFO", description="Logging level")

    mock_llm: bool = Field(default=False, description="Enable zero-cost offline mock mode using synthetic LLM fixtures")
    enable_podium_sniping: bool = Field(default=False, description="Enable Pass 3 multimodal podium lockout sniping")

    def require_gemini_api_key(self) -> str:
        """Validates that GEMINI_API_KEY is set. Call this at pipeline startup,
        not at import time, so that CLI commands like scan/stats still work."""
        if not self.gemini_api_key:
            if self.mock_llm:
                return "mock-gemini-key"
            raise ValueError(
                "GEMINI_API_KEY is required. Get a free key at https://aistudio.google.com/apikey "
                "and set it in your .env file or environment."
            )
        return self.gemini_api_key

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {sorted(valid_levels)}, got '{v}'")
        return v.upper()

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
        valid_types = {"float16", "float32", "int8", "int8_float16"}
        if v not in valid_types:
            raise ValueError(f"whisper_compute_type must be one of {sorted(valid_types)}")
        return v

    @field_validator("whisper_batch_size")
    @classmethod
    def validate_whisper_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("whisper_batch_size must be > 0")
        return v

    # Concurrency constraints
    llm_concurrency: int = Field(default=2, description="Number of concurrent episodes processed in LLM extraction")

    @field_validator("llm_concurrency")
    @classmethod
    def validate_llm_concurrency(cls, v: int) -> int:
        if v < 1 or v > 8:
            raise ValueError("llm_concurrency must be between 1 and 8")
        return v

    # Preflight & Hardware options
    hf_token: str = Field(default="", description="Hugging Face access token for pyannote speaker diarization")
    device: str = Field(default="auto", description="Device for WhisperX transcription ('auto', 'cuda', 'cpu')")
    allow_cpu: bool = Field(default=False, description="Allow running WhisperX on CPU if CUDA is unavailable")

    def model_post_init(self, __context: Any) -> None:
        import os

        if self.hf_token and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = self.hf_token

        mock_env = os.environ.get("TREBEK_MOCK_LLM", "").lower() or os.environ.get("MOCK_LLM", "").lower()
        if mock_env in ("1", "true", "yes", "on"):
            self.mock_llm = True

        # Validate that db_path is not an existing directory
        if os.path.isdir(self.db_path):
            raise ValueError(
                f"Database path '{self.db_path}' is a directory, not a file. "
                "If running with Docker, ensure a file path is specified "
                "(e.g. ./data:/app/data with DB_PATH=/app/data/trebek.db)."
            )
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except OSError:
                pass

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


try:
    settings = Settings()
except ValueError:
    # Allow importing trebek.config even if the default db_path on disk is a directory,
    # so diagnostic tools like `trebek doctor` can run and report the collision cleanly.
    settings = Settings.model_construct(
        db_path="trebek.db",
        output_dir="gpu_outputs",
        input_dir="input_videos",
        gemini_api_key="",
        log_level="INFO",
        mock_llm=False,
        enable_podium_sniping=False,
    )
