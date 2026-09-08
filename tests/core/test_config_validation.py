"""
Tests for configuration validation — log levels, GPU constraints,
GEMINI_API_KEY runtime validation, and Settings behavior.
"""

import pytest

from trebek.config import Settings, resolve_model_name, MODEL_FLASH, MODEL_PRO


class TestLogLevelValidation:
    """log_level field validator tests."""

    def test_valid_log_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            s = Settings(log_level=level, gemini_api_key="test-key")
            assert s.log_level == level

    def test_log_level_case_insensitive(self) -> None:
        s = Settings(log_level="debug", gemini_api_key="test-key")
        assert s.log_level == "DEBUG"

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(Exception, match="log_level must be one of"):
            Settings(log_level="TRACE", gemini_api_key="test-key")


class TestGpuVramValidation:
    """GPU VRAM constraint tests."""

    def test_valid_vram_range(self) -> None:
        for gb in (4, 8, 16, 24):
            s = Settings(gpu_vram_target_gb=gb, gemini_api_key="test-key")
            assert s.gpu_vram_target_gb == gb

    def test_vram_below_minimum_raises(self) -> None:
        with pytest.raises(Exception, match="gpu_vram_target_gb must be >= 4"):
            Settings(gpu_vram_target_gb=3, gemini_api_key="test-key")

    def test_vram_above_maximum_raises(self) -> None:
        with pytest.raises(Exception, match="gpu_vram_target_gb must be.*<= 24"):
            Settings(gpu_vram_target_gb=25, gemini_api_key="test-key")


class TestWhisperBatchSizeValidation:
    """Whisper batch size constraint tests."""

    def test_valid_batch_sizes(self) -> None:
        for bs in (1, 4, 8, 16):
            s = Settings(whisper_batch_size=bs, gemini_api_key="test-key")
            assert s.whisper_batch_size == bs

    def test_zero_batch_size_raises(self) -> None:
        with pytest.raises(Exception, match="whisper_batch_size must be > 0"):
            Settings(whisper_batch_size=0, gemini_api_key="test-key")

    def test_negative_batch_size_raises(self) -> None:
        with pytest.raises(Exception, match="whisper_batch_size must be > 0"):
            Settings(whisper_batch_size=-1, gemini_api_key="test-key")


class TestWhisperComputeTypeValidation:
    """Compute type constraint tests."""

    def test_valid_compute_types(self) -> None:
        for ct in ("float16", "float32", "int8", "int8_float16"):
            s = Settings(whisper_compute_type=ct, gemini_api_key="test-key")
            assert s.whisper_compute_type == ct

    def test_invalid_compute_type_raises(self) -> None:
        with pytest.raises(Exception, match="whisper_compute_type must be"):
            Settings(whisper_compute_type="invalid_type", gemini_api_key="test-key")


class TestGeminiApiKeyValidation:
    """Runtime API key validation tests."""

    def test_require_gemini_api_key_with_valid_key(self) -> None:
        s = Settings(gemini_api_key="test-key-123")
        result = s.require_gemini_api_key()
        assert result == "test-key-123"

    def test_require_gemini_api_key_raises_when_empty(self) -> None:
        s = Settings(gemini_api_key="")
        with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
            s.require_gemini_api_key()

    def test_settings_can_be_created_without_api_key(self) -> None:
        """Settings should instantiate even without API key (for scan/stats commands)."""
        s = Settings(gemini_api_key="")
        assert s.gemini_api_key == ""


class TestSettingsDefaults:
    """Default value tests."""

    def test_default_db_path(self) -> None:
        s = Settings(gemini_api_key="test-key")
        assert s.db_path == "trebek.db"

    def test_default_output_dir(self) -> None:
        s = Settings(gemini_api_key="test-key")
        assert s.output_dir == "gpu_outputs"

    def test_default_input_dir(self) -> None:
        s = Settings(gemini_api_key="test-key")
        assert s.input_dir == "input_videos"

    def test_default_llm_concurrency(self) -> None:
        s = Settings(gemini_api_key="test-key")
        assert s.llm_concurrency == 2


class TestLlmConcurrencyValidation:
    """LLM concurrency constraint tests."""

    def test_valid_concurrency(self) -> None:
        for c in (1, 2, 4, 8):
            s = Settings(llm_concurrency=c, gemini_api_key="test-key")
            assert s.llm_concurrency == c

    def test_concurrency_below_minimum_raises(self) -> None:
        with pytest.raises(Exception, match="llm_concurrency must be between 1 and 8"):
            Settings(llm_concurrency=0, gemini_api_key="test-key")

    def test_concurrency_above_maximum_raises(self) -> None:
        with pytest.raises(Exception, match="llm_concurrency must be between 1 and 8"):
            Settings(llm_concurrency=9, gemini_api_key="test-key")


class TestModelResolution:
    """Tests for model alias resolution and canonical Gemini model identification."""

    def test_resolve_flash_aliases_to_gemini_38(self) -> None:
        assert resolve_model_name("flash") == MODEL_FLASH
        assert resolve_model_name("gemini-flash") == MODEL_FLASH
        assert resolve_model_name("gemini_flash") == MODEL_FLASH
        assert resolve_model_name("GEMINI-FLASH") == MODEL_FLASH
        assert resolve_model_name("flash38") == "gemini-3.8-flash"
        assert resolve_model_name("gemini-3.8-flash") == "gemini-3.8-flash"

    def test_resolve_provider_prefixed_strings(self) -> None:
        assert resolve_model_name("models/gemini-flash") == "gemini-3.8-flash"
        assert resolve_model_name("models/gemini-3.8-flash") == "gemini-3.8-flash"
        assert resolve_model_name("gemini/gemini-flash") == "gemini-3.8-flash"
        assert resolve_model_name("google/gemini-flash") == "gemini-3.8-flash"

    def test_resolve_pro_aliases(self) -> None:
        assert resolve_model_name("pro") == "gemini-3.1-pro-preview"
        assert resolve_model_name("gemini-pro") == "gemini-3.1-pro-preview"
        assert resolve_model_name("gemini_pro") == "gemini-3.1-pro-preview"
        assert resolve_model_name("gemini/pro") == "gemini-3.1-pro-preview"

    def test_resolve_flash_lite(self) -> None:
        assert resolve_model_name("flash-lite") == "gemini-3.1-flash-lite-preview"
        assert resolve_model_name("gemini-flash-lite") == "gemini-3.1-flash-lite-preview"

    def test_resolve_unknown_explicit_model_passes_through(self) -> None:
        assert resolve_model_name("gemini-2.5-flash") == "gemini-2.5-flash"
        assert resolve_model_name("gemini-future-super-model") == "gemini-future-super-model"

    def test_resolve_empty_or_none_defaults_to_pro(self) -> None:
        assert resolve_model_name(None) == MODEL_PRO
        assert resolve_model_name("") == MODEL_PRO
        assert resolve_model_name("   ") == MODEL_PRO
