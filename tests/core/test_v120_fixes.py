"""
Unit tests verifying all bug fixes and improvements implemented for v1.2.0.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trebek.analysis.embeddings import enrich_clues_with_embeddings
from trebek.cli import build_parser
from trebek.config import Settings
from trebek.gpu import reset_warm_models
from trebek.gpu.worker import reset_gpu_models
from trebek.llm.pass3_multimodal import (
    _wait_for_file_active,
    execute_pass_3_multimodal_augmentation,
)
from trebek.schemas import BuzzAttempt, Clue, Episode, FinalJep


@pytest.mark.asyncio
async def test_wait_for_file_active_timeout_returns_false() -> None:
    """When a file never reaches ACTIVE within the polling window, it must return False."""
    client = SimpleNamespace(
        client=SimpleNamespace(
            files=SimpleNamespace(get=lambda name: SimpleNamespace(state=SimpleNamespace(name="PROCESSING")))
        )
    )
    mock_file = SimpleNamespace(name="test-file")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await _wait_for_file_active(client, mock_file)
    assert result is False


@pytest.mark.asyncio
async def test_pass3_mock_handles_none_host_finish_timestamp() -> None:
    """Pass 3 mock mode should not raise TypeError when host_finish_timestamp_ms is None."""
    clue = Clue(
        round="J!",
        category="SCIENCE",
        board_row=1,
        board_col=1,
        selection_order=1,
        is_daily_double=False,
        clue_text="A sample clue.",
        correct_response="What is science?",
        host_start_timestamp_ms=0.0,
        host_finish_timestamp_ms=0.0,
        clue_syllable_count=4,
        requires_visual_context=True,
        attempts=[
            BuzzAttempt(
                attempt_order=1,
                speaker="Contestant 1",
                buzz_timestamp_ms=100.0,
                response_start_timestamp_ms=200.0,
                is_lockout_inferred=False,
                response_given="What is science?",
                is_correct=True,
            )
        ],
    )
    # Set host_finish_timestamp_ms to None directly on object
    clue.host_finish_timestamp_ms = None  # type: ignore[assignment]

    episode = Episode(
        episode_date="2024-01-01",
        host_name="Ken Jennings",
        is_tournament=False,
        contestants=[],
        clues=[clue],
        final_jep=FinalJep(category="FINAL", clue_text="Final clue", correct_response="", wagers_and_responses=[]),
        score_adjustments=[],
    )

    with patch("trebek.config.settings.mock_llm", True):
        res_ep, usage = await execute_pass_3_multimodal_augmentation(
            episode=episode,
            video_filepath="/fake/video.mp4",
            output_dir="/fake/dir",
        )
    assert res_ep.clues[0].visual_context_description is not None
    # Podium light timestamp should remain unset since host_finish_timestamp_ms was None
    assert res_ep.clues[0].attempts[0].podium_light_timestamp_ms is None


def test_settings_database_path_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """DATABASE_PATH environment variable should be recognized by Settings."""
    monkeypatch.setenv("DATABASE_PATH", "/custom/path/trebek.db")
    s = Settings()
    assert s.db_path == "/custom/path/trebek.db"


def test_doctor_cli_mock_llm_flag() -> None:
    """trebek doctor --mock-llm flag should be supported by argument parser."""
    parser = build_parser()
    args = parser.parse_args(["doctor", "--mock-llm"])
    assert args.command == "doctor"
    assert args.mock_llm is True


def test_gpu_reset_warm_models_alias() -> None:
    """reset_warm_models alias should be identical to reset_gpu_models and clear caches."""
    assert reset_warm_models is reset_gpu_models

    import trebek.gpu.worker as gw

    gw._cached_hf_token = "fake-token"
    gw._cached_device = "cuda"
    reset_warm_models()
    assert gw._cached_hf_token is None
    assert gw._cached_device is None


@pytest.mark.asyncio
async def test_enrich_clues_with_category_prefix_and_final_jep() -> None:
    """enrich_clues_with_embeddings prepends category to clue_text and supports FinalJep."""
    clue = Clue(
        round="J!",
        category="ASTRONOMY",
        board_row=1,
        board_col=1,
        selection_order=1,
        is_daily_double=False,
        clue_text="The closest star to Earth.",
        correct_response="The Sun",
        host_start_timestamp_ms=1000.0,
        host_finish_timestamp_ms=3000.0,
        clue_syllable_count=7,
        requires_visual_context=False,
    )
    fj = FinalJep(
        category="LITERATURE",
        clue_text="This novel opens with Call me Ishmael.",
        correct_response="Moby-Dick",
        wagers_and_responses=[],
    )

    embedded_inputs: list[list[str]] = []

    async def mock_embed(texts: list[str]) -> list[list[float]]:
        embedded_inputs.append(texts)
        return [[0.1 * (i + 1)] * 768 for i in range(len(texts))]

    mock_client = MagicMock()
    mock_client.embed_content = AsyncMock(side_effect=mock_embed)

    items: list[Any] = [clue, fj]
    await enrich_clues_with_embeddings(items, client=mock_client)

    assert len(embedded_inputs) == 1
    passed_texts = embedded_inputs[0]
    # Check clue text had category prepended
    assert passed_texts[0] == "ASTRONOMY. The closest star to Earth."
    assert passed_texts[1] == "LITERATURE. This novel opens with Call me Ishmael."
    assert passed_texts[2] == "The Sun"
    assert passed_texts[3] == "Moby-Dick"

    assert clue.clue_embedding is not None
    assert fj.clue_embedding is not None
    assert fj.response_embedding is not None
    assert fj.semantic_lateral_distance is not None
