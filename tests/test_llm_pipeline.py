import pytest
from unittest.mock import patch, AsyncMock
from src.llm_pipeline import execute_pass_1_speaker_anchoring


@pytest.mark.asyncio
async def test_execute_pass_1_speaker_anchoring() -> None:
    with patch("src.llm_pipeline.ai_client") as mock_client:
        mock_client.generate_content = AsyncMock(return_value='{"SPEAKER_00": "Ken"}')

        mapping = await execute_pass_1_speaker_anchoring("test segment")
        assert mapping == {"SPEAKER_00": "Ken"}
        mock_client.generate_content.assert_awaited_once()
