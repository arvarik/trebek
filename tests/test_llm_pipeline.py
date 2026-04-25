import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_execute_pass_1_speaker_anchoring() -> None:
    with patch("trebek.llm_pipeline._get_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.generate_content = AsyncMock(return_value=('{"SPEAKER_00": "Ken"}', {}))

        from trebek.llm_pipeline import execute_pass_1_speaker_anchoring

        mapping, usage = await execute_pass_1_speaker_anchoring("test segment")
        assert mapping == {"SPEAKER_00": "Ken"}
        assert usage == {}
        mock_client.generate_content.assert_awaited_once()
