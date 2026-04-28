import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_execute_pass_1_speaker_anchoring() -> None:
    with patch("trebek.llm.pass1_anchoring._get_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.text = '{"SPEAKER_00": "Ken"}'
        mock_client.generate_content = AsyncMock(return_value=(mock_response, {}))
        mock_client.upload_file = AsyncMock()
        mock_client.upload_file.return_value.name = "mock_file"
        mock_client.delete_file = AsyncMock()

        from trebek.llm import execute_pass_1_speaker_anchoring

        mapping, usage = await execute_pass_1_speaker_anchoring("test segment")
        assert mapping == {"SPEAKER_00": "Ken"}
        assert usage == {}
        mock_client.generate_content.assert_awaited_once()
