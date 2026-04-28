import pytest
import os
from pathlib import Path
from unittest.mock import patch
from typing import Any
from trebek.gpu import GPUOrchestrator


@pytest.fixture(autouse=True)
def mock_path(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "mock_bin"))
    monkeypatch.setenv("PATH", f"{mock_bin}:{os.environ.get('PATH', '')}")


@pytest.mark.asyncio
async def test_gpu_orchestrator_execution(tmp_path: Path) -> None:
    output_dir = str(tmp_path / "gpu_outputs")
    orchestrator = GPUOrchestrator(output_dir=output_dir, max_workers=1)

    try:
        expected_path = os.path.join(output_dir, "audio_mock.json.gz")
        with open(expected_path, "w") as f:
            f.write("{}")

        with patch.object(orchestrator.executor, "submit") as mock_submit:
            import concurrent.futures

            fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
            fut.set_result((expected_path, 1500.5, 85.0))
            mock_submit.return_value = fut

            result_path, peak_vram, avg_util = await orchestrator.execute_gpu_work("/mock/video.mp4")

        assert os.path.exists(result_path)
        assert result_path.endswith(".json.gz")
        assert isinstance(peak_vram, float)
        assert isinstance(avg_util, float)
    finally:
        orchestrator.shutdown()
