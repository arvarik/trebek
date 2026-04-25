import pytest
import os
from pathlib import Path
from src.gpu_orchestrator import GPUOrchestrator

@pytest.fixture(autouse=True)
def mock_path(monkeypatch):
    mock_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "mock_bin"))
    monkeypatch.setenv("PATH", f"{mock_bin}:{os.environ.get('PATH', '')}")

@pytest.mark.asyncio
async def test_gpu_orchestrator_execution(tmp_path: Path) -> None:
    output_dir = str(tmp_path / "gpu_outputs")
    orchestrator = GPUOrchestrator(output_dir=output_dir, max_workers=1)

    try:
        # Mock video filepath
        result_path = await orchestrator.execute_gpu_work("/mock/video.mp4")
        assert os.path.exists(result_path)
        assert result_path.endswith(".json.gz")
    finally:
        orchestrator.shutdown()
