"""
Tests for performance and throughput enhancements:
1. GPU worker diarization model caching & warm start
2. Multimodal clip collision avoidance & usage aggregation
3. LLM worker interview slice cleanup & non-blocking I/O
4. CLI search subcommand execution
"""

import os
import sys
import json
import sqlite3
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

from trebek.gpu.worker import (
    reset_gpu_models,
)
import trebek.gpu.worker as gpu_worker_mod
from trebek.schemas import Episode, Clue, Contestant, BuzzAttempt, FinalJep
from trebek.llm.pass3_multimodal import execute_pass_3_multimodal_augmentation


class TestGPUWorkerWarmCaching:
    """Warm worker model caching in GPU worker."""

    def test_reset_gpu_models(self) -> None:
        gpu_worker_mod._whisperx_model = "mock_whisper"
        gpu_worker_mod._whisperx_align_model = "mock_align"
        gpu_worker_mod._whisperx_align_metadata = {"lang": "en"}
        gpu_worker_mod._whisperx_diarize_model = "mock_diarize"

        reset_gpu_models()

        assert gpu_worker_mod._whisperx_model is None
        assert gpu_worker_mod._whisperx_align_model is None
        assert gpu_worker_mod._whisperx_align_metadata is None
        assert gpu_worker_mod._whisperx_diarize_model is None

    def test_diarization_model_cached_across_invocations(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")
        reset_gpu_models()

        mock_diarize_pipeline = MagicMock()
        mock_diarize_instance = MagicMock()
        mock_diarize_instance.return_value = []
        mock_diarize_pipeline.return_value = mock_diarize_instance

        # Mock torch and whisperx module imports
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_audio.return_value = MagicMock()
        mock_whisperx.load_align_model.return_value = (MagicMock(), {})
        mock_whisperx.align.return_value = {"segments": []}
        mock_whisperx.assign_word_speakers.return_value = {"segments": []}

        # Mock subprocess.run for ffmpeg
        mock_proc_run = MagicMock()
        mock_proc_run.returncode = 0
        mock_proc_run.stderr = ""

        video_path = str(tmp_path / "test.mp4")
        with open(video_path, "wb") as f:
            f.write(b"fake_video_data")

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "whisperx": mock_whisperx,
                "whisperx.diarize": MagicMock(DiarizationPipeline=mock_diarize_pipeline),
            },
        ):
            with patch("subprocess.run", return_value=mock_proc_run):
                # Call 1: Cold start
                out1, _, _ = gpu_worker_mod.gpu_worker_task(
                    video_path, str(tmp_path), batch_size=4, compute_type="float32", device="cpu"
                )
                assert os.path.exists(out1)
                assert mock_diarize_pipeline.call_count == 1
                assert gpu_worker_mod._whisperx_diarize_model is mock_diarize_instance

                # Call 2: Warm start — DiarizationPipeline should NOT be instantiated again
                out2, _, _ = gpu_worker_mod.gpu_worker_task(
                    video_path, str(tmp_path), batch_size=4, compute_type="float32", device="cpu"
                )
                assert os.path.exists(out2)
                assert mock_diarize_pipeline.call_count == 1  # Still 1, reused!

        reset_gpu_models()


class TestMultimodalTemporalSniping:
    """Multimodal Pass 3 concurrency and collision safety."""

    @pytest.mark.asyncio
    async def test_clip_naming_and_usage_aggregation(self, tmp_path: Path) -> None:
        episode = Episode(
            episode_date="2024-02-01",
            host_name="Ken Jennings",
            is_tournament=False,
            contestants=[
                Contestant(
                    name="Alice",
                    podium_position=1,
                    occupational_category="Teacher",
                    is_returning_champion=False,
                    description="Contestant 1",
                ),
                Contestant(
                    name="Bob",
                    podium_position=2,
                    occupational_category="Engineer",
                    is_returning_champion=True,
                    description="Contestant 2",
                ),
                Contestant(
                    name="Carol",
                    podium_position=3,
                    occupational_category="Writer",
                    is_returning_champion=False,
                    description="Contestant 3",
                ),
            ],
            clues=[
                Clue(
                    round="J!",
                    category="ART",
                    board_row=1,
                    board_col=1,
                    selection_order=1,
                    is_daily_double=False,
                    host_start_timestamp_ms=10000.0,
                    host_finish_timestamp_ms=12500.0,
                    clue_syllable_count=6,
                    clue_text="A painting by Monet",
                    correct_response="Water Lilies",
                    requires_visual_context=True,
                    attempts=[
                        BuzzAttempt(
                            attempt_order=1,
                            speaker="Alice",
                            response_given="Water Lilies",
                            is_correct=True,
                            buzz_timestamp_ms=12800.0,
                            response_start_timestamp_ms=13000.0,
                            is_lockout_inferred=False,
                        )
                    ],
                ),
                Clue(
                    round="J!",
                    category="MAPS",
                    board_row=2,
                    board_col=1,
                    selection_order=2,
                    is_daily_double=False,
                    host_start_timestamp_ms=22000.0,
                    host_finish_timestamp_ms=25000.0,
                    clue_syllable_count=5,
                    clue_text="This island nation",
                    correct_response="Iceland",
                    requires_visual_context=True,
                    attempts=[
                        BuzzAttempt(
                            attempt_order=1,
                            speaker="Bob",
                            response_given="Iceland",
                            is_correct=True,
                            buzz_timestamp_ms=25400.0,
                            response_start_timestamp_ms=25600.0,
                            is_lockout_inferred=False,
                        )
                    ],
                ),
            ],
            final_jep=FinalJep(category="FINAL", clue_text="Clue", correct_response="Answer", wagers_and_responses=[]),
            score_adjustments=[],
        )

        mock_client = MagicMock()
        mock_client.upload_file = AsyncMock(return_value=MagicMock(name="files/clip123"))
        mock_client.delete_file = AsyncMock()

        # Mock file active state
        mock_file_info = MagicMock()
        mock_file_info.state.name = "ACTIVE"
        mock_client.client.files.get.return_value = mock_file_info

        mock_client.generate_content = AsyncMock(
            return_value=(
                MagicMock(text="0.45"),
                {"input_tokens": 100.0, "output_tokens": 20.0, "cost_usd": 0.005, "latency_ms": 200.0},
            )
        )

        extracted_clips: list[str] = []

        async def mock_subprocess_exec(*args, **kwargs):
            clip_path = args[10]
            extracted_clips.append(clip_path)
            Path(clip_path).touch()
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            return mock_proc

        with patch("trebek.llm.pass3_multimodal._get_client", return_value=mock_client):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess_exec):
                aug_ep, usage = await execute_pass_3_multimodal_augmentation(
                    episode, "/fake/ep_multimodal_test.mp4", str(tmp_path), episode_id="ep_multimodal_test"
                )

        # 1. Verify clips contain episode_id and are unique
        assert len(extracted_clips) == 2
        assert "ep_multimodal_test" in extracted_clips[0]
        assert "ep_multimodal_test" in extracted_clips[1]
        assert extracted_clips[0] != extracted_clips[1]

        # 2. Verify all clips were cleaned up from disk
        for clip in extracted_clips:
            assert not os.path.exists(clip)

        # 3. Verify files were deleted from Gemini
        assert mock_client.delete_file.call_count == 2

        # 4. Verify usage was aggregated across both clues
        assert usage["input_tokens"] == 200.0
        assert usage["output_tokens"] == 40.0
        assert usage["cost_usd"] == 0.01


class TestCLISearchSubcommand:
    """CLI search command output."""

    def test_cli_search_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        db_path = str(tmp_path / "search_cli.db")
        schema_path = Path(__file__).resolve().parents[2] / "trebek" / "schema.sql"

        with sqlite3.connect(db_path) as conn:
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
            conn.execute(
                "INSERT INTO episodes (episode_id, air_date, host_name, is_tournament) VALUES (?, ?, ?, ?)",
                ("ep101", "2024-03-10", "Ken Jennings", False),
            )
            conn.execute(
                "INSERT INTO clues (clue_id, episode_id, round, category, board_row, board_col, selection_order, clue_text, correct_response, is_verified) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "c101",
                    "ep101",
                    "J!",
                    "ASTRONOMY",
                    1,
                    1,
                    1,
                    "The Red Planet named after the Roman god of war",
                    "Mars",
                    True,
                ),
            )
            conn.commit()

        from trebek.config import settings

        monkeypatch.setattr(settings, "db_path", db_path)

        # Run CLI with search subcommand and --json
        monkeypatch.setattr(sys, "argv", ["trebek", "search", "Planet", "--json"])
        from trebek.cli import main

        main()
        captured = capsys.readouterr()
        # Find JSON array in captured output
        start = captured.out.find("[")
        end = captured.out.rfind("]") + 1
        data = json.loads(captured.out[start:end])
        assert len(data) == 1
        assert data[0]["clue_id"] == "c101"
        assert data[0]["correct_response"] == "Mars"

    def test_cli_search_table_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        db_path = str(tmp_path / "search_cli2.db")
        schema_path = Path(__file__).resolve().parents[2] / "trebek" / "schema.sql"

        with sqlite3.connect(db_path) as conn:
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
            conn.execute(
                "INSERT INTO episodes (episode_id, air_date, host_name, is_tournament) VALUES (?, ?, ?, ?)",
                ("ep102", "2024-03-11", "Ken Jennings", False),
            )
            conn.execute(
                "INSERT INTO clues (clue_id, episode_id, round, category, board_row, board_col, selection_order, clue_text, correct_response, is_verified) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "c102",
                    "ep102",
                    "J!",
                    "POETRY",
                    1,
                    1,
                    1,
                    "He wrote 'The Raven' and 'Annabel Lee'",
                    "Edgar Allan Poe",
                    True,
                ),
            )
            conn.commit()

        from trebek.config import settings

        monkeypatch.setattr(settings, "db_path", db_path)

        monkeypatch.setattr(sys, "argv", ["trebek", "search", "Raven"])
        from trebek.cli import main

        main()
        captured = capsys.readouterr()
        # Table is rendered to stderr via Rich console
        combined_output = captured.out + captured.err
        assert 'Search Results for "Raven"' in combined_output
        assert "Edgar Allan" in combined_output
        assert "Poe" in combined_output

    @pytest.mark.asyncio
    async def test_search_on_uninitialized_database(self, tmp_path: Path) -> None:
        """Searching before database or clues table exists returns empty list without error."""
        from trebek.database.writer import DatabaseWriter

        db_path = str(tmp_path / "empty.db")
        writer = DatabaseWriter(db_path)
        await writer.start()
        try:
            results = await writer.search_clues("query")
            assert results == []
        finally:
            await writer.stop()


class TestGpuWorkerResilience:
    """Tests for GPU model caching, device tracking, and OOM exception handling."""

    def test_clear_gpu_cache_and_device_tracking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trebek.gpu import clear_gpu_cache, reset_gpu_models
        import trebek.gpu.worker as worker

        worker._whisperx_model = "fake_model"
        worker._cached_device = "cuda"
        worker._cached_compute_type = "float16"

        clear_gpu_cache()
        assert worker._whisperx_model is None
        assert worker._cached_device is None
        assert worker._cached_compute_type is None

        worker._whisperx_model = "fake_model_2"
        reset_gpu_models()
        assert worker._whisperx_model is None

    def test_gpu_diarization_oom_reraised(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Verify that an OutOfMemoryError in pyannote diarization is reraised to trigger pool restart."""
        import trebek.gpu.worker as worker

        monkeypatch.setenv("HF_TOKEN", "fake-token")

        class FakeCUDAError(RuntimeError):
            pass

        FakeCUDAError.__name__ = "OutOfMemoryError"

        def fake_diarize(*args: Any, **kwargs: Any) -> Any:
            raise FakeCUDAError("CUDA out of memory in diarization")

        class FakeWhisperXModel:
            def transcribe(self, *args: Any, **kwargs: Any) -> Any:
                return {"segments": [{"text": "Sample"}]}

        monkeypatch.setattr(worker, "reset_gpu_models", lambda: None)
        worker._whisperx_model = FakeWhisperXModel()

        # Mock subprocess run for ffmpeg
        class FakeProc:
            returncode = 0
            stderr = ""

        import subprocess

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeProc())

        import sys

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        fake_whisperx = MagicMock()
        fake_whisperx.load_audio.return_value = []
        fake_whisperx.load_align_model.return_value = (MagicMock(), {})
        fake_whisperx.align.return_value = {"segments": [{"text": "Sample"}]}
        fake_whisperx.diarize.DiarizationPipeline.side_effect = fake_diarize

        monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
        monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_whisperx.diarize)

        with pytest.raises(MemoryError, match="CUDA OOM"):
            worker.gpu_worker_task("fake.mp4", str(tmp_path), device="cuda")
