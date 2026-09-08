"""
Tests for Developer Onboarding, Docker Setup, Doctor Command, and Mock LLM Mode.

Covers:
1. Docker volume and database path configuration:
   - Automatic parent directory creation for db_path (e.g. data/trebek.db)
   - Prevention of directory collision (ValueError if db_path is a directory)
   - Configuration via TREBEK_MOCK_LLM and MOCK_LLM environment variables
2. Pre-flight Environment Doctor command (`trebek doctor`):
   - Diagnostic checks (Python, FFmpeg, Hardware, SQLite, WAL locks, Disk space, API keys)
   - Exit code handling (0 for pass/warn, 1 for critical failure)
   - JSON output serialization
   - CLI parser subcommand and flags (--check-api, --json)
3. Zero-cost offline mock LLM mode:
   - Synthetic responses for Pass 1 (speaker anchoring)
   - Synthetic responses for Pass 2 (skeleton, metadata, clues)
   - Pass 3 multimodal bypass in mock mode
   - Verification pass bypass in mock mode
   - Zero-token and zero-cost telemetry tracking
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trebek.cli import build_parser
from trebek.config import Settings, settings
from trebek.llm.client import GeminiClient
from trebek.llm.mock import generate_mock_llm_response
from trebek.llm.pass1_anchoring import execute_pass_1_speaker_anchoring
from trebek.llm.pass3_multimodal import execute_pass_3_multimodal_augmentation
from trebek.llm.schemas import (
    EpisodeSkeleton,
    PartialClues,
    PartialEpisodeMeta,
    create_dynamic_clue_schema,
)
from trebek.llm.verify import verify_and_correct_clues, verify_final_jeopardy
from trebek.schemas import Clue, Episode, FinalJep
from trebek.ui.doctor import (
    DiagnosticCheck,
    DoctorReport,
    _check_binary,
    _test_sqlite_wal_locks,
    render_doctor_results,
    run_diagnostics,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Docker Volume Mount & Database Directory Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestDockerAndDatabaseSetup:
    """Verifies that database directories are auto-created and directory collisions are prevented."""

    def test_db_path_auto_creates_parent_directory(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "custom_data" / "nested"
        db_file = str(sub_dir / "trebek.db")

        assert not sub_dir.exists()
        s = Settings(db_path=db_file)
        assert sub_dir.exists()
        assert s.db_path == db_file

    def test_db_path_raises_if_it_is_a_directory(self, tmp_path: Path) -> None:
        dir_as_db = tmp_path / "colliding_trebek.db"
        dir_as_db.mkdir()

        with pytest.raises(ValueError, match="is a directory, not a file"):
            Settings(db_path=str(dir_as_db))

    def test_mock_llm_env_var_activates_setting(self) -> None:
        with patch.dict(os.environ, {"TREBEK_MOCK_LLM": "1"}):
            s = Settings()
            assert s.mock_llm is True

        with patch.dict(os.environ, {"MOCK_LLM": "true"}):
            s2 = Settings()
            assert s2.mock_llm is True

    def test_require_gemini_api_key_returns_mock_when_enabled(self) -> None:
        s = Settings(gemini_api_key="", mock_llm=True)
        key = s.require_gemini_api_key()
        assert key == "mock-gemini-key"

    def test_require_gemini_api_key_raises_when_empty_and_not_mock(self) -> None:
        s = Settings(gemini_api_key="", mock_llm=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
            s.require_gemini_api_key()


# ═════════════════════════════════════════════════════════════════════════════
# 2. Pre-flight Environment Doctor Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestDoctorCommand:
    """Verifies `trebek doctor` diagnostics, status checks, and rendering."""

    def test_check_binary_found(self) -> None:
        ok, detail = _check_binary("python3")
        assert ok is True
        assert "found" in detail or "Python" in detail

    def test_check_binary_not_found(self) -> None:
        ok, detail = _check_binary("non_existent_binary_xyz_123")
        assert ok is False
        assert "not found" in detail

    def test_sqlite_wal_locks_check(self, tmp_path: Path) -> None:
        ok, detail = _test_sqlite_wal_locks(str(tmp_path))
        assert ok is True
        assert "verified" in detail
        # Verify test db was cleaned up
        assert not (tmp_path / ".trebek_doctor_wal_test.db").exists()

    def test_run_diagnostics_report_structure(self, tmp_path: Path) -> None:
        s = Settings(db_path=str(tmp_path / "trebek.db"), mock_llm=True)
        report = run_diagnostics(s, check_api=False)

        assert isinstance(report, DoctorReport)
        assert isinstance(report.checks, list)
        assert len(report.checks) >= 10
        assert "pass" in report.summary
        assert "warn" in report.summary
        assert "fail" in report.summary

        categories = {c.category for c in report.checks}
        assert "System & Runtime" in categories
        assert "External Binaries" in categories
        assert "Hardware & ML" in categories
        assert "API & Authentication" in categories
        assert "Storage & Database" in categories
        assert "Pipeline Directories" in categories

    def test_run_diagnostics_fails_on_old_python(self, tmp_path: Path) -> None:
        s = Settings(db_path=str(tmp_path / "trebek.db"), mock_llm=True)
        with patch.object(sys, "version_info", (3, 9, 0)):
            report = run_diagnostics(s)
            py_check = next(c for c in report.checks if c.component == "Python Version")
            assert py_check.status == "FAIL"
            assert report.success is False

    def test_run_diagnostics_fails_on_missing_ffmpeg(self, tmp_path: Path) -> None:
        s = Settings(db_path=str(tmp_path / "trebek.db"), mock_llm=True)
        with patch("shutil.which", return_value=None):
            report = run_diagnostics(s)
            ffmpeg_check = next(c for c in report.checks if c.component == "FFmpeg")
            assert ffmpeg_check.status == "FAIL"
            assert report.success is False

    def test_run_diagnostics_passes_with_mock_llm_and_no_key(self, tmp_path: Path) -> None:
        s = Settings(db_path=str(tmp_path / "trebek.db"), gemini_api_key="", mock_llm=True)
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            report = run_diagnostics(s)
            api_check = next(c for c in report.checks if c.component == "Gemini API Key")
            assert api_check.status == "PASS"
            assert "mock mode" in api_check.detail

    def test_doctor_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        checks = [
            DiagnosticCheck(
                category="System & Runtime",
                component="Python Version",
                status="PASS",
                detail="3.11.0",
            )
        ]
        report = DoctorReport(
            success=True,
            summary={"pass": 1, "warn": 0, "fail": 0, "total": 1},
            checks=checks,
        )
        exit_code = render_doctor_results(report, as_json=True)
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["success"] is True
        assert data["summary"]["pass"] == 1
        assert len(data["checks"]) == 1
        assert data["checks"][0]["component"] == "Python Version"

    def test_doctor_terminal_rendering_with_remediation(self) -> None:
        checks = [
            DiagnosticCheck(
                category="External Binaries",
                component="FFmpeg",
                status="FAIL",
                detail="not found",
                remediation="Install ffmpeg via package manager.",
            ),
            DiagnosticCheck(
                category="API & Authentication",
                component="Hugging Face Token",
                status="WARN",
                detail="unset",
                remediation="Set HF_TOKEN in .env.",
            ),
        ]
        report = DoctorReport(
            success=False,
            summary={"pass": 0, "warn": 1, "fail": 1, "total": 2},
            checks=checks,
        )
        exit_code = render_doctor_results(report, as_json=False)
        assert exit_code == 1

    def test_cli_parser_doctor_and_doc_alias(self) -> None:
        parser = build_parser()

        args_doc = parser.parse_args(["doctor", "--check-api", "--json"])
        assert args_doc.command == "doctor"
        assert args_doc.check_api is True
        assert args_doc.json is True

        args_alias = parser.parse_args(["doc", "--json"])
        assert args_alias.command == "doc"
        assert args_alias.json is True

        args_run_mock = parser.parse_args(["run", "--mock-llm"])
        assert args_run_mock.command == "run"
        assert args_run_mock.mock_llm is True

    def test_doctor_directory_permission_error(self, tmp_path: Path) -> None:
        s = Settings(db_path=str(tmp_path / "sub" / "trebek.db"), mock_llm=True)
        with patch("os.makedirs", side_effect=PermissionError("Read-only file system")):
            report = run_diagnostics(s, check_api=False)
            db_check = next(c for c in report.checks if c.component == "SQLite WAL Locks")
            assert db_check.status == "FAIL"
            assert "Read-only file system" in db_check.detail
            assert report.success is False

    def test_doctor_check_api_huggingface(self, tmp_path: Path) -> None:
        s = Settings(db_path=str(tmp_path / "trebek.db"), mock_llm=True, hf_token="hf_mock_token_12345")

        # 1. Success case (200 OK)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            report = run_diagnostics(s, check_api=True)
            hf_check = next(c for c in report.checks if c.component == "Hugging Face Token")
            assert hf_check.status == "PASS"
            assert "pyannote access verified" in hf_check.detail

        # 2. Failure case (403 Forbidden - license not accepted)
        import urllib.error

        http_err = urllib.error.HTTPError("https://huggingface.co", 403, "Forbidden", {}, None)
        with patch("urllib.request.urlopen", side_effect=http_err):
            report_fail = run_diagnostics(s, check_api=True)
            hf_check_fail = next(c for c in report_fail.checks if c.component == "Hugging Face Token")
            assert hf_check_fail.status == "WARN"
            assert "403" in hf_check_fail.detail or "gated access check failed" in hf_check_fail.detail


# ═════════════════════════════════════════════════════════════════════════════
# 3. Zero-Cost Offline Mock LLM Mode Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestOfflineMockLLMMode:
    """Verifies that mock LLM mode runs deterministic, schema-valid synthetic passes."""

    def test_generate_mock_llm_response_pass1_anchoring(self) -> None:
        resp, usage = generate_mock_llm_response(
            model="gemini-3.1-flash-lite-preview",
            prompt="Listen to the host interview",
            invocation_context="Pass 1 Speaker Anchoring",
        )
        assert usage["cost_usd"] == 0.0
        assert usage["total_tokens"] == 0.0

        mapping = json.loads(resp.text)
        assert "SPEAKER_00" in mapping
        assert mapping["SPEAKER_00"] == "Ken Jennings"
        assert "SPEAKER_01" in mapping

    def test_generate_mock_llm_response_pass2_skeleton(self) -> None:
        resp, usage = generate_mock_llm_response(
            model="gemini-3.1-pro-preview",
            prompt="Extract skeleton",
            response_schema=EpisodeSkeleton,
            invocation_context="Pass 2 Skeleton",
        )
        skeleton = EpisodeSkeleton.model_validate_json(resp.text)
        assert len(skeleton.jeopardy_categories) == 6
        assert len(skeleton.double_jep_categories) == 6
        assert skeleton.total_jep_clues_played == 30
        assert skeleton.daily_double_count == 3

    def test_generate_mock_llm_response_pass2_metadata(self) -> None:
        resp, usage = generate_mock_llm_response(
            model="gemini-3.1-pro-preview",
            prompt="Extract metadata",
            response_schema=PartialEpisodeMeta,
            invocation_context="Pass 2 Metadata",
        )
        meta = PartialEpisodeMeta.model_validate_json(resp.text)
        assert meta.host_name == "Ken Jennings"
        assert len(meta.contestants) == 3
        assert meta.final_jep.category == "HISTORIC MONUMENTS"
        assert len(meta.final_jep.wagers_and_responses) == 3

    def test_generate_mock_llm_response_pass2_clues(self) -> None:
        resp, usage = generate_mock_llm_response(
            model="gemini-3.1-pro-preview",
            prompt="Extract clues for WORLD CAPITALS",
            response_schema=PartialClues,
            invocation_context="Pass 2 Chunk 1",
        )
        partial_clues = PartialClues.model_validate_json(resp.text)
        assert len(partial_clues.clues) > 0
        clue = partial_clues.clues[0]
        assert clue.round in ("J!", "Double J!")
        assert clue.correct_response.startswith("What is")
        assert len(clue.attempts) == 1
        assert clue.attempts[0].is_correct is True

    @pytest.mark.asyncio
    async def test_gemini_client_works_in_mock_mode_without_api_key(self) -> None:
        with patch.object(settings, "mock_llm", True), patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            client = GeminiClient()
            assert client.client is None

            resp, usage = await client.generate_content(
                model="gemini-3.1-pro-preview",
                prompt="test prompt",
                system_instruction="test instruction",
                invocation_context="Pass 2 Skeleton",
                response_schema=EpisodeSkeleton,
            )
            assert usage["cost_usd"] == 0.0
            assert usage["total_tokens"] == 0.0
            assert "jeopardy_categories" in resp.text

    @pytest.mark.asyncio
    async def test_pass1_anchoring_offline_in_mock_mode(self, tmp_path: Path) -> None:
        fake_audio = tmp_path / "fake_interview.mp3"
        fake_audio.write_bytes(b"dummy mp3 data")

        with patch.object(settings, "mock_llm", True):
            mapping, usage = await execute_pass_1_speaker_anchoring(str(fake_audio))
            assert "SPEAKER_00" in mapping
            assert mapping["SPEAKER_00"] == "Ken Jennings"
            assert usage["cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_pass3_multimodal_bypass_in_mock_mode(self, tmp_path: Path) -> None:
        dummy_episode = Episode(
            episode_date="2024-05-15",
            host_name="Ken Jennings",
            is_tournament=False,
            contestants=[],
            clues=[
                Clue(
                    round="J!",
                    category="ART",
                    board_row=1,
                    board_col=1,
                    clue_text="A painting.",
                    correct_response="What is the Mona Lisa?",
                    host_start_timestamp_ms=1000.0,
                    host_finish_timestamp_ms=4000.0,
                    selection_order=1,
                    is_daily_double=False,
                    requires_visual_context=True,
                    clue_syllable_count=3,
                )
            ],
            final_jep=FinalJep(
                category="ART HISTORY",
                clue_text="Clue",
                correct_response="Answer",
                wagers_and_responses=[],
            ),
            score_adjustments=[],
        )

        with patch.object(settings, "mock_llm", True):
            ep, usage = await execute_pass_3_multimodal_augmentation(
                dummy_episode,
                video_filepath="dummy.mp4",
                output_dir=str(tmp_path),
            )
            assert ep == dummy_episode
            assert usage["cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_verify_and_correct_clues_bypass_in_mock_mode(self) -> None:
        test_clue = Clue(
            round="J!",
            category="SCIENCE",
            board_row=1,
            board_col=1,
            clue_text="Water.",
            correct_response="What is H2O?",
            host_start_timestamp_ms=1000.0,
            host_finish_timestamp_ms=3000.0,
            selection_order=1,
            is_daily_double=False,
            requires_visual_context=False,
            clue_syllable_count=2,
        )

        with patch.object(settings, "mock_llm", True):
            corrections, usage = await verify_and_correct_clues(
                extracted_clues=[test_clue],
                segments=[],
                contestant_names=["Amy"],
            )
            assert corrections == []
            assert test_clue.is_verified is True
            assert usage["cost_usd"] == 0.0

    def test_mock_llm_dynamic_schema_reflection(self) -> None:
        dynamic_categories = ["ANCIENT ROME", "WORLD FLAGS"]
        dynamic_contestants = ["Mattea", "Amy", "Andrew"]
        dynamic_schema = create_dynamic_clue_schema(dynamic_categories, dynamic_contestants)

        resp, usage = generate_mock_llm_response(
            model="gemini-3.1-pro-preview",
            prompt="Extract clues for ANCIENT ROME",
            response_schema=dynamic_schema,
            invocation_context="Pass 2 Dynamic Chunk",
        )
        parsed = dynamic_schema.model_validate_json(resp.text)
        assert hasattr(parsed, "clues")
        assert len(parsed.clues) > 0
        for clue in parsed.clues:
            assert clue.category in dynamic_categories
            for att in clue.attempts:
                assert att.speaker in dynamic_contestants

    @pytest.mark.asyncio
    async def test_mock_llm_verify_final_jeopardy(self) -> None:
        final_jep = FinalJep(
            category="AUTHORS",
            clue_text="He wrote 1984.",
            correct_response="George Orwell",
            wagers_and_responses=[],
        )
        with patch.object(settings, "mock_llm", True):
            verified_resp, usage = await verify_final_jeopardy(
                final_jep, segments=[], contestant_names=["Alice", "Bob", "Charlie"]
            )
            assert verified_resp == "George Orwell"
            assert usage["cost_usd"] == 0.0
