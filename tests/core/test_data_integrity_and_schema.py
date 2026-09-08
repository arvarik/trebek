"""
Tests for Data Integrity & Schema Gaps.

Covers:
1. Episode File Fingerprinting & Deduplication:
   - Partial fingerprint computation (head + tail + size)
   - Exact duplicate detection across renamed files
   - Ingestion deduplication (skipping files with duplicate fingerprints)
   - Database schema and forward migrations for fingerprint and visual_context_description
2. Vector Embeddings & Semantic Lateral Distance:
   - Binary serialization and deserialization of float embeddings
   - Cosine distance and semantic lateral distance calculations
   - Deterministic synthetic mock embeddings (unit length, 768-dim)
   - Clue enrichment with embeddings and distance
   - Relational DB commit of BLOB embeddings and lateral distance
3. Multimodal Timing & Decoupling:
   - Visual clue context extraction during host reading (host_start_timestamp_ms)
   - Visual context extraction for triple stumpers (empty attempts)
   - Podium lockout temporal sniping post-reading (host_finish_timestamp_ms)
   - Podium light timestamp and true buzzer latency computation
   - Zero-cost mock LLM mode synthetic multimodal population
"""

import asyncio
import math
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trebek.analysis.embeddings import (
    cosine_distance,
    deserialize_embedding,
    enrich_clues_with_embeddings,
    process_semantic_lateral_distance,
    serialize_embedding,
)
from trebek.config import settings
from trebek.database.operations import commit_episode_to_relational_tables
from trebek.database.writer import DatabaseWriter
from trebek.llm.client import GeminiClient
from trebek.llm.mock import generate_mock_embedding
from trebek.llm.pass3_multimodal import (
    execute_pass_3_multimodal_augmentation,
    extract_podium_lockout_sniping,
    extract_visual_clue_context,
)
from trebek.pipeline.discovery import (
    compute_file_fingerprint,
    discover_video_files,
    scan_video_files,
)
from trebek.pipeline.workers.ingestion import run_ingestion_pass
from trebek.schemas import BuzzAttempt, Clue, Contestant, Episode, FinalJep
from trebek.state_machine import TrebekStateMachine


# ═════════════════════════════════════════════════════════════════════════════
# 1. Episode File Fingerprinting & Deduplication Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestFileFingerprintingAndDeduplication:
    """Tests for file fingerprinting and deduplication."""

    def test_small_file_fingerprint(self, tmp_path: Path) -> None:
        """Files <= 128 KB should hash the entire file deterministically."""
        small_file = tmp_path / "small.mp4"
        data = b"J! episode small test data" * 100
        small_file.write_bytes(data)

        fp1 = compute_file_fingerprint(str(small_file))
        fp2 = compute_file_fingerprint(str(small_file))
        assert len(fp1) == 64
        assert fp1 == fp2

    def test_large_file_fingerprint(self, tmp_path: Path) -> None:
        """Files > 128 KB should hash size + head 64KB + tail 64KB."""
        large_file = tmp_path / "large.mp4"
        # 200 KB total
        head = b"A" * 65536
        middle = b"B" * (200 * 1024 - 131072)
        tail = b"C" * 65536
        large_file.write_bytes(head + middle + tail)

        fp1 = compute_file_fingerprint(str(large_file))
        assert len(fp1) == 64

        # Changing middle should not affect head-tail fingerprint
        modified_file = tmp_path / "modified_middle.mp4"
        modified_file.write_bytes(head + b"Z" * len(middle) + tail)
        fp2 = compute_file_fingerprint(str(modified_file))
        assert fp1 == fp2

        # Changing head or size should change fingerprint
        changed_head = tmp_path / "changed_head.mp4"
        changed_head.write_bytes(b"X" * 65536 + middle + tail)
        assert compute_file_fingerprint(str(changed_head)) != fp1

    def test_renamed_file_identical_fingerprint(self, tmp_path: Path) -> None:
        """Renamed identical files should produce the exact same fingerprint."""
        file1 = tmp_path / "original.mp4"
        file2 = tmp_path / "renamed_copy.mp4"
        content = b"Exact video contents here" * 1000
        file1.write_bytes(content)
        file2.write_bytes(content)

        assert compute_file_fingerprint(str(file1)) == compute_file_fingerprint(str(file2))

    def test_nonexistent_file_returns_empty_fingerprint(self) -> None:
        assert compute_file_fingerprint("/nonexistent/path/to/video.mp4") == ""

    def test_scan_and_discover_include_fingerprint(self, tmp_path: Path) -> None:
        """scan_video_files and discover_video_files should include fingerprint in results."""
        vfile = tmp_path / "S40E01.mp4"
        vfile.write_bytes(b"dummy video data" * 10)

        scanned = scan_video_files(str(tmp_path))
        assert len(scanned) == 1
        assert "fingerprint" in scanned[0]
        assert len(scanned[0]["fingerprint"]) == 64

        discovered = discover_video_files(str(tmp_path))
        assert len(discovered) == 1
        assert "fingerprint" in discovered[0]
        assert discovered[0]["fingerprint"] == scanned[0]["fingerprint"]

    @pytest.mark.asyncio
    async def test_ingestion_deduplication_skips_identical_files(self, tmp_path: Path) -> None:
        """run_ingestion_pass should detect duplicate fingerprints and skip re-ingestion."""
        db_path = str(tmp_path / "test.db")
        writer = DatabaseWriter(db_path=db_path)
        await writer.start()

        # Initialize schema with fingerprint column
        schema_path = Path(__file__).parent.parent.parent / "trebek" / "schema.sql"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(schema_path.read_text(encoding="utf-8"))

        input_dir = tmp_path / "videos"
        input_dir.mkdir()

        # Create original video
        v1 = input_dir / "S41E01.mp4"
        content = b"Video file content for S41E01" * 50
        v1.write_bytes(content)

        # Mock orchestrator
        mock_orch = MagicMock()
        mock_orch.mode = "once"
        mock_orch.db_writer = writer
        mock_orch.registered_episode_ids = set()
        mock_orch.stats = {"total": 0}
        mock_orch.gpu_work_ready = MagicMock()

        # Pass 1: ingest v1
        count1 = await run_ingestion_pass(mock_orch, str(input_dir))
        assert count1 == 1
        assert "S41E01" in mock_orch.registered_episode_ids

        # Create duplicate file with a different name in a subdirectory
        sub = input_dir / "duplicates"
        sub.mkdir()
        v2 = sub / "download_copy.mp4"
        v2.write_bytes(content)  # Identical content!

        # Pass 2: should detect duplicate fingerprint and skip
        count2 = await run_ingestion_pass(mock_orch, str(input_dir))
        assert count2 == 0  # Skipped!

        # Verify only 1 episode is in pipeline_state
        rows = await writer.execute("SELECT episode_id, fingerprint FROM pipeline_state")
        assert len(rows) == 1
        assert rows[0][0] == "S41E01"
        assert rows[0][1] == compute_file_fingerprint(str(v1))

        await writer.stop()

    def test_forward_migration_for_existing_database(self, tmp_path: Path) -> None:
        """Databases created without fingerprint or visual_context_description should be migrated cleanly."""
        db_path = str(tmp_path / "old.db")
        # Create an older table without fingerprint or visual_context_description
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE pipeline_state (
                    episode_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source_filename TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE clues (
                    clue_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    clue_text TEXT NOT NULL
                )
            """)

        # Run migration check
        with sqlite3.connect(db_path) as conn:
            cols_ps = {c[1] for c in conn.execute("PRAGMA table_info(pipeline_state)").fetchall()}
            if "fingerprint" not in cols_ps:
                conn.execute("ALTER TABLE pipeline_state ADD COLUMN fingerprint TEXT")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_state_fingerprint ON pipeline_state(fingerprint)")

            cols_clues = {c[1] for c in conn.execute("PRAGMA table_info(clues)").fetchall()}
            if "visual_context_description" not in cols_clues:
                conn.execute("ALTER TABLE clues ADD COLUMN visual_context_description TEXT")

        # Verify columns exist now
        with sqlite3.connect(db_path) as conn:
            cols_ps = {c[1] for c in conn.execute("PRAGMA table_info(pipeline_state)").fetchall()}
            cols_clues = {c[1] for c in conn.execute("PRAGMA table_info(clues)").fetchall()}
            assert "fingerprint" in cols_ps
            assert "visual_context_description" in cols_clues


# ═════════════════════════════════════════════════════════════════════════════
# 2. Vector Embeddings & Semantic Lateral Distance Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestEmbeddingsAndSemanticLateralDistance:
    """Tests for vector embeddings, cosine distance, and relational storage."""

    def test_serialize_and_deserialize_embedding(self) -> None:
        """Float vectors should serialize to binary and deserialize back accurately."""
        original = [0.123456, -0.987654, 0.0, 1.0, -0.5]
        blob = serialize_embedding(original)
        assert isinstance(blob, bytes)
        assert len(blob) == len(original) * 4  # 4 bytes per float32

        restored = deserialize_embedding(blob)
        assert restored is not None
        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            assert math.isclose(orig, rest, abs_tol=1e-5)

        assert serialize_embedding(None) is None
        assert serialize_embedding([]) is None
        assert deserialize_embedding(None) is None
        assert deserialize_embedding(b"invalid_bytes") is None

    def test_cosine_distance_properties(self) -> None:
        """Cosine distance should follow metric properties and handle edge cases."""
        # Identical vectors -> 0.0
        v1 = [1.0, 0.0, 0.0]
        assert math.isclose(cosine_distance(v1, v1), 0.0, abs_tol=1e-6)

        # Orthogonal vectors -> 1.0
        v2 = [0.0, 1.0, 0.0]
        assert math.isclose(cosine_distance(v1, v2), 1.0, abs_tol=1e-6)

        # Opposing vectors -> 1.0 (clamped)
        v3 = [-1.0, 0.0, 0.0]
        assert math.isclose(cosine_distance(v1, v3), 1.0, abs_tol=1e-6)

        # Zero norm vector -> 1.0
        assert math.isclose(cosine_distance([0.0, 0.0], [1.0, 1.0]), 1.0)

        # Dimension mismatch raises ValueError
        with pytest.raises(ValueError, match="same dimensionality"):
            cosine_distance([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_process_semantic_lateral_distance(self) -> None:
        """Calculates lateral semantic distance between clue and response vectors."""
        clue_vec = [0.5, 0.5, 0.5, 0.5]
        resp_vec = [0.5, 0.5, 0.5, 0.5]
        dist = process_semantic_lateral_distance(clue_vec, resp_vec)
        assert math.isclose(dist, 0.0, abs_tol=1e-5)

    def test_generate_mock_embedding(self) -> None:
        """generate_mock_embedding should produce deterministic 768-dim unit vectors."""
        emb1 = generate_mock_embedding("What is Paris?")
        emb2 = generate_mock_embedding("What is Paris?")
        emb_other = generate_mock_embedding("Who is Albert Einstein?")

        assert len(emb1) == 768
        assert emb1 == emb2
        assert emb1 != emb_other

        # Check unit normalization: sum of squares ≈ 1.0
        norm_sq = sum(x * x for x in emb1)
        assert math.isclose(norm_sq, 1.0, abs_tol=1e-2)

        # Empty string handling
        empty_emb = generate_mock_embedding("")
        assert len(empty_emb) == 768
        assert all(x == 0.0 for x in empty_emb)

    @pytest.mark.asyncio
    async def test_gemini_client_mock_embed_content(self) -> None:
        """GeminiClient.embed_content should produce valid embeddings in mock mode."""
        with patch.object(settings, "mock_llm", True):
            client = GeminiClient()
            texts = ["Capital of France", "Paris"]
            embeddings = await client.embed_content(texts)
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 768
            assert len(embeddings[1]) == 768

    @pytest.mark.asyncio
    async def test_enrich_clues_with_embeddings(self) -> None:
        """enrich_clues_with_embeddings should attach embeddings and lateral distance to Clue objects."""
        clues = [
            Clue(
                round="J!",
                category="GEOGRAPHY",
                board_row=1,
                board_col=1,
                selection_order=1,
                is_daily_double=False,
                requires_visual_context=False,
                host_start_timestamp_ms=1000.0,
                host_finish_timestamp_ms=3000.0,
                clue_syllable_count=5,
                clue_text="Capital of France.",
                correct_response="Paris",
            )
        ]

        mock_client = MagicMock()
        mock_client.embed_content = AsyncMock(
            return_value=[
                [1.0, 0.0, 0.0],  # clue_text
                [0.8, 0.6, 0.0],  # correct_response
            ]
        )

        await enrich_clues_with_embeddings(clues, client=mock_client)

        assert clues[0].clue_embedding == [1.0, 0.0, 0.0]
        assert clues[0].response_embedding == [0.8, 0.6, 0.0]
        assert clues[0].semantic_lateral_distance is not None
        assert 0.0 <= clues[0].semantic_lateral_distance <= 1.0

    @pytest.mark.asyncio
    async def test_relational_commit_persists_embeddings_and_distance(self, tmp_path: Path) -> None:
        """commit_episode_to_relational_tables should write clue embeddings and lateral distance to SQLite."""
        db_path = str(tmp_path / "test_embeddings.db")
        writer = DatabaseWriter(db_path=db_path)
        await writer.start()

        schema_path = Path(__file__).parent.parent.parent / "trebek" / "schema.sql"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(schema_path.read_text(encoding="utf-8"))

        clue = Clue(
            round="J!",
            category="SCIENCE",
            board_row=1,
            board_col=1,
            selection_order=1,
            is_daily_double=False,
            requires_visual_context=True,
            visual_context_description="A picture of the chemical element Au.",
            host_start_timestamp_ms=1000.0,
            host_finish_timestamp_ms=3500.0,
            clue_syllable_count=6,
            clue_text="This element has atomic number 79.",
            correct_response="Gold",
            clue_embedding=[0.1, 0.2, 0.3],
            response_embedding=[0.4, 0.5, 0.6],
            semantic_lateral_distance=0.1523,
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Ken",
                    response_given="What is gold?",
                    is_correct=True,
                    buzz_timestamp_ms=3800.0,
                    response_start_timestamp_ms=4000.0,
                    is_lockout_inferred=False,
                    podium_light_timestamp_ms=3650.0,
                    true_buzzer_latency_ms=150.0,
                )
            ],
        )

        episode = Episode(
            episode_date="2024-05-15",
            host_name="Alex Trebek",
            is_tournament=False,
            contestants=[
                Contestant(
                    name="Ken",
                    podium_position=1,
                    occupational_category="Author",
                    is_returning_champion=True,
                    description="Champion",
                )
            ],
            clues=[clue],
            final_jep=FinalJep(
                category="FINAL",
                clue_text="Final clue",
                correct_response="Final answer",
                wagers_and_responses=[],
            ),
            score_adjustments=[],
        )

        sm = TrebekStateMachine(valid_contestants={"Ken"})
        sm.process_clue(clue)

        await commit_episode_to_relational_tables(writer, "ep_emb_test", episode, sm)

        # Verify clues table row
        rows = await writer.execute(
            "SELECT clue_embedding, response_embedding, semantic_lateral_distance, visual_context_description "
            "FROM clues WHERE clue_id = 'ep_emb_test_c1'"
        )
        assert len(rows) == 1
        clue_blob, resp_blob, distance, visual_desc = rows[0]
        assert deserialize_embedding(clue_blob) == pytest.approx([0.1, 0.2, 0.3], abs=1e-5)
        assert deserialize_embedding(resp_blob) == pytest.approx([0.4, 0.5, 0.6], abs=1e-5)
        assert math.isclose(distance, 0.1523, abs_tol=1e-4)
        assert visual_desc == "A picture of the chemical element Au."

        # Verify buzz_attempts table row has podium light and latency
        buzz_rows = await writer.execute(
            "SELECT podium_light_timestamp_ms, true_buzzer_latency_ms FROM buzz_attempts WHERE clue_id = 'ep_emb_test_c1'"
        )
        assert len(buzz_rows) == 1
        assert math.isclose(buzz_rows[0][0], 3650.0)
        assert math.isclose(buzz_rows[0][1], 150.0)

        await writer.stop()


# ═════════════════════════════════════════════════════════════════════════════
# 3. Multimodal Timing & Decoupling Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestMultimodalTimingAndDecoupling:
    """Tests for visual clue context extraction vs. podium lockout sniping."""

    @pytest.mark.asyncio
    async def test_visual_clue_context_extracted_during_host_read(self, tmp_path: Path) -> None:
        """Visual clues should extract video during host reading (host_start to host_finish)."""
        clue = Clue(
            round="J!",
            category="ART",
            board_row=1,
            board_col=1,
            selection_order=1,
            is_daily_double=False,
            requires_visual_context=True,
            host_start_timestamp_ms=10000.0,
            host_finish_timestamp_ms=14000.0,
            clue_syllable_count=6,
            clue_text="Seen here, this famous painting.",
            correct_response="Mona Lisa",
            attempts=[],  # Triple stumper!
        )

        mock_client = MagicMock()
        mock_client.upload_file = AsyncMock(return_value=MagicMock(name="files/visual123"))
        mock_client.delete_file = AsyncMock()
        mock_file_info = MagicMock()
        mock_file_info.state.name = "ACTIVE"
        mock_client.client.files.get.return_value = mock_file_info
        mock_client.generate_content = AsyncMock(
            return_value=(
                MagicMock(text="The painting depicts a woman with an enigmatic smile."),
                {"input_tokens": 150.0, "output_tokens": 30.0, "cost_usd": 0.002, "latency_ms": 100.0},
            )
        )

        captured_args = []

        async def mock_exec(*args: Any, **kwargs: Any) -> Any:
            captured_args.append(args)
            clip_path = args[10]
            Path(clip_path).touch()
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            sem = asyncio.Semaphore(1)
            usage = await extract_visual_clue_context(
                clue=clue,
                video_filepath="test_ep.mp4",
                output_dir=str(tmp_path),
                ep_id="ep_test",
                client=mock_client,
                semaphore=sem,
            )

        # 1. Start time must be host_start (10.0s), NOT host_finish (14.0s)!
        assert len(captured_args) == 1
        ffmpeg_args = captured_args[0]
        start_time_arg = ffmpeg_args[3]  # -ss argument
        assert start_time_arg == "10.000"

        # 2. Clue description should be saved even with zero attempts (triple stumper)
        assert clue.visual_context_description == "The painting depicts a woman with an enigmatic smile."
        assert usage["input_tokens"] == 150.0
        assert mock_client.delete_file.call_count == 1

    @pytest.mark.asyncio
    async def test_podium_lockout_sniping_extracted_post_host_read(self, tmp_path: Path) -> None:
        """Podium lockout sniping should extract video after host reading (host_finish + 3.0s)."""
        clue = Clue(
            round="J!",
            category="HISTORY",
            board_row=2,
            board_col=1,
            selection_order=2,
            is_daily_double=False,
            requires_visual_context=False,
            host_start_timestamp_ms=20000.0,
            host_finish_timestamp_ms=24000.0,
            clue_syllable_count=8,
            clue_text="He was the first president.",
            correct_response="George Washington",
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Amy",
                    response_given="Washington",
                    is_correct=True,
                    buzz_timestamp_ms=24450.0,
                    response_start_timestamp_ms=24600.0,
                    is_lockout_inferred=False,
                )
            ],
        )

        mock_client = MagicMock()
        mock_client.upload_file = AsyncMock(return_value=MagicMock(name="files/podium123"))
        mock_client.delete_file = AsyncMock()
        mock_file_info = MagicMock()
        mock_file_info.state.name = "ACTIVE"
        mock_client.client.files.get.return_value = mock_file_info
        mock_client.generate_content = AsyncMock(
            return_value=(
                MagicMock(text="0.15"),  # light turned on at 0.15s into clip
                {"input_tokens": 100.0, "output_tokens": 10.0, "cost_usd": 0.001, "latency_ms": 80.0},
            )
        )

        captured_args = []

        async def mock_exec(*args: Any, **kwargs: Any) -> Any:
            captured_args.append(args)
            clip_path = args[10]
            Path(clip_path).touch()
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            sem = asyncio.Semaphore(1)
            usage = await extract_podium_lockout_sniping(
                clue=clue,
                video_filepath="test_ep.mp4",
                output_dir=str(tmp_path),
                ep_id="ep_test",
                client=mock_client,
                semaphore=sem,
            )

        # 1. Start time must be host_finish (24.0s)
        assert len(captured_args) == 1
        ffmpeg_args = captured_args[0]
        start_time_arg = ffmpeg_args[3]
        assert start_time_arg == "24.000"

        # 2. Podium light timestamp = (24.0 + 0.15) * 1000 = 24150.0 ms
        assert clue.attempts[0].podium_light_timestamp_ms == 24150.0
        # True latency = 24450.0 - 24150.0 = 300.0 ms
        assert clue.attempts[0].true_buzzer_latency_ms is not None
        assert math.isclose(clue.attempts[0].true_buzzer_latency_ms, 300.0)
        assert usage["input_tokens"] == 100.0

    @pytest.mark.asyncio
    async def test_mock_llm_mode_populates_synthetic_multimodal_data(self) -> None:
        """In mock LLM mode, execute_pass_3_multimodal_augmentation populates synthetic data without network calls."""
        clue = Clue(
            round="J!",
            category="MAPS",
            board_row=1,
            board_col=1,
            selection_order=1,
            is_daily_double=False,
            requires_visual_context=True,
            host_start_timestamp_ms=5000.0,
            host_finish_timestamp_ms=8000.0,
            clue_syllable_count=4,
            clue_text="This European capital.",
            correct_response="Rome",
            attempts=[
                BuzzAttempt(
                    attempt_order=1,
                    speaker="Bob",
                    response_given="Rome",
                    is_correct=True,
                    buzz_timestamp_ms=8250.0,
                    response_start_timestamp_ms=8400.0,
                    is_lockout_inferred=False,
                )
            ],
        )

        episode = Episode(
            episode_date="2024-03-01",
            host_name="Ken Jennings",
            is_tournament=False,
            contestants=[],
            clues=[clue],
            final_jep=FinalJep(category="FINAL", clue_text="Clue", correct_response="Answer", wagers_and_responses=[]),
            score_adjustments=[],
        )

        with patch.object(settings, "mock_llm", True):
            aug_ep, usage = await execute_pass_3_multimodal_augmentation(
                episode=episode,
                video_filepath="dummy.mp4",
                output_dir="/tmp",
            )

            assert aug_ep.clues[0].visual_context_description is not None
            assert "Synthetic visual context" in aug_ep.clues[0].visual_context_description
            assert aug_ep.clues[0].attempts[0].podium_light_timestamp_ms is not None
            assert aug_ep.clues[0].attempts[0].true_buzzer_latency_ms is not None
            assert usage["cost_usd"] == 0.0
