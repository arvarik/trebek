#!/usr/bin/env python3
"""
Trebek LLM Pipeline End-to-End Validation Script.

Two modes:
  --dry-run (default): Validates GPU output structure + runs all post-extraction
                       checks with synthetic data. No API calls.
  --live:              Calls Gemini API for full Pass 1 → Pass 2 → Validation →
                       State Machine pipeline against real GPU output.

Usage:
  # Dry-run: validate GPU output + run offline checks
  .venv/bin/python tests/test_llm_pipeline_e2e.py --gpu-output gpu_outputs/gpu_output_*.json.gz

  # Live: full end-to-end with Gemini API
  .venv/bin/python tests/test_llm_pipeline_e2e.py --live --gpu-output gpu_outputs/gpu_output_*.json.gz

  # Live with specific interview audio slice
  .venv/bin/python tests/test_llm_pipeline_e2e.py --live \\
      --gpu-output gpu_outputs/gpu_output_*.json.gz \\
      --audio-slice gpu_outputs/*_interview_slice.mp3
"""

import argparse
import asyncio
import gzip
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


# ── Validation Checks ────────────────────────────────────────────────


class ValidationResult:
    """Accumulates pass/fail/warn results for a pipeline run."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        self.checks.append({"name": name, "status": status, "detail": detail})
        icon = "✅" if passed else "❌"
        msg = f"  {icon} {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    def warn(self, name: str, detail: str = "") -> None:
        self.checks.append({"name": name, "status": "WARN", "detail": detail})
        print(f"  ⚠️  {name} — {detail}")

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "FAIL")

    @property
    def warned(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "WARN")

    def summary(self) -> str:
        total = len(self.checks)
        return (
            f"\n{'=' * 60}\n"
            f"  RESULTS: {self.passed}/{total} passed, "
            f"{self.failed} failed, {self.warned} warnings\n"
            f"{'=' * 60}"
        )


# ── GPU Output Validation ────────────────────────────────────────────


def validate_gpu_output(gpu_data: dict[str, Any], result: ValidationResult) -> list[dict[str, Any]]:
    """Validates the structure and quality of GPU (WhisperX) output."""
    print("\n── GPU Output Structure ──────────────────────────────")

    # Top-level keys
    transcript = gpu_data.get("transcript", {})
    result.check("GPU: 'transcript' key exists", bool(transcript))

    segments = transcript.get("segments", [])
    result.check("GPU: segments array exists", isinstance(segments, list))
    result.check("GPU: segments non-empty", len(segments) > 0, f"{len(segments)} segments")

    if not segments:
        return []

    # Segment structure
    first_seg = segments[0]
    result.check("GPU: segment has 'text' key", "text" in first_seg)
    result.check("GPU: segment has 'start' key", "start" in first_seg)
    result.check("GPU: segment has 'end' key", "end" in first_seg)
    result.check("GPU: segment has 'speaker' key", "speaker" in first_seg)

    # Timestamp sanity
    has_start = sum(1 for s in segments if s.get("start") is not None)
    has_end = sum(1 for s in segments if s.get("end") is not None)
    null_start = len(segments) - has_start
    null_end = len(segments) - has_end
    result.check(
        "GPU: >95% segments have 'start' timestamp",
        has_start / len(segments) > 0.95,
        f"{has_start}/{len(segments)} have start, {null_start} null",
    )
    result.check(
        "GPU: >95% segments have 'end' timestamp",
        has_end / len(segments) > 0.95,
        f"{has_end}/{len(segments)} have end, {null_end} null",
    )

    # Monotonicity check
    non_monotonic = 0
    for i in range(1, len(segments)):
        prev_start = segments[i - 1].get("start", 0) or 0
        curr_start = segments[i].get("start", 0) or 0
        if curr_start < prev_start - 0.5:  # Allow small jitter
            non_monotonic += 1
    result.check(
        "GPU: timestamps are monotonic",
        non_monotonic < len(segments) * 0.02,
        f"{non_monotonic} non-monotonic transitions",
    )

    # Speaker diarization
    speakers = {s.get("speaker", "UNKNOWN") for s in segments}
    result.check("GPU: multiple speakers detected", len(speakers) >= 2, f"speakers: {sorted(speakers)}")
    result.check("GPU: speaker count reasonable (2-10)", 2 <= len(speakers) <= 10, f"{len(speakers)} speakers")

    # Duration check
    last_end = max((s.get("end", 0) or 0) for s in segments)
    duration_min = last_end / 60
    result.check("GPU: episode duration 15-35 min", 15 <= duration_min <= 35, f"{duration_min:.1f} minutes")

    # Text quality
    empty_text = sum(1 for s in segments if not (s.get("text", "").strip()))
    result.check(
        "GPU: <5% empty text segments", empty_text / len(segments) < 0.05, f"{empty_text}/{len(segments)} empty"
    )

    # Jeopardy content markers
    full_text = " ".join(s.get("text", "") for s in segments).lower()
    has_jeopardy_markers = any(
        m in full_text
        for m in [
            "jeopardy",
            "double jeopardy",
            "daily double",
            "final jeopardy",
            "what is",
            "who is",
            "correct",
            "right",
        ]
    )
    result.check("GPU: Jeopardy content markers found", has_jeopardy_markers)

    return list(segments)


# ── Post-Extraction Validation ────────────────────────────────────────


def validate_episode(episode: Any, result: ValidationResult, segments: list[dict[str, Any]]) -> None:
    """Validates an extracted Episode object against domain invariants."""
    from trebek.llm.validation import _validate_extraction_integrity

    print("\n── Episode Structure ─────────────────────────────────")

    # Contestants
    result.check("Episode: has contestants", len(episode.contestants) > 0, f"{len(episode.contestants)} contestants")
    result.check(
        "Episode: exactly 3 contestants (standard)", len(episode.contestants) == 3, f"{len(episode.contestants)} found"
    )
    result.check("Episode: host name extracted", bool(episode.host_name), episode.host_name)
    result.check("Episode: episode date extracted", bool(episode.episode_date), episode.episode_date)

    # Unique podium positions
    positions = [c.podium_position for c in episode.contestants]
    result.check("Episode: unique podium positions", len(set(positions)) == len(positions), f"positions: {positions}")

    print("\n── Clue Extraction Quality ───────────────────────────")

    # Clue counts
    j_clues = [c for c in episode.clues if c.round == "Jeopardy"]
    dj_clues = [c for c in episode.clues if c.round == "Double Jeopardy"]
    total = len(episode.clues)

    result.check("Clues: total > 0", total > 0, f"{total} clues")
    result.check("Clues: Jeopardy round 20-30", 20 <= len(j_clues) <= 30, f"{len(j_clues)} J clues")
    result.check("Clues: Double Jeopardy round 20-30", 20 <= len(dj_clues) <= 30, f"{len(dj_clues)} DJ clues")

    # Daily Doubles
    dd_count = sum(1 for c in episode.clues if c.is_daily_double)
    result.check("Clues: 1-3 Daily Doubles", 1 <= dd_count <= 3, f"{dd_count} DDs")

    # Categories
    j_cats = {c.category.lower().strip() for c in j_clues}
    dj_cats = {c.category.lower().strip() for c in dj_clues}
    result.check("Clues: J round has 5-6 categories", 5 <= len(j_cats) <= 6, f"{len(j_cats)}: {sorted(j_cats)}")
    result.check("Clues: DJ round has 5-6 categories", 5 <= len(dj_cats) <= 6, f"{len(dj_cats)}: {sorted(dj_cats)}")

    # Timestamps vs segments
    max_seg_time = max((s.get("end", 0) or 0) for s in segments) * 1000 if segments else 0
    out_of_bounds = sum(1 for c in episode.clues if c.host_start_timestamp_ms > max_seg_time * 1.05)
    result.check(
        "Clues: timestamps within segment range", out_of_bounds == 0, f"{out_of_bounds} clues beyond segment time range"
    )

    # Buzz attempts
    total_attempts = sum(len(c.attempts) for c in episode.clues)
    result.check("Clues: buzz attempts extracted", total_attempts > 0, f"{total_attempts} total attempts")

    # Triple stumpers (no attempts or all wrong)
    triple_stumpers = sum(1 for c in episode.clues if len(c.attempts) == 0 or all(not a.is_correct for a in c.attempts))
    result.check(
        "Clues: triple stumper rate < 50%", triple_stumpers / max(1, total) < 0.5, f"{triple_stumpers}/{total}"
    )

    # Contestant FK consistency
    contestant_names = {c.name.lower().strip() for c in episode.contestants}
    unknown = set()
    for clue in episode.clues:
        for att in clue.attempts:
            if att.speaker.lower().strip() not in contestant_names:
                unknown.add(att.speaker)
    result.check(
        "Clues: all buzzers are known contestants",
        len(unknown) == 0,
        f"unknown: {unknown}" if unknown else "all mapped",
    )

    print("\n── Final Jeopardy ───────────────────────────────────")
    fj = episode.final_jeopardy
    result.check("FJ: category extracted", bool(fj.category), fj.category)
    result.check(
        "FJ: clue text extracted",
        bool(fj.clue_text),
        fj.clue_text[:60] + "..." if len(fj.clue_text) > 60 else fj.clue_text,
    )
    result.check("FJ: has wager responses", len(fj.wagers_and_responses) > 0, f"{len(fj.wagers_and_responses)} wagers")

    # FJ contestant consistency
    fj_unknown = [w.contestant for w in fj.wagers_and_responses if w.contestant.lower().strip() not in contestant_names]
    result.check(
        "FJ: all wagerers are known contestants",
        len(fj_unknown) == 0,
        f"unknown: {fj_unknown}" if fj_unknown else "all mapped",
    )

    print("\n── Integrity Validation ──────────────────────────────")
    warnings = _validate_extraction_integrity(episode)
    result.check("Integrity: passed all checks", len(warnings) == 0, f"{len(warnings)} warnings")
    for w in warnings:
        result.warn("Integrity warning", w)


def validate_state_machine(episode: Any, result: ValidationResult) -> None:
    """Runs the state machine and validates final scores are reasonable."""
    from trebek.state_machine import TrebekStateMachine

    print("\n── State Machine Verification ────────────────────────")

    sm = TrebekStateMachine()
    sm.load_adjustments(episode.score_adjustments)

    errors: list[str] = []
    for clue in episode.clues:
        try:
            sm.process_clue(clue)
        except Exception as e:
            errors.append(f"Clue {clue.selection_order}: {e}")

    result.check(
        "StateMachine: processed all clues without errors",
        len(errors) == 0,
        f"{len(errors)} errors" if errors else f"{len(episode.clues)} clues processed",
    )
    for err in errors[:5]:
        result.warn("StateMachine error", err)

    # Score reasonableness
    for name, score in sm.scores.items():
        result.check(f"StateMachine: {name} score reasonable (-10k to +50k)", -10000 <= score <= 50000, f"${score:,}")

    # At least one positive score
    has_positive = any(s > 0 for s in sm.scores.values())
    result.check("StateMachine: at least one positive score", has_positive)

    # Board control tracked
    result.check(
        "StateMachine: board control tracked",
        sm.current_board_control_contestant is not None,
        sm.current_board_control_contestant or "None",
    )


def validate_db_commit_dry_run(episode: Any, result: ValidationResult) -> None:
    """Validates that the episode data would commit cleanly to the DB schema."""
    import sqlite3
    import tempfile
    from trebek.state_machine import TrebekStateMachine
    from trebek.database.operations import commit_episode_to_relational_tables
    from trebek.database.writer import DatabaseWriter

    print("\n── Database Commit (Dry Run) ─────────────────────────")

    # Create a temp DB with the schema
    schema_path = Path(__file__).resolve().parents[2] / "trebek" / "schema.sql"
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with sqlite3.connect(tmp_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            with open(schema_path) as f:
                conn.executescript(f.read())

        # Run state machine
        sm = TrebekStateMachine()
        sm.load_adjustments(episode.score_adjustments)
        for clue in episode.clues:
            sm.process_clue(clue)

        # Attempt commit
        async def _commit() -> None:
            db = DatabaseWriter(tmp_path)
            await db.start()
            try:
                await commit_episode_to_relational_tables(db, "test_episode", episode, sm)
            finally:
                await db.stop()

        asyncio.run(_commit())

        # Verify data was written
        with sqlite3.connect(tmp_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            ep_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            clue_count = conn.execute("SELECT COUNT(*) FROM clues").fetchone()[0]
            buzz_count = conn.execute("SELECT COUNT(*) FROM buzz_attempts").fetchone()[0]
            contestant_count = conn.execute("SELECT COUNT(*) FROM contestants").fetchone()[0]

        result.check("DB: episode row inserted", ep_count == 1)
        result.check("DB: clues inserted", clue_count == len(episode.clues), f"{clue_count} rows")
        result.check("DB: buzz attempts inserted", buzz_count > 0, f"{buzz_count} rows")
        result.check(
            "DB: contestants inserted", contestant_count == len(episode.contestants), f"{contestant_count} rows"
        )

    except Exception as e:
        result.check("DB: commit succeeded", False, str(e)[:200])
        traceback.print_exc()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Main ──────────────────────────────────────────────────────────────


def load_gpu_output(path: str) -> dict[str, Any]:
    """Load GPU output from .json.gz or .json file."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return dict(data)


async def run_live_pipeline(
    segments: list[dict[str, Any]],
    audio_slice_path: str | None,
    result: ValidationResult,
) -> Any:
    """Runs the full LLM pipeline (Pass 1 + Pass 2) against real data."""
    from trebek.llm import execute_pass_1_speaker_anchoring, execute_pass_2_data_extraction

    print("\n── Pass 1: Speaker Anchoring ─────────────────────────")
    if audio_slice_path and os.path.exists(audio_slice_path):
        start_t = time.perf_counter()
        speaker_mapping, usage1 = await execute_pass_1_speaker_anchoring(audio_slice_path)
        pass1_ms = (time.perf_counter() - start_t) * 1000

        result.check("Pass1: returned mapping", isinstance(speaker_mapping, dict), f"{len(speaker_mapping)} speakers")
        result.check("Pass1: found 2+ speakers", len(speaker_mapping) >= 2, f"mapping: {speaker_mapping}")
        result.check("Pass1: latency < 30s", pass1_ms < 30000, f"{pass1_ms:.0f}ms")
        result.check("Pass1: cost < $0.05", usage1.get("cost_usd", 0) < 0.05, f"${usage1.get('cost_usd', 0):.4f}")
    else:
        print("  ⚠️  No audio slice provided — using empty speaker mapping")
        speaker_mapping = {}
        result.warn("Pass1: skipped (no audio slice)")

    print("\n── Pass 2: Structured Extraction ─────────────────────")
    start_t = time.perf_counter()
    episode, usage2, retries = await execute_pass_2_data_extraction(
        segments,
        speaker_mapping,
        max_retries=3,
    )
    pass2_ms = (time.perf_counter() - start_t) * 1000

    result.check("Pass2: returned Episode", episode is not None)
    result.check("Pass2: extracted clues", len(episode.clues) > 0, f"{len(episode.clues)} clues")
    result.check("Pass2: latency < 120s", pass2_ms < 120000, f"{pass2_ms:.0f}ms")
    total_cost = usage2.get("cost_usd", 0)
    result.check("Pass2: cost < $1.00", total_cost < 1.0, f"${total_cost:.4f}")
    result.check("Pass2: retries <= 2", retries <= 2, f"{retries} retries")

    return episode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trebek LLM Pipeline E2E Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gpu-output", required=True, help="Path to GPU output .json.gz file")
    parser.add_argument("--audio-slice", default=None, help="Path to interview audio slice .mp3 (for --live mode)")
    parser.add_argument("--live", action="store_true", help="Run live LLM extraction (requires GEMINI_API_KEY)")
    parser.add_argument(
        "--episode-json", default=None, help="Path to pre-extracted episode .json (skip LLM, validate only)"
    )
    args = parser.parse_args()

    result = ValidationResult()
    print("=" * 60)
    print("  Trebek LLM Pipeline E2E Validation")
    print(f"  Mode: {'LIVE (API calls)' if args.live else 'DRY-RUN (offline)'}")
    print(f"  GPU output: {args.gpu_output}")
    print("=" * 60)

    # ── Load and validate GPU output ──────────────────────────────
    try:
        gpu_data = load_gpu_output(args.gpu_output)
    except Exception as e:
        result.check("GPU: file loaded", False, str(e))
        print(result.summary())
        sys.exit(1)

    result.check("GPU: file loaded", True, f"{len(json.dumps(gpu_data)):,} chars")
    segments = validate_gpu_output(gpu_data, result)

    if not segments:
        print("\n❌ No segments found — cannot continue")
        print(result.summary())
        sys.exit(1)

    # ── Episode extraction / loading ──────────────────────────────
    episode = None

    if args.episode_json:
        # Load pre-extracted episode
        from trebek.schemas import Episode

        print(f"\n── Loading pre-extracted episode: {args.episode_json}")
        with open(args.episode_json) as f:
            episode = Episode.model_validate_json(f.read())
        result.check("Episode: loaded from file", True, f"{len(episode.clues)} clues")
    elif args.live:
        # Run full LLM pipeline
        if not os.environ.get("GEMINI_API_KEY"):
            result.check("API: GEMINI_API_KEY set", False, "missing env var")
            print(result.summary())
            sys.exit(1)

        # Auto-detect audio slice if not provided
        if not args.audio_slice:
            gpu_dir = os.path.dirname(args.gpu_output)
            mp3_files = [f for f in os.listdir(gpu_dir or ".") if f.endswith("_interview_slice.mp3")]
            if mp3_files:
                args.audio_slice = os.path.join(gpu_dir or ".", mp3_files[0])
                print(f"  Auto-detected audio slice: {args.audio_slice}")

        episode = asyncio.run(run_live_pipeline(segments, args.audio_slice, result))
    else:
        # Dry-run: check if episode JSON exists alongside GPU output
        gpu_dir = os.path.dirname(args.gpu_output) or "."
        episode_files = [f for f in os.listdir(gpu_dir) if f.startswith("episode_") and f.endswith(".json")]
        if episode_files:
            from trebek.schemas import Episode

            ep_path = os.path.join(gpu_dir, episode_files[0])
            print(f"\n── Found existing episode JSON: {ep_path}")
            with open(ep_path) as f:
                episode = Episode.model_validate_json(f.read())
            result.check("Episode: loaded from existing file", True, f"{len(episode.clues)} clues")

    # ── Post-extraction validation ────────────────────────────────
    if episode:
        validate_episode(episode, result, segments)
        validate_state_machine(episode, result)
        validate_db_commit_dry_run(episode, result)
    else:
        print("\n  ℹ️  No episode data available for post-extraction validation")
        print("     Run with --live or --episode-json to validate extraction output")

    # ── Summary ───────────────────────────────────────────────────
    print(result.summary())

    # Exit code
    sys.exit(1 if result.failed > 0 else 0)


if __name__ == "__main__":
    main()
