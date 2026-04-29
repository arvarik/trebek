#!/usr/bin/env python3
"""
Local LLM Extraction Test Runner.

Auto-discovers episodes from gpu_outputs/ and runs the full extraction
pipeline locally (Pass 1 + Pass 2 + State Machine verification).
No database, no Docker, no GPU required.

File Discovery:
    Each episode in gpu_outputs/ is identified by its .json.gz transcript.
    The script derives the episode name from the video_filepath stored
    inside the .json.gz, then locates the matching _interview_slice.mp3
    for Pass 1 speaker anchoring.

Usage:
    # Run all discovered episodes:
    python scripts/test_local_extraction.py

    # Run a specific episode (substring match):
    python scripts/test_local_extraction.py --episode S42E125

    # Skip Pass 1 (faster iteration on Pass 2 changes):
    python scripts/test_local_extraction.py --skip-pass1

    # Save extracted JSON outputs:
    python scripts/test_local_extraction.py --save

    # Validate previously extracted episode JSONs (no LLM calls):
    python scripts/test_local_extraction.py --validate-only

Prerequisites:
    - .env with GEMINI_API_KEY in the repo root
    - gpu_outputs/ directory with rsync'd files from the GPU server
"""

import asyncio
import argparse
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── Bootstrap ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env file (strips inline # comments)
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            if "#" in _val:
                _val = _val[: _val.index("#")]
            os.environ.setdefault(_key.strip(), _val.strip().strip("'\""))

GPU_OUTPUT_DIR = REPO_ROOT / "gpu_outputs"


# ── Episode Discovery ────────────────────────────────────────────────


def _derive_episode_name(video_filepath: str) -> str:
    """Extract a clean episode name from the video_filepath in the .json.gz.

    Examples:
        '/home/arvarik/trebek_test/Jeopardy (1984) - S42E125 - Jeopardy .ts'
        → 'Jeopardy_(1984)_-_S42E125_-_Jeopardy_'
    """
    stem = Path(video_filepath).stem  # strip extension
    return stem.replace(" ", "_")


def _extract_short_id(episode_name: str) -> str:
    """Extract a human-friendly short ID like 'S42E125' from the full name."""
    match = re.search(r"S\d+E\d+", episode_name)
    return match.group(0) if match else episode_name[:20]


def discover_episodes() -> list[dict[str, Any]]:
    """Auto-discover episodes from gpu_outputs/ by scanning .json.gz files.

    Returns a list of episode dicts with keys:
        - gpu_file: str (filename of .json.gz)
        - episode_name: str (derived from video_filepath)
        - short_id: str (e.g. 'S42E125')
        - audio_file: Optional[str] (matching _interview_slice.mp3, or None)
    """
    if not GPU_OUTPUT_DIR.exists():
        return []

    # Index all .mp3 files for fast lookup
    mp3_files = {f for f in os.listdir(GPU_OUTPUT_DIR) if f.endswith(".mp3")}

    episodes = []
    for fname in sorted(os.listdir(GPU_OUTPUT_DIR)):
        if not fname.endswith(".json.gz"):
            continue

        # Read video_filepath from inside the .json.gz
        gz_path = GPU_OUTPUT_DIR / fname
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                header = json.load(f)
            video_filepath = header.get("video_filepath", "")
        except Exception:
            continue

        if not video_filepath:
            continue

        episode_name = _derive_episode_name(video_filepath)
        short_id = _extract_short_id(episode_name)

        # Find matching audio file: {episode_name}_interview_slice.mp3
        audio_file: Optional[str] = None
        expected_mp3 = f"{episode_name}_interview_slice.mp3"
        if expected_mp3 in mp3_files:
            audio_file = expected_mp3

        episodes.append(
            {
                "gpu_file": fname,
                "episode_name": episode_name,
                "short_id": short_id,
                "audio_file": audio_file,
            }
        )

    return episodes


# ── Display Helpers ───────────────────────────────────────────────────


def print_header(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_category_grid(clues: list[Any], round_name: str) -> None:
    """Print a visual category completeness grid."""
    round_clues = [c for c in clues if c.round == round_name]
    categories: dict[str, set[int]] = {}
    for c in round_clues:
        categories.setdefault(c.category, set()).add(c.board_row)

    print(f"\n  {round_name} Category Grid:")
    print(f"  {'Category':<35} {'R1':>3} {'R2':>3} {'R3':>3} {'R4':>3} {'R5':>3} {'Total':>6}")
    print(f"  {'-' * 35} {'-' * 3} {'-' * 3} {'-' * 3} {'-' * 3} {'-' * 3} {'-' * 6}")
    for cat, rows in sorted(categories.items()):
        row_marks = ["✅" if r in rows else "❌" for r in range(1, 6)]
        total = f"{len(rows)}/5"
        status = "✅" if len(rows) == 5 else "⚠️"
        print(
            f"  {cat:<35} {row_marks[0]:>3} {row_marks[1]:>3} {row_marks[2]:>3} {row_marks[3]:>3} {row_marks[4]:>3} {total:>4} {status}"
        )


def print_episode_report(episode: Any, quality: str, elapsed_ms: float, label: str) -> None:
    """Print a detailed extraction report for one episode."""
    print_header(f"Report: {label}")

    j_clues = [c for c in episode.clues if c.round == "J!"]
    dj_clues = [c for c in episode.clues if c.round == "Double J!"]

    print("\n  📊 Summary:")
    print(f"     Quality:       {quality}")
    print(f"     Elapsed:       {elapsed_ms:.0f}ms")
    print(f"     Host:          {episode.host_name}")
    print(f"     Date:          {episode.episode_date}")
    print(f"     Tournament:    {episode.is_tournament}")
    print(f"     Contestants:   {[c.name for c in episode.contestants]}")
    print(f"     J! Clues:      {len(j_clues)}/30")
    print(f"     DJ! Clues:     {len(dj_clues)}/30")
    print(f"     Total Clues:   {len(episode.clues)}/60")
    print(f"     FJ Category:   {episode.final_jep.category}")

    # Category grids
    print_category_grid(episode.clues, "J!")
    print_category_grid(episode.clues, "Double J!")

    # Daily Doubles
    dd_clues = [c for c in episode.clues if c.is_daily_double]
    print(f"\n  🎯 Daily Doubles ({len(dd_clues)}):")
    for dd in dd_clues:
        print(f"     {dd.round} | {dd.category} | Wager: {dd.daily_double_wager} | Wagerer: {dd.wagerer_name}")

    # Verification stats (Stage 3.5)
    verified_count = sum(1 for c in episode.clues if getattr(c, "is_verified", False))
    corrected_count = sum(1 for c in episode.clues if getattr(c, "original_response", None) is not None)
    print("\n  🔍 Verification (Stage 3.5):")
    print(f"     Verified:      {verified_count}/{len(episode.clues)}")
    print(f"     Corrected:     {corrected_count}")

    # Buzz statistics
    total_buzzes = sum(len(c.attempts) for c in episode.clues)
    correct_buzzes = sum(1 for c in episode.clues for a in c.attempts if a.is_correct)
    triple_stumpers = sum(1 for c in episode.clues if len(c.attempts) == 0)
    print("\n  🔔 Buzz Stats:")
    print(f"     Total buzzes:    {total_buzzes}")
    print(f"     Correct:         {correct_buzzes}")
    print(f"     Triple Stumpers: {triple_stumpers}")

    # Per-contestant buzz counts
    contestant_buzzes: dict[str, dict[str, int]] = {}
    for c in episode.clues:
        for a in c.attempts:
            contestant_buzzes.setdefault(a.speaker, {"correct": 0, "incorrect": 0, "total": 0})
            contestant_buzzes[a.speaker]["total"] += 1
            if a.is_correct:
                contestant_buzzes[a.speaker]["correct"] += 1
            else:
                contestant_buzzes[a.speaker]["incorrect"] += 1

    if contestant_buzzes:
        print("\n  👤 Per-Contestant Buzzes:")
        for name, stats in sorted(contestant_buzzes.items()):
            print(f"     {name:<25} {stats['correct']:>3}✅  {stats['incorrect']:>3}❌  ({stats['total']} total)")


def print_validation_report(episode: Any, label: str) -> None:
    """Print a validation-only report for a previously extracted episode."""
    from trebek.state_machine import TrebekStateMachine
    from trebek.llm.validation import _validate_extraction_integrity

    print_header(f"Validation: {label}")

    # Run state machine
    valid_contestants = {c.name for c in episode.contestants}
    sm = TrebekStateMachine(valid_contestants=valid_contestants)
    sm.load_adjustments(episode.score_adjustments)
    for clue in episode.clues:
        sm.process_clue(clue)
    sm.process_final_jep(episode.final_jep)

    # Run integrity checks
    warnings = _validate_extraction_integrity(episode)

    j_count = sum(1 for c in episode.clues if c.round == "J!")
    dj_count = sum(1 for c in episode.clues if c.round == "Double J!")
    dd_count = sum(1 for c in episode.clues if c.is_daily_double)
    verified_count = sum(1 for c in episode.clues if getattr(c, "is_verified", False))
    corrected_count = sum(1 for c in episode.clues if getattr(c, "original_response", None) is not None)

    print("\n  📊 Structure:")
    print(f"     Host:            {episode.host_name}")
    print(f"     Contestants:     {[c.name for c in episode.contestants]}")
    print(f"     Clues:           {len(episode.clues)}/60 (J!={j_count}, DJ!={dj_count})")
    print(f"     Daily Doubles:   {dd_count}/3")
    print(f"     Verified:        {verified_count}/{len(episode.clues)}")
    print(f"     Corrected:       {corrected_count}")

    print("\n  🏆 State Machine Scores:")
    for c in episode.contestants:
        final = sm.scores.get(c.name, 0)
        coryat = sm.coryat_scores.get(c.name, 0)
        print(f"     {c.name:<25} Final: ${final:>7,}   Coryat: ${coryat:>7,}")

    if warnings:
        print(f"\n  ⚠️  Integrity Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"     • {w}")
    else:
        print("\n  ✅ All integrity checks passed")

    # Quality gate (same logic as state_machine_worker)
    weighted = sum(0.5 if "duplicate board_row" in w else 1.0 for w in warnings)
    quality = "PASS" if not warnings else ("DEGRADED" if weighted <= 3.0 else "FAIL")
    if len(episode.clues) < 45:
        quality = "FAIL"
    print(f"\n  📋 Quality: {quality}")

    return quality


# ── Extraction Runner ─────────────────────────────────────────────────


def load_gpu_transcript(gpu_file: str) -> tuple[list[dict[str, Any]], str]:
    """Load GPU transcript output and return (segments, video_filepath)."""
    path = GPU_OUTPUT_DIR / gpu_file
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("transcript", {}).get("segments", [])
    video_filepath = data.get("video_filepath", "unknown")
    return segments, video_filepath


async def run_episode(
    episode: dict[str, Any],
    skip_pass1: bool = False,
    save: bool = False,
) -> tuple[Any, str, float]:
    """Run extraction for a single episode and return (episode_data, quality, elapsed_ms)."""
    from trebek.llm import execute_pass_1_speaker_anchoring, execute_pass_2_data_extraction

    label = episode["short_id"]
    print_header(f"Extracting: {label} ({episode['episode_name']})")

    # Load GPU output
    segments, video_path = load_gpu_transcript(episode["gpu_file"])
    print(f"  📄 Loaded {len(segments)} segments from {episode['gpu_file']}")

    start = time.perf_counter()

    # Pass 1: Speaker Anchoring
    if skip_pass1:
        print("  ⏭️  Skipping Pass 1 (--skip-pass1)")
        speaker_mapping: dict[str, str] = {}
    elif episode["audio_file"] is None:
        print("  ⚠️  No audio file found — skipping Pass 1")
        speaker_mapping = {}
    else:
        audio_path = str(GPU_OUTPUT_DIR / episode["audio_file"])
        print("  🎤 Running Pass 1 (audio anchoring)...")
        speaker_mapping, usage1 = await execute_pass_1_speaker_anchoring(audio_path)
        print(f"  ✅ Pass 1 complete: {speaker_mapping}")
        print(f"     Tokens: in={usage1.get('input_tokens', 0):.0f} out={usage1.get('output_tokens', 0):.0f}")

    # Pass 2: Structured Extraction
    print("  📝 Running Pass 2 (Manifest-Verify-Fill extraction)...")
    episode_data, usage2, retries, quality = await execute_pass_2_data_extraction(
        segments,
        speaker_mapping,
    )

    elapsed = (time.perf_counter() - start) * 1000

    print(f"  ✅ Pass 2 complete: {len(episode_data.clues)} clues, quality={quality}")
    print(f"     Tokens: in={usage2.get('input_tokens', 0):.0f} out={usage2.get('output_tokens', 0):.0f}")
    print(f"     Cost:   ${usage2.get('cost_usd', 0):.4f}")
    print(f"     Retries: {retries}")

    # Save output
    if save:
        out_path = GPU_OUTPUT_DIR / f"local_episode_{label}.json"
        with open(out_path, "w") as f:
            f.write(episode_data.model_dump_json(indent=2))
        print(f"  💾 Saved to {out_path}")

    return episode_data, quality, elapsed


# ── Main ──────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local LLM extraction test runner — auto-discovers episodes from gpu_outputs/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--episode",
        type=str,
        help="Filter to episodes matching this substring (e.g. 'S42E125')",
    )
    parser.add_argument(
        "--skip-pass1",
        action="store_true",
        help="Skip Pass 1 speaker anchoring (faster iteration on Pass 2)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save extracted episode JSON to gpu_outputs/",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing episode JSONs without running LLM extraction",
    )
    args = parser.parse_args()

    # Check prerequisites
    if not GPU_OUTPUT_DIR.exists():
        print("❌ gpu_outputs/ not found. Rsync from server first:")
        print(f"   rsync -avz arvarik@192.168.4.240:/opt/stacks/trebek/gpu_outputs/ {GPU_OUTPUT_DIR}/")
        sys.exit(1)

    # ── Validate-only mode: use existing episode_*.json files ─────────
    if args.validate_only:
        from trebek.schemas import Episode

        episode_jsons = sorted(
            f for f in os.listdir(GPU_OUTPUT_DIR) if f.startswith("episode_") and f.endswith(".json")
        )

        if args.episode:
            episode_jsons = [f for f in episode_jsons if args.episode in f]

        if not episode_jsons:
            print("❌ No episode_*.json files found in gpu_outputs/")
            sys.exit(1)

        print(f"📋 Validating {len(episode_jsons)} episode(s)...\n")

        results = []
        for fname in episode_jsons:
            path = GPU_OUTPUT_DIR / fname
            episode_data = Episode.model_validate_json(path.read_text(encoding="utf-8"))
            short_id = _extract_short_id(fname)
            quality = print_validation_report(episode_data, short_id)
            results.append((short_id, episode_data, quality))

        if len(results) > 1:
            print_header("Validation Summary")
            print(f"  {'Episode':<10} {'Clues':<7} {'DD':<4} {'Verified':<10} {'Quality':<10}")
            print(f"  {'-' * 10} {'-' * 7} {'-' * 4} {'-' * 10} {'-' * 10}")
            for sid, ep, q in results:
                clue_count = len(ep.clues)
                dd_count = sum(1 for c in ep.clues if c.is_daily_double)
                verified = sum(1 for c in ep.clues if getattr(c, "is_verified", False))
                icon = "✅" if q == "PASS" else ("⚠️" if q == "DEGRADED" else "❌")
                print(f"  {sid:<10} {clue_count:<7} {dd_count:<4} {verified}/{clue_count:<6} {icon} {q}")

        return

    # ── Full extraction mode ──────────────────────────────────────────
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not set. Create a .env file in the repo root.")
        sys.exit(1)

    # Discover episodes
    episodes = discover_episodes()
    if not episodes:
        print("❌ No .json.gz files found in gpu_outputs/")
        sys.exit(1)

    if args.episode:
        episodes = [e for e in episodes if args.episode in e["episode_name"] or args.episode in e["short_id"]]
        if not episodes:
            print(f"❌ No episodes matching '{args.episode}'")
            sys.exit(1)

    print(f"🔍 Discovered {len(episodes)} episode(s):")
    for ep in episodes:
        audio_status = "🎤" if ep["audio_file"] else "⚠️ no audio"
        print(f"   {ep['short_id']:<10} {ep['episode_name']} ({audio_status})")

    results: list[tuple[str, Any, str, float]] = []
    for ep in episodes:
        try:
            episode_data, quality, elapsed = await run_episode(ep, args.skip_pass1, args.save)
            results.append((ep["short_id"], episode_data, quality, elapsed))
            print_episode_report(episode_data, quality, elapsed, ep["short_id"])
        except Exception as e:
            print(f"\n  ❌ {ep['short_id']} FAILED: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    # Summary table
    if len(results) > 1:
        print_header("Summary")
        print(f"  {'Episode':<10} {'J!':<6} {'DJ!':<6} {'Total':<7} {'Verified':<10} {'Quality':<10} {'Time':>8}")
        print(f"  {'-' * 10} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 10} {'-' * 10} {'-' * 8}")
        for sid, ep, q, ms in results:
            j = sum(1 for c in ep.clues if c.round == "J!")
            dj = sum(1 for c in ep.clues if c.round == "Double J!")
            verified = sum(1 for c in ep.clues if getattr(c, "is_verified", False))
            icon = "✅" if q == "PASS" else ("⚠️" if q == "DEGRADED" else "❌")
            print(f"  {sid:<10} {j:<6} {dj:<6} {j + dj:<7} {verified}/{j + dj:<6} {icon} {q:<8} {ms:>7.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
