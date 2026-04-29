#!/usr/bin/env python3
"""
Local LLM Extraction Test Runner.

Runs Pass 1 + Pass 2 extraction locally using pre-downloaded gpu_outputs/
from the GPU server. Bypasses the full pipeline (no DB, no Docker, no GPU).

Usage:
    # Run all 3 test episodes:
    python scripts/test_local_extraction.py

    # Run a single episode:
    python scripts/test_local_extraction.py --episode S42E04

    # Skip Pass 1 (faster iteration on Pass 2 changes):
    python scripts/test_local_extraction.py --skip-pass1

    # Save outputs for comparison:
    python scripts/test_local_extraction.py --save

Prerequisites:
    - .env with GEMINI_API_KEY in the repo root
    - gpu_outputs/ directory with GPU output files (scp from server)
"""

import asyncio
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env file (strips inline # comments)
env_path = REPO_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            # Strip inline comments: "INFO  # comment" → "INFO"
            if "#" in value:
                value = value[: value.index("#")]
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))

GPU_OUTPUT_DIR = REPO_ROOT / "gpu_outputs"

# Map episode shortnames to GPU output files
EPISODE_MAP = {
    "S42E04": {
        "gpu_file": "gpu_output_82577109966145228bcc0c190c7e5921.json.gz",
        "audio_file": "Jeopardy_(1984)_-_S42E04_-_Jeopardy__interview_slice.mp3",
        "label": "S42E04 (the problem child — Paulo/Paolo, under-extraction)",
    },
    "S42E18": {
        "gpu_file": "gpu_output_5d8ecedfa3384569bd7e6fa7e8255d11.json.gz",
        "audio_file": "Jeopardy_(1984)_-_S42E18_-_Jeopardy__interview_slice.mp3",
        "label": "S42E18 (host misidentification test)",
    },
    "S42E25": {
        "gpu_file": "gpu_output_512761594c5a4e34b6cf3df975365341.json.gz",
        "audio_file": "Jeopardy_(1984)_-_S42E25_-_Jeopardy__interview_slice.mp3",
        "label": "S42E25 (clean baseline)",
    },
}


def load_gpu_output(gpu_file: str) -> tuple[list[dict[str, Any]], str]:
    """Load GPU transcript output and return (segments, video_filepath)."""
    path = GPU_OUTPUT_DIR / gpu_file
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("transcript", {}).get("segments", [])
    video_filepath = data.get("video_filepath", "unknown")
    return segments, video_filepath


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


async def run_episode(
    episode_key: str,
    episode_info: dict[str, str],
    skip_pass1: bool = False,
    save: bool = False,
) -> tuple[Any, str, float]:
    """Run extraction for a single episode and return (episode, quality, elapsed_ms)."""
    from trebek.llm import execute_pass_1_speaker_anchoring, execute_pass_2_data_extraction

    print_header(f"Extracting: {episode_info['label']}")

    # Load GPU output
    segments, video_path = load_gpu_output(episode_info["gpu_file"])
    print(f"  📄 Loaded {len(segments)} segments from {episode_info['gpu_file']}")

    start = time.perf_counter()

    # Pass 1: Speaker Anchoring
    if skip_pass1:
        print("  ⏭️  Skipping Pass 1 (--skip-pass1)")
        speaker_mapping: dict[str, str] = {}
    else:
        audio_path = str(GPU_OUTPUT_DIR / episode_info["audio_file"])
        if not os.path.exists(audio_path):
            print(f"  ⚠️  Audio file not found: {audio_path} — skipping Pass 1")
            speaker_mapping = {}
        else:
            print("  🎤 Running Pass 1 (audio anchoring)...")
            speaker_mapping, usage1 = await execute_pass_1_speaker_anchoring(audio_path)
            print(f"  ✅ Pass 1 complete: {speaker_mapping}")
            print(f"     Tokens: in={usage1.get('input_tokens', 0)} out={usage1.get('output_tokens', 0)}")

    # Pass 2: Structured Extraction
    print("  📝 Running Pass 2 (Manifest-Verify-Fill extraction)...")
    episode, usage2, retries, quality = await execute_pass_2_data_extraction(
        segments,
        speaker_mapping,
    )

    elapsed = (time.perf_counter() - start) * 1000

    print(f"  ✅ Pass 2 complete: {len(episode.clues)} clues, quality={quality}")
    print(f"     Tokens: in={usage2.get('input_tokens', 0)} out={usage2.get('output_tokens', 0)}")
    print(f"     Cost:   ${usage2.get('cost_usd', 0):.4f}")
    print(f"     Retries: {retries}")

    # Save output
    if save:
        out_path = GPU_OUTPUT_DIR / f"local_episode_{episode_key}.json"
        with open(out_path, "w") as f:
            f.write(episode.model_dump_json(indent=2))
        print(f"  💾 Saved to {out_path}")

    return episode, quality, elapsed


async def main() -> None:
    parser = argparse.ArgumentParser(description="Local LLM extraction test runner")
    parser.add_argument(
        "--episode",
        choices=list(EPISODE_MAP.keys()),
        help="Run a single episode (default: all 3)",
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
    args = parser.parse_args()

    # Check prerequisites
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not set. Create a .env file in the repo root.")
        sys.exit(1)

    if not GPU_OUTPUT_DIR.exists():
        print("❌ gpu_outputs/ not found. SCP from server first:")
        print(f"   scp -r arvarik@192.168.4.240:/opt/stacks/trebek/gpu_outputs {REPO_ROOT}/")
        sys.exit(1)

    episodes_to_run = {args.episode: EPISODE_MAP[args.episode]} if args.episode else EPISODE_MAP

    results: list[tuple[str, Any, str, float]] = []
    for key, info in episodes_to_run.items():
        try:
            episode, quality, elapsed = await run_episode(key, info, args.skip_pass1, args.save)
            results.append((key, episode, quality, elapsed))
            print_episode_report(episode, quality, elapsed, info["label"])
        except Exception as e:
            print(f"\n  ❌ {key} FAILED: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    # Summary table
    if len(results) > 1:
        print_header("Summary")
        print(f"  {'Episode':<10} {'J!':<6} {'DJ!':<6} {'Total':<7} {'Quality':<10} {'Time':>8}")
        print(f"  {'-' * 10} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 10} {'-' * 8}")
        for key, ep, q, ms in results:
            j = sum(1 for c in ep.clues if c.round == "J!")
            dj = sum(1 for c in ep.clues if c.round == "Double J!")
            icon = "✅" if q == "PASS" else ("⚠️" if q == "DEGRADED" else "❌")
            print(f"  {key:<10} {j:<6} {dj:<6} {j + dj:<7} {icon} {q:<8} {ms:>7.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
