#!/usr/bin/env python3
"""
Standalone Trebek LLM Pipeline Validator — zero trebek imports.
Copy the validate_llm_pipeline/ folder to a remote machine, pip install google-genai pydantic, run.

Usage:
  # Auto-detect files in gpu_outputs/
  python -m validate_llm_pipeline --dir gpu_outputs/

  # Or run the folder directly
  python validate_llm_pipeline/ --dir gpu_outputs/

  # Explicit paths
  python validate_llm_pipeline/ \
    --transcript gpu_outputs/gpu_output_*.json.gz \
    --audio gpu_outputs/*_interview_slice.mp3

  # Skip Pass 1 (no audio slice)
  python validate_llm_pipeline/ --transcript gpu_outputs/gpu_output_*.json.gz --skip-pass1

  # Validate existing episode JSON only (no API calls)
  python validate_llm_pipeline/ --episode-json gpu_outputs/episode_*.json

Requires: pip install google-genai pydantic
Env: GEMINI_API_KEY must be set
"""

import argparse
import asyncio
import glob
import gzip
import json
import os
import sys

try:
    from .schemas import Results
    from .extraction import validate_gpu, get_client, run_pass1, run_pass2
    from .diagnostics import (
        print_transcript_intelligence,
        print_round_summary_table,
        print_board_coverage,
        print_contestant_matrix,
        print_api_summary,
        validate_extraction,
        validate_state_machine,
        validate_episode_json,
    )
except ImportError:
    # Direct execution: python scripts/validate_llm_pipeline/
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from validate_llm_pipeline.schemas import Results
    from validate_llm_pipeline.extraction import validate_gpu, get_client, run_pass1, run_pass2
    from validate_llm_pipeline.diagnostics import (
        print_transcript_intelligence,
        print_round_summary_table,
        print_board_coverage,
        print_contestant_matrix,
        print_api_summary,
        validate_extraction,
        validate_state_machine,
        validate_episode_json,
    )


def main():
    ap = argparse.ArgumentParser(description="Standalone Trebek LLM Pipeline Validator")
    ap.add_argument("--dir", help="Directory containing GPU outputs (auto-detect files)")
    ap.add_argument("--transcript", help="Path to gpu_output_*.json.gz")
    ap.add_argument("--audio", help="Path to *_interview_slice.mp3")
    ap.add_argument("--episode-json", help="Validate existing episode JSON only (no API)")
    ap.add_argument("--skip-pass1", action="store_true", help="Skip Pass 1 speaker anchoring")
    ap.add_argument("--save", help="Save extracted episode to this JSON path")
    args = ap.parse_args()

    R = Results()
    print("=" * 60)
    print("  Trebek LLM Pipeline Standalone Validator")
    print("=" * 60)

    # Auto-detect files
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"  ❌ Directory not found: {args.dir}")
            sys.exit(1)

        # List actual files for debugging
        all_files = os.listdir(args.dir)
        print(f"  📁 Directory '{args.dir}' contains {len(all_files)} files:")
        for f in sorted(all_files):
            print(f"     {f}")

        # Auto-detect transcript (.json.gz or .json with 'gpu_output' prefix)
        gzs = glob.glob(os.path.join(args.dir, "gpu_output_*.json.gz"))
        if not gzs:
            gzs = glob.glob(os.path.join(args.dir, "*.json.gz"))
        if not gzs:
            # Fall back to any large JSON that looks like a transcript
            gzs = [os.path.join(args.dir, f) for f in all_files if f.endswith(".json.gz")]

        # Auto-detect audio slice
        mp3s = glob.glob(os.path.join(args.dir, "*_interview_slice.mp3"))
        if not mp3s:
            mp3s = [os.path.join(args.dir, f) for f in all_files if f.endswith(".mp3")]

        # Auto-detect episode JSON
        epjs = glob.glob(os.path.join(args.dir, "episode_*.json"))
        if not epjs:
            epjs = [os.path.join(args.dir, f) for f in all_files if f.endswith(".json") and not f.endswith(".json.gz")]

        if gzs and not args.transcript:
            args.transcript = gzs[0]
        if mp3s and not args.audio:
            args.audio = mp3s[0]
        if epjs and not args.episode_json and not args.transcript:
            args.episode_json = epjs[0]
        print("\n  Auto-detected:")
        print(f"    transcript = {args.transcript}")
        print(f"    audio      = {args.audio}")
        print(f"    episode    = {args.episode_json}")

    # Mode 1: Validate existing episode JSON only
    if args.episode_json and not args.transcript:
        validate_episode_json(args.episode_json, R)
        fail = R.summary()
        sys.exit(1 if fail else 0)

    if not args.transcript:
        print("❌ No --transcript or --dir provided")
        sys.exit(1)

    # Load GPU output
    print(f"\n  Loading: {args.transcript}")
    if args.transcript.endswith(".gz"):
        with gzip.open(args.transcript, "rt") as f:
            gpu = json.load(f)
    else:
        with open(args.transcript) as f:
            gpu = json.load(f)
    R.check("GPU: file loaded", True, f"{os.path.getsize(args.transcript):,} bytes")
    segs = validate_gpu(gpu, R)
    if not segs:
        print("\n❌ No segments")
        R.summary()
        sys.exit(1)

    # Transcript diagnostics (free — no LLM calls)
    print_transcript_intelligence(segs, R)

    # Run LLM pipeline
    async def _run():
        client = get_client()
        mapping = {}
        api_calls_all = []
        if not args.skip_pass1 and args.audio and os.path.exists(args.audio):
            mapping = await run_pass1(client, args.audio, R)
        elif args.skip_pass1:
            R.warn("Pass1 skipped by flag")
        else:
            R.warn("Pass1 skipped — no audio slice found")
        meta, clues, api_calls = await run_pass2(client, segs, mapping, R)
        api_calls_all.extend(api_calls)
        return meta, clues, api_calls_all

    meta, clues, api_calls = asyncio.run(_run())
    validate_extraction(meta, clues, R)

    # ── Dense diagnostic sections ──────────────────────────────────
    print_round_summary_table(clues, meta)
    print_board_coverage(clues, R)

    scores = validate_state_machine(meta, clues, R)
    print_contestant_matrix(scores, clues, meta)
    print_api_summary(api_calls)

    # Save output
    if args.save:
        out = {"meta": meta.model_dump(), "clues": clues, "scores": scores}
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  💾 Saved to {args.save}")

    fail = R.summary()
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
