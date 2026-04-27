#!/usr/bin/env python3
"""
Standalone Trebek LLM Pipeline Validator — zero trebek imports.
Copy to remote machine, pip install google-genai pydantic, run.

Usage:
  # Auto-detect files in gpu_outputs/
  python validate_llm_pipeline.py --dir gpu_outputs/

  # Explicit paths
  python validate_llm_pipeline.py \
    --transcript gpu_outputs/gpu_output_*.json.gz \
    --audio gpu_outputs/*_interview_slice.mp3

  # Skip Pass 1 (no audio slice)
  python validate_llm_pipeline.py --transcript gpu_outputs/gpu_output_*.json.gz --skip-pass1

  # Validate existing episode JSON only (no API calls)
  python validate_llm_pipeline.py --episode-json gpu_outputs/episode_*.json

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
import time
from typing import Optional, Literal
from pydantic import BaseModel, Field

# ── Pydantic Schemas (inlined from trebek) ─────────────────────────


class BuzzAttemptExt(BaseModel):
    attempt_order: int
    speaker: str
    response_given: str
    is_correct: bool
    buzz_line_id: str
    is_lockout_inferred: bool


class ClueExt(BaseModel):
    round: Literal["Jeopardy", "Double Jeopardy", "Final Jeopardy", "Tiebreaker"]
    category: str
    board_row: int
    board_col: int
    is_daily_double: bool
    requires_visual_context: bool
    host_read_start_line_id: str
    host_read_end_line_id: str
    daily_double_wager: Optional[str] = None
    wagerer_name: Optional[str] = None
    correct_response: str
    attempts: list[BuzzAttemptExt] = []


class PartialClues(BaseModel):
    clues: list[ClueExt]


class FJWager(BaseModel):
    contestant: str
    wager: int
    response: str
    is_correct: bool


class FJ(BaseModel):
    category: str
    clue_text: str
    wagers_and_responses: list[FJWager]


class Contestant(BaseModel):
    name: str
    podium_position: int = Field(ge=1, le=3)
    occupational_category: str
    is_returning_champion: bool
    description: str


class ScoreAdj(BaseModel):
    contestant: str
    points_adjusted: int
    reason: str
    effective_after_clue_selection_order: int


class PartialMeta(BaseModel):
    episode_date: str
    host_name: str
    is_tournament: bool
    contestants: list[Contestant]
    final_jeopardy: FJ
    score_adjustments: list[ScoreAdj]


# ── Result Tracker ─────────────────────────────────────────────────


class Results:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail=""):
        self.checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name, detail=""):
        self.checks.append({"name": name, "status": "WARN", "detail": detail})
        print(f"  ⚠️  {name} — {detail}")

    def summary(self):
        p = sum(1 for c in self.checks if c["status"] == "PASS")
        f = sum(1 for c in self.checks if c["status"] == "FAIL")
        w = sum(1 for c in self.checks if c["status"] == "WARN")
        t = len(self.checks)
        print(f"\n{'=' * 60}\n  RESULTS: {p}/{t} passed, {f} failed, {w} warnings\n{'=' * 60}")
        return f


# ── GPU Output Validation ──────────────────────────────────────────


def validate_gpu(gpu_data, R):
    print("\n── GPU Output Structure ──────────────────────────────")
    tr = gpu_data.get("transcript", {})
    segs = tr.get("segments", [])
    R.check("GPU: transcript key exists", bool(tr))
    R.check("GPU: segments non-empty", len(segs) > 0, f"{len(segs)} segments")
    if not segs:
        return []
    R.check("GPU: has text/start/end/speaker", all(k in segs[0] for k in ["text", "start", "end", "speaker"]))
    hs = sum(1 for s in segs if s.get("start") is not None)
    R.check("GPU: >95% have timestamps", hs / len(segs) > 0.95, f"{hs}/{len(segs)}")
    spk = {s.get("speaker", "?") for s in segs}
    R.check("GPU: multiple speakers", len(spk) >= 2, f"{sorted(spk)}")
    last = max((s.get("end", 0) or 0) for s in segs)
    R.check("GPU: duration 15-35 min", 15 <= last / 60 <= 35, f"{last / 60:.1f}m")
    txt = " ".join(s.get("text", "") for s in segs).lower()
    R.check("GPU: Jeopardy markers found", any(m in txt for m in ["jeopardy", "daily double", "what is"]))
    # Monotonicity
    bad = sum(
        1 for i in range(1, len(segs)) if (segs[i].get("start", 0) or 0) < (segs[i - 1].get("start", 0) or 0) - 0.5
    )
    R.check("GPU: timestamps monotonic", bad < len(segs) * 0.02, f"{bad} violations")
    return segs


# ── Gemini API Helpers ─────────────────────────────────────────────


def get_client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


MODEL_FLASH = "gemini-3.1-flash-lite-preview"
MODEL_PRO = "gemini-3.1-pro-preview"


async def call_gemini(client, model, prompt, system, schema_cls=None, max_tokens=65536, ctx=""):
    from google.genai import types

    kw = {"temperature": 0.0, "max_output_tokens": max_tokens}
    if not isinstance(prompt, list):
        kw["system_instruction"] = system
    if schema_cls:
        kw["response_mime_type"] = "application/json"
        kw["response_json_schema"] = schema_cls.model_json_schema()
        kw["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    else:
        kw["response_mime_type"] = "application/json"
    cfg = types.GenerateContentConfig(**kw)
    t0 = time.perf_counter()
    resp = await client.aio.models.generate_content(model=model, contents=prompt, config=cfg)
    ms = (time.perf_counter() - t0) * 1000
    usage = {}
    if hasattr(resp, "usage_metadata") and resp.usage_metadata:
        usage = {
            "input": getattr(resp.usage_metadata, "prompt_token_count", 0) or 0,
            "output": getattr(resp.usage_metadata, "candidates_token_count", 0) or 0,
            "thinking": getattr(resp.usage_metadata, "thoughts_token_count", 0) or 0,
        }
    usage["ms"] = ms
    return resp, usage


# ── Pass 1: Speaker Anchoring ──────────────────────────────────────


async def run_pass1(client, audio_path, R):
    print("\n── Pass 1: Speaker Anchoring ─────────────────────────")
    sys_p = (
        "You are a strict data extractor. Analyze the audio of the Jeopardy host interview. "
        "Map distinct vocal timbres to Diarization Speaker IDs. "
        'Return a pure JSON dict: {"SPEAKER_00":"Name","SPEAKER_01":"Name"}. No markdown.'
    )
    t0 = time.perf_counter()
    uploaded = await asyncio.to_thread(client.files.upload, file=audio_path)
    prompt = ["Map speakers to SPEAKER_XX IDs from this interview audio.", uploaded]
    resp, usage = await call_gemini(client, MODEL_FLASH, prompt, sys_p, max_tokens=2048, ctx="Pass1")
    await asyncio.to_thread(client.files.delete, name=uploaded.name)
    ms = (time.perf_counter() - t0) * 1000
    clean = str(resp.text).replace("```json", "").replace("```", "").strip()
    try:
        mapping = json.loads(clean)
    except json.JSONDecodeError:
        import ast

        mapping = dict(ast.literal_eval(clean))
    R.check("Pass1: returned mapping", isinstance(mapping, dict), f"{mapping}")
    R.check("Pass1: 2+ speakers", len(mapping) >= 2, f"{len(mapping)} speakers")
    R.check("Pass1: latency <30s", ms < 30000, f"{ms:.0f}ms")
    print(f"  📊 Tokens: in={usage.get('input', 0)} out={usage.get('output', 0)} | {ms:.0f}ms")
    return mapping


# ── Pass 2: Structured Extraction ──────────────────────────────────


def abbrev(spk):
    return "S" + spk[len("SPEAKER_") :] if spk.startswith("SPEAKER_") else spk


def fmt_transcript(segs):
    return "\n".join(f"L{i} {abbrev(s.get('speaker', '?'))}: {s.get('text', '').strip()}" for i, s in enumerate(segs))


async def extract_with_retry(client, prompt, system, schema_cls, max_retries=3, ctx=""):
    for attempt in range(max_retries + 1):
        try:
            resp, usage = await call_gemini(client, MODEL_PRO, prompt, system, schema_cls, ctx=ctx)
            txt = str(resp.text).replace("```json", "").replace("```", "").strip()
            data = schema_cls.model_validate_json(txt)
            return data, usage, attempt
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"  ⚠️  Attempt {attempt + 1} failed: {str(e)[:120]}. Retrying...")
            await asyncio.sleep(2.0 * (2**attempt))
    raise RuntimeError("unreachable")


async def run_pass2(client, segs, speaker_mapping, R):
    print("\n── Pass 2: Structured Extraction ─────────────────────")
    comp_map = {abbrev(k): v for k, v in speaker_mapping.items()}
    transcript = fmt_transcript(segs)
    base_sys = (
        "You are Trebek, an expert Jeopardy data extraction pipeline. "
        f"CRITICAL: Map speakers using: {json.dumps(comp_map)}. "
        "Speaker IDs are abbreviated (S0=SPEAKER_00). "
        "Do NOT hallucinate names. Use Line IDs (e.g. L0, L105) for timestamps."
    )
    # Meta extraction
    print("  Extracting episode metadata...")
    meta_prompt = f"Transcript:\n{transcript}\n\nExtract episode_date, host_name, is_tournament, contestants, final_jeopardy, score_adjustments. Do NOT extract clues."

    t0 = time.perf_counter()
    resp, mu = await call_gemini(client, MODEL_PRO, meta_prompt, base_sys, PartialMeta, ctx="Meta")
    meta_txt = str(resp.text).replace("```json", "").replace("```", "").strip()
    meta = PartialMeta.model_validate_json(meta_txt)
    meta_ms = (time.perf_counter() - t0) * 1000
    R.check("Pass2 Meta: contestants", len(meta.contestants) > 0, f"{[c.name for c in meta.contestants]}")
    R.check("Pass2 Meta: host", bool(meta.host_name), meta.host_name)
    R.check("Pass2 Meta: FJ category", bool(meta.final_jeopardy.category), meta.final_jeopardy.category)
    print(f"  📊 Meta: {meta_ms:.0f}ms | in={mu.get('input', 0)} out={mu.get('output', 0)}")

    # Chunk + extract clues
    lines = transcript.split("\n")
    markers = ["double jeopardy", "final jeopardy", "tiebreaker"]
    chunks, cur = [], []
    for ln in lines:
        cur.append(ln)
        if any(m in ln.lower() for m in markers) and len(cur) > 10:
            chunks.append("\n".join(cur))
            cur = []
    if cur:
        chunks.append("\n".join(cur))
    print(f"  Extracting clues from {len(chunks)} chunks...")

    all_clues = []
    total_usage = {"input": 0, "output": 0, "thinking": 0, "ms": 0}
    for ci, chunk in enumerate(chunks):
        prompt = (
            f"Transcript Chunk ({ci + 1}/{len(chunks)}):\n{chunk}\n\n"
            "Extract ALL Jeopardy/Double Jeopardy clues. Do NOT extract Final Jeopardy. "
            "Skip clues cut off at chunk boundaries. Use Line IDs for timestamps."
        )
        cdata, cu, att = await extract_with_retry(client, prompt, base_sys, PartialClues, ctx=f"Chunk{ci + 1}")
        all_clues.extend(cdata.clues)
        for k in total_usage:
            total_usage[k] += cu.get(k, 0)
        print(f"    Chunk {ci + 1}/{len(chunks)}: {len(cdata.clues)} clues (attempt {att + 1})")

    R.check("Pass2: total clues > 0", len(all_clues) > 0, f"{len(all_clues)} raw clues")
    print(f"  📊 Clues: {total_usage['ms']:.0f}ms total | in={total_usage['input']} out={total_usage['output']}")

    # Reconstruct timestamps + build final clues
    clues_out = []
    dropped = 0
    for ec in all_clues:
        if ec.round == "Final Jeopardy":
            continue
        sid = ec.host_read_start_line_id.replace("L", "").replace("[", "").replace("]", "").strip()
        eid = ec.host_read_end_line_id.replace("L", "").replace("[", "").replace("]", "").strip()
        try:
            si, ei = int(sid), int(eid)
        except ValueError:
            dropped += 1
            continue
        si = max(0, min(si, len(segs) - 1))
        ei = max(0, min(ei, len(segs) - 1))
        if ei < si:
            ei = si
        text = " ".join(s.get("text", "").strip() for s in segs[si : ei + 1])
        rs = segs[si].get("start") or 0.0
        re_ = segs[ei].get("end") or 0.0
        clues_out.append(
            {
                "round": ec.round,
                "category": ec.category,
                "board_row": ec.board_row,
                "board_col": ec.board_col,
                "is_daily_double": ec.is_daily_double,
                "clue_text": text,
                "correct_response": ec.correct_response,
                "host_start_ms": float(rs) * 1000,
                "host_finish_ms": float(re_) * 1000,
                "daily_double_wager": ec.daily_double_wager,
                "wagerer_name": ec.wagerer_name,
                "attempts": [
                    {
                        "speaker": a.speaker,
                        "is_correct": a.is_correct,
                        "order": a.attempt_order,
                        "response": a.response_given,
                    }
                    for a in ec.attempts
                ],
            }
        )
    if dropped:
        R.warn(f"Pass2: dropped {dropped} clues with invalid Line IDs")

    # Dedup
    unique = {}
    for c in clues_out:
        bucket = round(c["host_start_ms"] / 2000) * 2000
        key = f"{bucket}_{c['round']}_{c['category'].lower().strip()}"
        if key not in unique or len(c["attempts"]) > len(unique[key]["attempts"]):
            unique[key] = c
    deduped = sorted(unique.values(), key=lambda c: c["host_start_ms"])
    for i, c in enumerate(deduped):
        c["selection_order"] = i + 1
    R.check(
        "Pass2: clues after dedup", len(deduped) > 0, f"{len(deduped)} (removed {len(clues_out) - len(deduped)} dupes)"
    )
    return meta, deduped


# ── Validation ─────────────────────────────────────────────────────


def validate_extraction(meta, clues, R):
    print("\n── Extraction Quality ───────────────────────────────")
    j = [c for c in clues if c["round"] == "Jeopardy"]
    dj = [c for c in clues if c["round"] == "Double Jeopardy"]
    R.check("Clues: J round 15-30", 15 <= len(j) <= 30, f"{len(j)}")
    R.check("Clues: DJ round 15-30", 15 <= len(dj) <= 30, f"{len(dj)}")
    dd = sum(1 for c in clues if c["is_daily_double"])
    R.check("Clues: 1-3 Daily Doubles", 1 <= dd <= 3, f"{dd}")
    jcats = {c["category"].lower().strip() for c in j}
    djcats = {c["category"].lower().strip() for c in dj}
    R.check("Clues: J has 5-6 categories", 5 <= len(jcats) <= 6, f"{len(jcats)}: {sorted(jcats)}")
    R.check("Clues: DJ has 5-6 categories", 5 <= len(djcats) <= 6, f"{len(djcats)}: {sorted(djcats)}")
    cnames = {c.name.lower().strip() for c in meta.contestants}
    unk = set()
    for c in clues:
        for a in c["attempts"]:
            if a["speaker"].lower().strip() not in cnames:
                unk.add(a["speaker"])
    R.check("FK: all buzzers known", len(unk) == 0, f"unknown: {unk}" if unk else "all mapped")
    fj_unk = [
        w.contestant for w in meta.final_jeopardy.wagers_and_responses if w.contestant.lower().strip() not in cnames
    ]
    R.check("FK: FJ wagerers known", len(fj_unk) == 0, f"unknown: {fj_unk}" if fj_unk else "all mapped")
    empty_txt = sum(1 for c in clues if not c["clue_text"].strip())
    R.check("Quality: no empty clue texts", empty_txt == 0, f"{empty_txt} empty")
    inverted = sum(1 for c in clues if c["host_finish_ms"] < c["host_start_ms"])
    R.check("Quality: no inverted timestamps", inverted == 0, f"{inverted} inverted")
    atts = sum(len(c["attempts"]) for c in clues)
    R.check("Quality: buzz attempts extracted", atts > 0, f"{atts} total")


def validate_state_machine(meta, clues, R):
    print("\n── State Machine ────────────────────────────────────")
    scores = {}
    for c in clues:
        if c["round"] == "Jeopardy":
            val = c["board_row"] * 200
        elif c["round"] == "Double Jeopardy":
            val = c["board_row"] * 400
        else:
            val = 0
        if c["is_daily_double"] and c["daily_double_wager"] and c["wagerer_name"]:
            w = c["wagerer_name"]
            scores.setdefault(w, 0)
            if c["daily_double_wager"] == "True Daily Double":
                mx = 1000 if c["round"] == "Jeopardy" else 2000
                wamt = max(scores[w], mx)
            else:
                try:
                    wamt = int(c["daily_double_wager"])
                except (ValueError, TypeError):
                    wamt = val
            if c["attempts"] and c["attempts"][0]["is_correct"]:
                scores[w] += wamt
            elif c["attempts"]:
                scores[w] -= wamt
        else:
            for a in c["attempts"]:
                p = a["speaker"]
                scores.setdefault(p, 0)
                if a["is_correct"]:
                    scores[p] += val
                    break
                else:
                    scores[p] -= val
    for name, score in scores.items():
        R.check(f"Score: {name}", -10000 <= score <= 50000, f"${score:,}")
    R.check("Scores: at least one positive", any(s > 0 for s in scores.values()))
    return scores


# ── Episode JSON Validation ────────────────────────────────────────


def validate_episode_json(path, R):
    """Validate a pre-extracted episode JSON file without API calls."""
    print(f"\n── Validating Episode JSON: {os.path.basename(path)} ──")
    with open(path) as f:
        data = json.load(f)
    R.check("JSON: has contestants", len(data.get("contestants", [])) > 0, f"{len(data.get('contestants', []))}")
    R.check("JSON: has clues", len(data.get("clues", [])) > 0, f"{len(data.get('clues', []))}")
    R.check("JSON: has host", bool(data.get("host_name")), data.get("host_name", ""))
    R.check(
        "JSON: has FJ",
        bool(data.get("final_jeopardy", {}).get("category")),
        data.get("final_jeopardy", {}).get("category", ""),
    )
    clues = data.get("clues", [])
    j = [c for c in clues if c.get("round") == "Jeopardy"]
    dj = [c for c in clues if c.get("round") == "Double Jeopardy"]
    R.check("JSON: J clues 15-30", 15 <= len(j) <= 30, f"{len(j)}")
    R.check("JSON: DJ clues 15-30", 15 <= len(dj) <= 30, f"{len(dj)}")
    dd = sum(1 for c in clues if c.get("is_daily_double"))
    R.check("JSON: 1-3 DDs", 1 <= dd <= 3, f"{dd}")
    cnames = {c["name"].lower().strip() for c in data.get("contestants", [])}
    unk = set()
    for c in clues:
        for a in c.get("attempts", []):
            if a.get("speaker", "").lower().strip() not in cnames:
                unk.add(a.get("speaker", "?"))
    R.check("JSON FK: buzzers known", len(unk) == 0, f"unknown: {unk}" if unk else "all mapped")
    return data


# ── Main ───────────────────────────────────────────────────────────


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
        gzs = glob.glob(os.path.join(args.dir, "gpu_output_*.json.gz"))
        mp3s = glob.glob(os.path.join(args.dir, "*_interview_slice.mp3"))
        epjs = glob.glob(os.path.join(args.dir, "episode_*.json"))
        if gzs and not args.transcript:
            args.transcript = gzs[0]
        if mp3s and not args.audio:
            args.audio = mp3s[0]
        if epjs and not args.episode_json and not args.transcript:
            args.episode_json = epjs[0]
        print(f"  Auto-detected: transcript={args.transcript}, audio={args.audio}, episode={args.episode_json}")

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

    # Run LLM pipeline
    async def _run():
        client = get_client()
        mapping = {}
        if not args.skip_pass1 and args.audio and os.path.exists(args.audio):
            mapping = await run_pass1(client, args.audio, R)
        elif args.skip_pass1:
            R.warn("Pass1 skipped by flag")
        else:
            R.warn("Pass1 skipped — no audio slice found")
        meta, clues = await run_pass2(client, segs, mapping, R)
        return meta, clues

    meta, clues = asyncio.run(_run())
    validate_extraction(meta, clues, R)
    scores = validate_state_machine(meta, clues, R)

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
