"""
GPU validation, Gemini API helpers, Pass 1 speaker anchoring, and Pass 2
structured extraction — all LLM interaction lives here.
"""

import asyncio
import json
import os
import time

from .schemas import (
    PartialClues,
    PartialMeta,
)


# ── GPU Output Validation ──────────────────────────────────────────


def validate_gpu(gpu_data, R):
    print("\n── GPU Output Structure ──────────────────────────────")
    tr = gpu_data.get("transcript", {})
    segs = tr.get("segments", [])
    R.check("GPU: transcript key exists", bool(tr))
    R.check("GPU: segments non-empty", len(segs) > 0, f"{len(segs)} segments")
    if not segs:
        return []
    actual_keys = sorted(segs[0].keys())
    expected = {"text", "start", "end", "speaker"}
    missing = expected - set(actual_keys)
    R.check(
        "GPU: has text/start/end/speaker",
        len(missing) == 0,
        f"missing: {missing}, actual keys: {actual_keys}" if missing else f"keys: {actual_keys}",
    )
    hs = sum(1 for s in segs if s.get("start") is not None)
    R.check("GPU: >95% have timestamps", hs / len(segs) > 0.95, f"{hs}/{len(segs)}")
    spk = {s.get("speaker", "?") for s in segs}
    real_speakers = {s for s in spk if s.startswith("SPEAKER_")}
    unassigned_count = sum(1 for s in segs if s.get("speaker", "?") == "?")
    unassigned_pct = unassigned_count / len(segs) * 100 if segs else 0
    has_speakers = len(real_speakers) >= 2
    R.check(
        "GPU: multiple speakers diarized",
        has_speakers,
        f"{len(real_speakers)} speakers: {sorted(real_speakers)}"
        if has_speakers
        else f"found {sorted(spk)} — diarization may be missing",
    )
    if has_speakers and unassigned_pct > 20:
        R.warn(
            "GPU: high unassigned speaker ratio",
            f"{unassigned_count}/{len(segs)} ({unassigned_pct:.0f}%) segments have no speaker",
        )
    if has_speakers and len(real_speakers) > 8:
        R.warn(
            "GPU: possible over-segmentation",
            f"{len(real_speakers)} distinct speakers detected (expected 3-5 for standard Jeopardy)",
        )
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
        "You are a strict data extractor. Analyze the audio of the Jeopardy host interview segment. "
        "Identify the host and each contestant by their distinct voice. "
        "Return ONLY a JSON object mapping diarization speaker IDs to full names. "
        'Example: {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Jane Doe", "SPEAKER_02": "John Smith"}. '
        "Do NOT return transcribed text. Do NOT return a list. Return ONLY the mapping dict."
    )
    t0 = time.perf_counter()
    uploaded = await asyncio.to_thread(client.files.upload, file=audio_path)
    prompt = [
        "Listen to this Jeopardy interview audio. Identify who each SPEAKER_XX is. "
        "Return a JSON dict mapping each SPEAKER_XX ID to their full name.",
        uploaded,
    ]
    resp, usage = await call_gemini(client, MODEL_FLASH, prompt, sys_p, max_tokens=2048, ctx="Pass1")
    await asyncio.to_thread(client.files.delete, name=uploaded.name)
    ms = (time.perf_counter() - t0) * 1000
    clean = str(resp.text).replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        import ast

        parsed = ast.literal_eval(clean)

    # Handle both dict (expected) and list (fallback) response formats
    if isinstance(parsed, dict):
        # Check if it's a proper {SPEAKER_XX: Name} mapping or something else
        if all(isinstance(v, str) for v in parsed.values()):
            mapping = parsed
        else:
            print("  ⚠️  Dict values are not strings, attempting to extract names...")
            mapping = {k: str(v) for k, v in parsed.items()}
    elif isinstance(parsed, list):
        # Gemini returned a list of segments — extract speaker→name mapping
        print("  ⚠️  Got segment list instead of mapping — inferring speaker names...")
        mapping = _infer_speaker_mapping_from_segments(parsed)
    else:
        mapping = {}

    R.check("Pass1: returned mapping", isinstance(mapping, dict) and len(mapping) > 0, f"{mapping}")
    R.check("Pass1: 2+ speakers", len(mapping) >= 2, f"{len(mapping)} speakers")
    R.check("Pass1: latency <30s", ms < 30000, f"{ms:.0f}ms")
    print(f"  📊 Tokens: in={usage.get('input', 0)} out={usage.get('output', 0)} | {ms:.0f}ms")
    return mapping


def _infer_speaker_mapping_from_segments(segments):
    """Infer SPEAKER_XX → Name mapping from a list of diarized transcript segments.

    Heuristic: The host is the speaker with the most segments (they read clues,
    call on contestants, and narrate). Contestant names are mentioned by the host.
    """
    import re as _re

    # Count segments per speaker
    speaker_counts = {}
    speaker_texts = {}
    for seg in segments:
        spk = seg.get("speaker", "")
        if not spk:
            continue
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        speaker_texts.setdefault(spk, []).append(seg.get("text", ""))

    if not speaker_counts:
        return {}

    # Host is the most frequent speaker (reads clues, narrates)
    host_speaker = max(speaker_counts, key=speaker_counts.get)

    # Known Jeopardy hosts
    known_hosts = ["Ken Jennings", "Ryan Seacrest", "Mayim Bialik", "Alex Trebek"]

    # Extract contestant names mentioned by the host
    host_text = " ".join(speaker_texts.get(host_speaker, []))
    # Look for patterns like "Scott?", "Elise, it's your turn", "Dan, what is"
    name_pattern = _re.findall(r"\b([A-Z][a-z]{2,})\b", host_text)

    # Filter to likely contestant names (mentioned multiple times, not common words)
    common_words = {
        "The",
        "And",
        "But",
        "For",
        "Not",
        "You",
        "Your",
        "Yes",
        "That",
        "This",
        "What",
        "Who",
        "Where",
        "When",
        "How",
        "Which",
        "Here",
        "There",
        "Back",
        "Right",
        "Correct",
        "Answer",
        "Daily",
        "Double",
        "Jeopardy",
        "Final",
        "Category",
        "Categories",
        "Clue",
        "Score",
        "Place",
        "First",
        "Second",
        "Third",
        "Let",
        "Take",
        "Look",
        "Start",
        "Pick",
        "Select",
        "Got",
        "Put",
        "Going",
        "Come",
        "Most",
        "Last",
        "Next",
        "Okay",
        "Well",
        "Now",
        "Get",
        "See",
        "Say",
        "Try",
        "Make",
        "Give",
        "Just",
        "Also",
        "Still",
        "Much",
        "Many",
        "Some",
        "More",
        "Very",
        "Than",
        "Only",
    }
    name_freq = {}
    for name in name_pattern:
        if name not in common_words and len(name) > 2:
            name_freq[name] = name_freq.get(name, 0) + 1

    # Top 3 most-mentioned names are likely contestants
    contestant_names = sorted(name_freq, key=name_freq.get, reverse=True)[:3]

    # Build mapping
    mapping = {}

    # Detect host name
    host_name = "Host"
    for known in known_hosts:
        if known.split()[0] in host_text or known.split()[-1] in host_text:
            host_name = known
            break
    mapping[host_speaker] = host_name

    # Try to match other speakers to contestant names by what the host says about them
    non_host_speakers = [
        s for s in speaker_counts if s != host_speaker and not s.startswith("SPEAKER_0") or s != host_speaker
    ]
    non_host_speakers = sorted(
        [s for s in speaker_counts if s != host_speaker],
        key=lambda s: int(s.replace("SPEAKER_", "")) if s.startswith("SPEAKER_") else 999,
    )

    # Assign names to non-host speakers
    used_names = set()
    for spk in non_host_speakers:
        # Check if any contestant name appears right after this speaker's text in host response
        best_name = None
        for name in contestant_names:
            if name not in used_names:
                best_name = name
                break
        if best_name:
            mapping[spk] = best_name
            used_names.add(best_name)
        else:
            mapping[spk] = f"Contestant ({spk})"

    print(f"  📋 Inferred mapping: {mapping}")
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
    try:
        from .diagnostics import print_chunk_diagnostics
    except ImportError:
        from validate_llm_pipeline.diagnostics import print_chunk_diagnostics

    print("\n── Pass 2: Structured Extraction ─────────────────────")
    comp_map = {abbrev(k): v for k, v in speaker_mapping.items()}
    transcript = fmt_transcript(segs)
    base_sys = (
        "You are Trebek, an expert Jeopardy data extraction pipeline. "
        f"CRITICAL: Map speakers using: {json.dumps(comp_map)}. "
        "Speaker IDs are abbreviated (S0=SPEAKER_00). "
        "Do NOT hallucinate names. Use Line IDs (e.g. L0, L105) for timestamps."
    )
    api_calls = []  # Track all API calls for summary table

    # Meta extraction
    print("  Extracting episode metadata...")
    meta_prompt = f"Transcript:\n{transcript}\n\nExtract episode_date, host_name, is_tournament, contestants, final_jeopardy, score_adjustments. Do NOT extract clues."

    t0 = time.perf_counter()
    resp, mu = await call_gemini(client, MODEL_PRO, meta_prompt, base_sys, PartialMeta, ctx="Meta")
    meta_txt = str(resp.text).replace("```json", "").replace("```", "").strip()
    meta = PartialMeta.model_validate_json(meta_txt)
    meta_ms = (time.perf_counter() - t0) * 1000
    api_calls.append({"name": "Pass 2 Meta", "model": MODEL_PRO, **mu, "ms": meta_ms})
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

    # Print chunk diagnostics before extraction
    print_chunk_diagnostics(chunks, segs)
    print(f"\n  Extracting clues from {len(chunks)} chunks...")

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
        api_calls.append({"name": f"Pass 2 Chunk {ci + 1}/{len(chunks)}", "model": MODEL_PRO, **cu})
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

    # ── Speaker Normalization ──────────────────────────────────────
    # The LLM often outputs abbreviated speaker IDs (S01, S02) or first names
    # instead of full contestant names. Normalize them using the speaker mapping
    # and contestant list from metadata.
    contestant_names = [c.name for c in meta.contestants]
    # Build a lookup: abbreviated ID → full name, first name → full name
    norm_map = {}
    # From speaker mapping: SPEAKER_01 → "Scott" → find "Scott Riccardi"
    for spk_id, short_name in speaker_mapping.items():
        abbr = abbrev(spk_id)  # e.g. "S01"
        # Try to match short_name to a full contestant name
        matched = _match_to_contestant(short_name, contestant_names)
        if matched:
            norm_map[abbr] = matched
            norm_map[short_name] = matched
            norm_map[short_name.lower()] = matched
        else:
            norm_map[abbr] = short_name  # Keep the short name at minimum

    # Also map zero-padded variants (S1 vs S01)
    for k, v in list(norm_map.items()):
        if k.startswith("S") and k[1:].isdigit():
            norm_map[f"S{int(k[1:]):02d}"] = v  # S1 → S01 mapping
            norm_map[f"S{int(k[1:])}"] = v  # S01 → S1 mapping

    normalized_count = 0
    for c in deduped:
        for a in c["attempts"]:
            original = a["speaker"]
            if original in norm_map:
                a["speaker"] = norm_map[original]
                if a["speaker"] != original:
                    normalized_count += 1
            elif original.lower() in norm_map:
                a["speaker"] = norm_map[original.lower()]
                if a["speaker"] != original:
                    normalized_count += 1
            else:
                # Try substring matching as last resort
                matched = _match_to_contestant(original, contestant_names)
                if matched:
                    a["speaker"] = matched
                    normalized_count += 1
        # Also normalize wagerer_name for Daily Doubles
        if c.get("wagerer_name"):
            wn = c["wagerer_name"]
            if wn in norm_map:
                c["wagerer_name"] = norm_map[wn]
            elif wn.lower() in norm_map:
                c["wagerer_name"] = norm_map[wn.lower()]
            else:
                matched = _match_to_contestant(wn, contestant_names)
                if matched:
                    c["wagerer_name"] = matched

    if normalized_count:
        print(f"  📋 Normalized {normalized_count} speaker references → full contestant names")

    return meta, deduped, api_calls


def _match_to_contestant(name, contestant_names):
    """Match a partial name to a full contestant name via substring/case-insensitive matching."""
    if not name:
        return None
    name_lower = name.lower().strip()
    # Exact match
    for cn in contestant_names:
        if cn.lower() == name_lower:
            return cn
    # First name match
    for cn in contestant_names:
        if cn.lower().startswith(name_lower) or name_lower in cn.lower():
            return cn
    # Last name match
    for cn in contestant_names:
        parts = cn.lower().split()
        if any(name_lower == p for p in parts):
            return cn
    return None
