"""
Validation checks, state machine, and diagnostic display functions.

All analysis and display logic — no LLM API calls happen here.
"""

import json
import os


def validate_extraction(meta, clues, R):
    print("\n── Extraction Quality ───────────────────────────────")
    j = [c for c in clues if c["round"] == "Jeopardy"]
    dj = [c for c in clues if c["round"] == "Double Jeopardy"]

    # ── Round-level clue counts ──────────────────────────────────────
    R.check("Clues: J round 15-30", 15 <= len(j) <= 30, f"{len(j)}")
    R.check("Clues: DJ round 15-30", 15 <= len(dj) <= 30, f"{len(dj)}")
    if 0 < len(j) < 15:
        R.warn(f"J round severely under-extracted: {len(j)} clues (expected 25-30)")
    if 0 < len(dj) < 15:
        R.warn(f"DJ round severely under-extracted: {len(dj)} clues (expected 25-30)")

    # ── Daily Double constraints ─────────────────────────────────────
    dd = sum(1 for c in clues if c["is_daily_double"])
    dd_j = sum(1 for c in j if c["is_daily_double"])
    dd_dj = sum(1 for c in dj if c["is_daily_double"])
    R.check("Clues: 1-3 Daily Doubles", 1 <= dd <= 3, f"{dd}")
    if dd_j > 1:
        R.warn(f"Found {dd_j} DDs in Jeopardy round (expected 1)")
    if dd_dj > 2:
        R.warn(f"Found {dd_dj} DDs in Double Jeopardy (expected max 2)")

    # DD structural: wager + wagerer present, exactly 1 attempt
    dd_clues = [c for c in clues if c["is_daily_double"]]
    dd_wagers = sum(1 for c in dd_clues if c.get("daily_double_wager") and c.get("wagerer_name"))
    R.check("Clues: All DDs have wagers", dd == dd_wagers, f"{dd_wagers}/{dd}")
    for c in dd_clues:
        if len(c["attempts"]) > 1:
            R.warn(f"DD in '{c['category']}' has {len(c['attempts'])} attempts (expected 1)")
        if c.get("daily_double_wager"):
            try:
                wval = int(c["daily_double_wager"])
                if wval <= 0:
                    R.warn(f"DD wager ≤ 0: ${wval} in '{c['category']}'")
                elif wval > 50000:
                    R.warn(f"DD wager suspiciously high: ${wval:,} in '{c['category']}'")
            except (ValueError, TypeError):
                pass  # "True Daily Double" is valid

    # ── Category count per round ─────────────────────────────────────
    jcats = {c["category"].lower().strip() for c in j}
    djcats = {c["category"].lower().strip() for c in dj}
    R.check("Clues: J has 5-6 categories", 5 <= len(jcats) <= 6, f"{len(jcats)}: {sorted(jcats)}")
    R.check("Clues: DJ has 5-6 categories", 5 <= len(djcats) <= 6, f"{len(djcats)}: {sorted(djcats)}")
    if len(jcats) > 6:
        R.warn(f"J has {len(jcats)} categories (>6 suggests variant spellings)")
    if len(djcats) > 6:
        R.warn(f"DJ has {len(djcats)} categories (>6 suggests variant spellings)")

    # ── Board position bounds ────────────────────────────────────────
    oob_rows = sum(1 for c in clues if c["round"] in ("Jeopardy", "Double Jeopardy") and not (1 <= c["board_row"] <= 5))
    oob_cols = sum(1 for c in clues if c["round"] in ("Jeopardy", "Double Jeopardy") and not (1 <= c["board_col"] <= 6))
    R.check("Board: row bounds [1-5]", oob_rows == 0, f"{oob_rows} out-of-bounds")
    R.check("Board: col bounds [1-6]", oob_cols == 0, f"{oob_cols} out-of-bounds")

    # ── Duplicate board positions per category+round ─────────────────
    dup_positions = 0
    for rnd_name, rnd_clues in [("Jeopardy", j), ("Double Jeopardy", dj)]:
        seen = {}
        for c in rnd_clues:
            key = (c["category"].lower().strip(), c["board_row"])
            if key in seen:
                dup_positions += 1
            seen[key] = True
    R.check("Board: no duplicate positions", dup_positions == 0, f"{dup_positions} duplicates")

    # ── Contestant FK consistency ────────────────────────────────────
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

    # ── Per-clue data quality ────────────────────────────────────────
    empty_txt = sum(1 for c in clues if not c["clue_text"].strip())
    empty_resp = sum(1 for c in clues if not c.get("correct_response", "").strip())
    R.check("Quality: no empty clue texts", empty_txt == 0, f"{empty_txt} empty")
    R.check("Quality: no empty responses", empty_resp == 0, f"{empty_resp} empty")

    inverted = sum(1 for c in clues if c["host_finish_ms"] < c["host_start_ms"])
    R.check("Quality: no inverted timestamps", inverted == 0, f"{inverted} inverted")

    zero_ts = sum(1 for c in clues if c.get("selection_order", 0) > 3 and c["host_start_ms"] == 0.0)
    if zero_ts > 0:
        R.warn(f"Quality: {zero_ts} mid-game clue(s) have host_start=0 (Line ID failure)")

    long_read = sum(1 for c in clues if c["host_finish_ms"] - c["host_start_ms"] > 60000)
    if long_read > 0:
        R.warn(f"Quality: {long_read} clue(s) have read duration >60s (bad Line IDs)")

    buzz_before_start = sum(
        1 for c in clues for a in c.get("attempts", []) if a.get("buzz_timestamp_ms", float("inf")) < c["host_start_ms"]
    )
    if buzz_before_start > 0:
        R.warn(f"Quality: {buzz_before_start} buzz(es) before host_start (hallucinated buzz Line ID)")

    # ── Buzz attempt stats ───────────────────────────────────────────
    atts = sum(len(c["attempts"]) for c in clues)
    R.check("Quality: buzz attempts extracted", atts > 0, f"{atts} total")

    triple_stumpers = sum(
        1
        for c in clues
        if c["round"] in ("Jeopardy", "Double Jeopardy")
        and (not c["attempts"] or all(not a["is_correct"] for a in c["attempts"]))
    )
    rebounds = sum(1 for c in clues if len(c["attempts"]) > 1)
    print(f"  📊 Triple stumpers: {triple_stumpers} | Rebounds (multi-buzz): {rebounds}")

    # ── Clue text length distribution ────────────────────────────────
    word_counts = [len(c["clue_text"].split()) for c in clues if c["clue_text"].strip()]
    if word_counts:
        word_counts.sort()
        median = word_counts[len(word_counts) // 2]
        print(
            f"  📊 Clue text words: min={word_counts[0]} median={median} "
            f"max={word_counts[-1]} avg={sum(word_counts) / len(word_counts):.1f}"
        )
        short_clues = sum(1 for w in word_counts if w < 5)
        if short_clues > 0:
            R.warn(f"Quality: {short_clues} clue(s) have <5 words (possibly truncated)")

    # ── Timestamp coverage ───────────────────────────────────────────
    if clues:
        first_clue_ms = min(c["host_start_ms"] for c in clues if c["host_start_ms"] > 0)
        last_clue_ms = max(c["host_finish_ms"] for c in clues)
        coverage_min = (last_clue_ms - first_clue_ms) / 60000
        print(f"  📊 Timestamp span: {first_clue_ms / 1000:.0f}s – {last_clue_ms / 1000:.0f}s ({coverage_min:.1f} min)")

    # ── FJ validation ────────────────────────────────────────────────
    fj = meta.final_jeopardy
    R.check("FJ: category extracted", bool(fj.category.strip()), fj.category)
    R.check("FJ: clue text extracted", bool(fj.clue_text.strip()), f"{len(fj.clue_text)} chars")
    R.check("FJ: 1-3 wagers", 1 <= len(fj.wagers_and_responses) <= 3, f"{len(fj.wagers_and_responses)} wagers")


def validate_state_machine(meta, clues, R):
    print("\n── State Machine ────────────────────────────────────")
    scores = {}
    coryat = {}  # Coryat: clue value only, no DD wagers, no FJ
    board_control_trace = []
    current_control = None
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
            coryat.setdefault(w, 0)
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
                coryat[w] += val  # Coryat uses clue value, not wager
                if current_control != w:
                    board_control_trace.append(w)
                    current_control = w
            elif c["attempts"]:
                scores[w] -= wamt
                coryat[w] -= val
        else:
            for a in c["attempts"]:
                p = a["speaker"]
                scores.setdefault(p, 0)
                coryat.setdefault(p, 0)
                if a["is_correct"]:
                    scores[p] += val
                    coryat[p] += val
                    if current_control != p:
                        board_control_trace.append(p)
                        current_control = p
                    break
                else:
                    scores[p] -= val
                    coryat[p] -= val
    for w in meta.final_jeopardy.wagers_and_responses:
        p = w.contestant
        scores.setdefault(p, 0)
        if w.is_correct:
            scores[p] += w.wager
        else:
            scores[p] -= w.wager

    for name, score in scores.items():
        R.check(f"Score: {name}", -10000 <= score <= 50000, f"${score:,}")
    R.check("Scores: at least one positive", any(s > 0 for s in scores.values()))

    # Coryat scores
    print("  Coryat Scores (no DD wagers, no FJ):")
    for name in sorted(coryat, key=coryat.get, reverse=True):
        print(f"    {name}: ${coryat[name]:,}")

    # Board control trace (compact)
    if board_control_trace:
        # Abbreviate names to first name for compact display
        short_names = [n.split()[0] if " " in n else n for n in board_control_trace]
        trace_str = " → ".join(short_names[:20])
        if len(short_names) > 20:
            trace_str += f" … (+{len(short_names) - 20} more)"
        print(f"  Board Control: {trace_str}")
        # Count controls per contestant
        ctrl_counts = {}
        for n in board_control_trace:
            ctrl_counts[n] = ctrl_counts.get(n, 0) + 1
        for n in sorted(ctrl_counts, key=ctrl_counts.get, reverse=True):
            print(f"    {n}: {ctrl_counts[n]} selections")

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


# ── Transcript Intelligence ────────────────────────────────────────


def print_transcript_intelligence(segs, R):
    """Extract free diagnostic info from raw WhisperX segments before any LLM calls."""
    print("\n── Transcript Intelligence ───────────────────────────")

    # Speaker distribution
    speaker_stats = {}
    for s in segs:
        spk = s.get("speaker", "?")
        text = s.get("text", "")
        dur = (s.get("end", 0) or 0) - (s.get("start", 0) or 0)
        if spk not in speaker_stats:
            speaker_stats[spk] = {"segments": 0, "words": 0, "duration": 0.0}
        speaker_stats[spk]["segments"] += 1
        speaker_stats[spk]["words"] += len(text.split())
        speaker_stats[spk]["duration"] += max(0, dur)

    print("  Speaker Distribution:")
    print(f"    {'Speaker':<14} {'Segs':>5} {'Words':>6} {'Duration':>8} {'Avg Seg':>8}")
    print(f"    {'─' * 14} {'─' * 5} {'─' * 6} {'─' * 8} {'─' * 8}")
    for spk in sorted(speaker_stats, key=lambda s: speaker_stats[s]["segments"], reverse=True):
        st = speaker_stats[spk]
        avg = st["duration"] / st["segments"] if st["segments"] else 0
        print(f"    {spk:<14} {st['segments']:>5} {st['words']:>6} {st['duration']:>7.1f}s {avg:>7.2f}s")

    # Host detection heuristic: most words = host
    host_spk = max(speaker_stats, key=lambda s: speaker_stats[s]["words"])
    print(f"  📋 Likely host (most words): {host_spk} ({speaker_stats[host_spk]['words']} words)")

    # Round boundary detection
    markers_found = []
    for i, s in enumerate(segs):
        text = s.get("text", "").lower()
        ts = s.get("start", 0) or 0
        if "double jeopardy" in text:
            markers_found.append(("Double Jeopardy", i, ts))
        elif "final jeopardy" in text:
            markers_found.append(("Final Jeopardy", i, ts))
        elif "tiebreaker" in text:
            markers_found.append(("Tiebreaker", i, ts))

    if markers_found:
        print("  Round Boundaries Detected:")
        for name, idx, ts in markers_found:
            print(f'    L{idx} @ {ts / 60:.1f}m — "{name}"')
    else:
        R.warn("No round boundary markers found in transcript")

    # Silence gap analysis (gaps > 3s = likely commercial breaks)
    gaps = []
    for i in range(1, len(segs)):
        prev_end = segs[i - 1].get("end", 0) or 0
        curr_start = segs[i].get("start", 0) or 0
        gap = curr_start - prev_end
        if gap > 3.0:
            gaps.append((i, gap, curr_start))

    if gaps:
        print(f"  Silence Gaps (>3s): {len(gaps)} detected")
        for idx, gap, ts in gaps[:5]:
            print(f"    L{idx} @ {ts / 60:.1f}m — {gap:.1f}s gap")
        if len(gaps) > 5:
            print(f"    ... and {len(gaps) - 5} more")
    else:
        print("  Silence Gaps: none >3s")

    # Word density
    word_counts = [len(s.get("text", "").split()) for s in segs]
    if word_counts:
        avg_wc = sum(word_counts) / len(word_counts)
        empty_segs = sum(1 for w in word_counts if w == 0)
        short_segs = sum(1 for w in word_counts if 0 < w <= 2)
        print(f"  Segment Word Density: avg={avg_wc:.1f} | empty={empty_segs} | short(≤2)={short_segs}")

    R.check(
        "Transcript: host speaker identified",
        speaker_stats[host_spk]["words"] > 500,
        f"{host_spk} has {speaker_stats[host_spk]['words']} words",
    )
    R.check("Transcript: round markers found", len(markers_found) >= 2, f"{len(markers_found)} markers")


def print_chunk_diagnostics(chunks, segs):
    """Print per-chunk stats before LLM extraction."""
    print(f"\n  Chunk Diagnostics ({len(chunks)} chunks):")
    print(f"    {'Chunk':>5} {'Lines':>6} {'Chars':>7} {'~Tokens':>7} {'First Line':<30} {'Last Line':<30}")
    print(f"    {'─' * 5} {'─' * 6} {'─' * 7} {'─' * 7} {'─' * 30} {'─' * 30}")
    for i, chunk in enumerate(chunks):
        lines = chunk.split("\n")
        chars = len(chunk)
        est_tokens = chars // 4
        first = lines[0][:29] if lines else "?"
        last = lines[-1][:29] if lines else "?"
        print(f"    {i + 1:>5} {len(lines):>6} {chars:>7} {est_tokens:>7} {first:<30} {last:<30}")


def print_round_summary_table(clues, meta):
    """Print per-round summary as a compact ASCII table."""
    print("\n── Per-Round Summary ─────────────────────────────────")
    rounds = {"Jeopardy": [], "Double Jeopardy": [], "Final Jeopardy": []}
    for c in clues:
        rounds.setdefault(c["round"], []).append(c)

    fj = meta.final_jeopardy
    header = f"  {'Round':<18}│{'Clues':>6} │{'Cats':>5} │{'DDs':>4} │{'Buzzes':>7} │{'Stumps':>7} │{'Rebounds':>8} │{'Avg Wds':>8}"
    sep = f"  {'─' * 18}┼{'─' * 6}─┼{'─' * 5}─┼{'─' * 4}─┼{'─' * 7}─┼{'─' * 7}─┼{'─' * 8}─┼{'─' * 8}"
    print(header)
    print(sep)

    total_clues = total_buzzes = total_stumps = total_rebounds = 0
    total_words = 0
    total_cats = 0

    for rnd_name in ["Jeopardy", "Double Jeopardy"]:
        rnd = rounds.get(rnd_name, [])
        cats = len({c["category"].lower().strip() for c in rnd})
        dds = sum(1 for c in rnd if c["is_daily_double"])
        buzzes = sum(len(c["attempts"]) for c in rnd)
        stumps = sum(1 for c in rnd if not c["attempts"] or all(not a["is_correct"] for a in c["attempts"]))
        rebounds = sum(1 for c in rnd if len(c["attempts"]) > 1)
        words = [len(c["clue_text"].split()) for c in rnd]
        avg_words = sum(words) / len(words) if words else 0
        total_clues += len(rnd)
        total_buzzes += buzzes
        total_stumps += stumps
        total_rebounds += rebounds
        total_words += sum(words)
        total_cats += cats
        print(
            f"  {rnd_name:<18}│{len(rnd):>6} │{cats:>5} │{dds:>4} │{buzzes:>7} │{stumps:>7} │{rebounds:>8} │{avg_words:>7.1f}"
        )

    # FJ row
    fj_buzzes = len(fj.wagers_and_responses)
    fj_words = len(fj.clue_text.split()) if fj.clue_text else 0
    fj_stumps = 1 if fj_buzzes == 0 or all(not w.is_correct for w in fj.wagers_and_responses) else 0
    total_clues += 1
    total_buzzes += fj_buzzes
    total_stumps += fj_stumps
    total_words += fj_words
    total_cats += 1
    print(
        f"  {'Final Jeopardy':<18}│{'—':>6} │{'1':>5} │{'—':>4} │{fj_buzzes:>7} │{fj_stumps:>7} │{'—':>8} │{fj_words:>7.1f}"
    )
    print(sep)
    avg_total = total_words / total_clues if total_clues else 0
    print(
        f"  {'TOTAL':<18}│{total_clues:>6} │{total_cats:>5} │{'':>4} │{total_buzzes:>7} │{total_stumps:>7} │{total_rebounds:>8} │{avg_total:>7.1f}"
    )


def print_board_coverage(clues, R):
    """Print 5×6 board occupancy matrix per round."""
    for rnd_name, mult in [("Jeopardy", 200), ("Double Jeopardy", 400)]:
        rnd = [c for c in clues if c["round"] == rnd_name]
        if not rnd:
            continue
        cats = sorted({c["category"].lower().strip() for c in rnd})
        cat_short = [c[:10] for c in cats]

        # Build occupancy grid
        grid = {}  # (row, cat) -> status
        for c in rnd:
            cat_key = c["category"].lower().strip()
            row = c["board_row"]
            status = "DD" if c["is_daily_double"] else "✓"
            grid[(row, cat_key)] = status

        print(f"\n  {rnd_name} Board Coverage ({len(rnd)}/30 clues):")
        # Header
        hdr = "        " + "".join(f"{s:<12}" for s in cat_short[:6])
        print(hdr)
        for row in range(1, 6):
            val = f"${row * mult}"
            cells = ""
            for cat in cats[:6]:
                cell = grid.get((row, cat), "·")
                cells += f"{cell:<12}"
            print(f"    {val:<4} {cells}")

        covered = len(grid)
        expected = min(len(cats), 6) * 5
        R.check(
            f"Board: {rnd_name} coverage",
            covered >= expected * 0.8,
            f"{covered}/{expected} cells filled ({covered / expected * 100:.0f}%)" if expected else "no categories",
        )


def print_contestant_matrix(scores, clues, meta):
    """Print per-contestant stats matrix."""
    print("\n── Contestant Performance ────────────────────────────")
    cnames = {c.name for c in meta.contestants}
    stats = {}
    for name in cnames:
        stats[name] = {"buzzes": 0, "correct": 0, "value_won": 0, "value_lost": 0, "rebounds": 0}

    for c in clues:
        for a in c["attempts"]:
            spk = a["speaker"]
            if spk not in stats:
                continue
            stats[spk]["buzzes"] += 1
            if a["is_correct"]:
                stats[spk]["correct"] += 1

    # Calculate pre-FJ scores (Coryat-like)
    pre_fj = {}
    for name in cnames:
        pre_fj[name] = 0
    for c in clues:
        if c["round"] == "Jeopardy":
            val = c["board_row"] * 200
        elif c["round"] == "Double Jeopardy":
            val = c["board_row"] * 400
        else:
            continue
        if c["is_daily_double"] and c.get("daily_double_wager") and c.get("wagerer_name"):
            w = c["wagerer_name"]
            if w in pre_fj:
                try:
                    wamt = int(c["daily_double_wager"])
                except (ValueError, TypeError):
                    wamt = val
                if c["attempts"] and c["attempts"][0]["is_correct"]:
                    pre_fj[w] += wamt
                elif c["attempts"]:
                    pre_fj[w] -= wamt
        else:
            for a in c["attempts"]:
                p = a["speaker"]
                if p not in pre_fj:
                    continue
                if a["is_correct"]:
                    pre_fj[p] += val
                    break
                else:
                    pre_fj[p] -= val

    fj = meta.final_jeopardy
    fj_wagers = {w.contestant: w.wager for w in fj.wagers_and_responses}

    header = f"  {'Name':<20}│{'Buzz':>5} │{'Corr':>5} │{'Acc%':>6} │{'Pre-FJ':>9} │{'FJ Wager':>9} │{'Final':>9}"
    sep_line = f"  {'─' * 20}┼{'─' * 5}─┼{'─' * 5}─┼{'─' * 6}─┼{'─' * 9}─┼{'─' * 9}─┼{'─' * 9}"
    print(header)
    print(sep_line)

    for name in sorted(cnames, key=lambda n: scores.get(n, 0), reverse=True):
        s = stats.get(name, {"buzzes": 0, "correct": 0})
        acc = (s["correct"] / s["buzzes"] * 100) if s["buzzes"] else 0
        fj_w = fj_wagers.get(name, "—")
        fj_w_str = f"${fj_w:,}" if isinstance(fj_w, int) else str(fj_w)
        final = scores.get(name, 0)
        pf = pre_fj.get(name, 0)
        print(
            f"  {name:<20}│{s['buzzes']:>5} │{s['correct']:>5} │{acc:>5.1f}%│{f'${pf:,}':>9} │{fj_w_str:>9} │{f'${final:,}':>9}"
        )


def print_api_summary(api_calls):
    """Print consolidated API cost/token table."""
    if not api_calls:
        return
    print("\n── API Summary ──────────────────────────────────────")
    header = f"  {'Call':<22}│{'Model':<16}│{'In':>7} │{'Out':>7} │{'Think':>6} │{'Latency':>8} │{'Cost':>8}"
    sep_line = f"  {'─' * 22}┼{'─' * 16}┼{'─' * 7}─┼{'─' * 7}─┼{'─' * 6}─┼{'─' * 8}─┼{'─' * 8}"
    print(header)
    print(sep_line)

    totals = {"input": 0, "output": 0, "thinking": 0, "ms": 0}
    for call in api_calls:
        name = call.get("name", "?")[:21]
        model = call.get("model", "?")
        model_short = model.split("-preview")[0].replace("gemini-", "") if model else "?"
        inp = call.get("input", 0)
        out = call.get("output", 0)
        think = call.get("thinking", 0)
        ms = call.get("ms", 0)
        cost = inp * 0.00000125 + out * 0.00001  # rough estimate
        totals["input"] += inp
        totals["output"] += out
        totals["thinking"] += think
        totals["ms"] += ms
        print(f"  {name:<22}│{model_short:<16}│{inp:>7,} │{out:>7,} │{think:>6,} │{ms:>7,.0f}ms│${cost:>6.4f}")

    print(sep_line)
    total_cost = totals["input"] * 0.00000125 + totals["output"] * 0.00001
    print(
        f"  {'TOTAL':<22}│{'':<16}│{totals['input']:>7,} │{totals['output']:>7,} │{totals['thinking']:>6,} │{totals['ms']:>7,.0f}ms│${total_cost:>6.4f}"
    )
