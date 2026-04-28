"""
Deterministic integrity validation and deduplication for extracted J! data.

Encodes hard domain rules (board dimensions, Daily Double limits, contestant FK
consistency, wager bounds, board position uniqueness, question format) as
programmatic checks. These run after every extraction to catch LLM
hallucinations and data corruption before the relational DB commit.
"""

import structlog

from trebek.schemas import Episode, Clue

logger = structlog.get_logger()


def _validate_extraction_integrity(episode: Episode) -> list[str]:
    """
    Runs all deterministic integrity checks against an assembled Episode.

    Returns a list of human-readable warning strings. An empty list means
    the episode passed all checks.
    """
    warnings: list[str] = []

    # ── Fatal structural checks ──────────────────────────────────────

    if not episode.contestants:
        warnings.append("No contestants extracted — all FK references will fail")

    if not episode.clues:
        warnings.append("Zero clues extracted — episode is empty")
        return warnings  # No point running clue-level checks

    # ── Round-level checks ───────────────────────────────────────────

    j_clues = [c for c in episode.clues if c.round == "J!"]
    dj_clues = [c for c in episode.clues if c.round == "Double J!"]

    if len(j_clues) > 30:
        warnings.append(f"J! round has {len(j_clues)} clues (max 30)")
    if len(dj_clues) > 30:
        warnings.append(f"Double J! round has {len(dj_clues)} clues (max 30)")
    if len(j_clues) == 0:
        warnings.append("No J! round clues extracted")
    elif len(j_clues) < 15:
        warnings.append(f"J! round severely under-extracted: {len(j_clues)} clues (expected 25-30)")
    if len(dj_clues) == 0:
        warnings.append("No Double J! round clues extracted")
    elif len(dj_clues) < 15:
        warnings.append(f"Double J! round severely under-extracted: {len(dj_clues)} clues (expected 25-30)")

    # ── Timestamp ordering within rounds ─────────────────────────────

    for round_name, round_clues in [("J!", j_clues), ("Double J!", dj_clues)]:
        sorted_round = sorted(round_clues, key=lambda c: c.host_start_timestamp_ms)
        for i in range(1, len(sorted_round)):
            if sorted_round[i].host_start_timestamp_ms < sorted_round[i - 1].host_finish_timestamp_ms:
                warnings.append(
                    f"{round_name} clue {i + 1} overlaps with clue {i}: "
                    f"start {sorted_round[i].host_start_timestamp_ms} < "
                    f"prev finish {sorted_round[i - 1].host_finish_timestamp_ms}"
                )
                if len([w for w in warnings if "overlaps with clue" in w]) >= 5:
                    break  # Cap overlap warnings at 5 to avoid noise

    # ── Daily Double constraints ─────────────────────────────────────

    dd_j = sum(1 for c in j_clues if c.is_daily_double)
    dd_dj = sum(1 for c in dj_clues if c.is_daily_double)
    total_dd = dd_j + dd_dj

    if total_dd > 3:
        warnings.append(f"Found {total_dd} Daily Doubles (expected max 3)")
    if dd_j > 1:
        warnings.append(f"Found {dd_j} Daily Doubles in J! round (expected 1)")
    if dd_dj > 2:
        warnings.append(f"Found {dd_dj} Daily Doubles in Double J! (expected 2)")

    # ── Daily Double structural constraints ──────────────────────────

    for clue in episode.clues:
        if clue.is_daily_double:
            # DD clues must have exactly 1 attempt (only the wagerer responds)
            if len(clue.attempts) > 1:
                warnings.append(
                    f"Daily Double in '{clue.category}' has {len(clue.attempts)} "
                    f"attempts (expected 1 — only wagerer responds)"
                )
            # DD wager bounds validation
            if clue.daily_double_wager is not None and isinstance(clue.daily_double_wager, int):
                if clue.daily_double_wager <= 0:
                    warnings.append(f"Daily Double wager <= 0: ${clue.daily_double_wager} in '{clue.category}'")
                elif clue.daily_double_wager > 50000:
                    warnings.append(
                        f"Daily Double wager suspiciously high: ${clue.daily_double_wager} in '{clue.category}'"
                    )

    # ── Contestant FK consistency ────────────────────────────────────

    contestant_names = {c.name.lower().strip() for c in episode.contestants}

    # Check buzz attempt speakers
    unknown_buzzers: set[str] = set()
    for clue in episode.clues:
        for attempt in clue.attempts:
            if attempt.speaker.lower().strip() not in contestant_names:
                unknown_buzzers.add(attempt.speaker)
    if unknown_buzzers:
        warnings.append(f"Unknown contestants in buzz attempts: {unknown_buzzers}")

    # Check FJ wager contestant names
    for wager in episode.final_jep.wagers_and_responses:
        if wager.contestant.lower().strip() not in contestant_names:
            warnings.append(
                f"FJ wager references unknown contestant: '{wager.contestant}' "
                f"(known: {[c.name for c in episode.contestants]})"
            )

    # Check score adjustment contestant names
    for adj in episode.score_adjustments:
        if adj.contestant.lower().strip() not in contestant_names:
            warnings.append(
                f"Score adjustment references unknown contestant: '{adj.contestant}' "
                f"(known: {[c.name for c in episode.contestants]})"
            )

    # ── Board position bounds ────────────────────────────────────────

    for clue in episode.clues:
        if clue.round in ("J!", "Double J!"):
            if not (1 <= clue.board_row <= 5):
                warnings.append(f"Clue '{clue.clue_text[:30]}' has invalid board_row: {clue.board_row}")
            if not (1 <= clue.board_col <= 6):
                warnings.append(f"Clue '{clue.clue_text[:30]}' has invalid board_col: {clue.board_col}")

    # ── Duplicate board positions per category+round ─────────────────
    # The LLM frequently hallucidates board_row since it can't see the board.
    # Two clues in the same category+round should never share a board_row.

    for round_name, round_clues in [("J!", j_clues), ("Double J!", dj_clues)]:
        seen_positions: dict[str, set[int]] = {}
        for clue in round_clues:
            cat_key = clue.category.lower().strip()
            seen_positions.setdefault(cat_key, set())
            if clue.board_row in seen_positions[cat_key]:
                warnings.append(f"{round_name}: duplicate board_row {clue.board_row} in category '{clue.category}'")
            seen_positions[cat_key].add(clue.board_row)

    # ── Per-clue data quality ────────────────────────────────────────

    empty_text_count = 0
    empty_response_count = 0
    inverted_timestamp_count = 0
    zero_start_mid_game_count = 0
    long_read_duration_count = 0
    buzz_before_host_start_count = 0

    for clue in episode.clues:
        # Empty clue text
        if not clue.clue_text or not clue.clue_text.strip():
            empty_text_count += 1

        # Empty correct response
        if not clue.correct_response or not clue.correct_response.strip():
            empty_response_count += 1

        # Inverted timestamps (finish before start)
        if clue.host_finish_timestamp_ms < clue.host_start_timestamp_ms:
            inverted_timestamp_count += 1

        # Zero timestamp for non-first clue (signals Line ID resolution failure)
        if clue.selection_order > 3 and clue.host_start_timestamp_ms == 0.0:
            zero_start_mid_game_count += 1

        # Unrealistically long clue read (>60s signals bad Line IDs)
        read_duration = clue.host_finish_timestamp_ms - clue.host_start_timestamp_ms
        if read_duration > 60000:
            long_read_duration_count += 1

        # Buzz before host starts reading (not finishes — buzz during reading is legal)
        for attempt in clue.attempts:
            if attempt.buzz_timestamp_ms < clue.host_start_timestamp_ms:
                buzz_before_host_start_count += 1
                break  # One per clue is enough

    if empty_text_count > 0:
        warnings.append(f"{empty_text_count} clue(s) have empty clue_text")
    if empty_response_count > 0:
        warnings.append(f"{empty_response_count} clue(s) have empty correct_response")
    if inverted_timestamp_count > 0:
        warnings.append(f"{inverted_timestamp_count} clue(s) have host_finish < host_start (inverted timestamps)")
    if zero_start_mid_game_count > 0:
        warnings.append(
            f"{zero_start_mid_game_count} mid-game clue(s) have host_start_timestamp_ms=0.0 "
            "(likely Line ID resolution failure)"
        )
    if long_read_duration_count > 0:
        warnings.append(f"{long_read_duration_count} clue(s) have host read duration > 60s (likely bad Line IDs)")
    if buzz_before_host_start_count > 0:
        warnings.append(
            f"{buzz_before_host_start_count} clue(s) have buzz_timestamp before host_start (hallucinated buzz Line ID)"
        )

    # ── Category count sanity check ───────────────────────────────────
    # A standard J! board has exactly 6 unique categories per round.
    # More than 6 suggests the LLM is generating variant spellings.
    # Fewer than 5 with many clues suggests categories are being collapsed.
    for round_name, round_clues in [("J!", j_clues), ("Double J!", dj_clues)]:
        categories = {c.category.lower().strip() for c in round_clues}
        if len(categories) > 6:
            warnings.append(
                f"{round_name} has {len(categories)} distinct categories (expected max 6): {sorted(categories)[:8]}"
            )
        if len(categories) < 5 and len(round_clues) > 15:
            warnings.append(
                f"{round_name} has {len(round_clues)} clues across only {len(categories)} "
                f"categories — possible category merge or extraction failure"
            )

    return warnings


def _deduplicate_clues(all_clues: list[Clue]) -> list[Clue]:
    """
    Deduplicates clues from overlapping chunk regions using a composite key
    of time bucket + round + category. When duplicates collide, keeps the
    clue with more buzz attempts (richer data), breaking ties by longer text.
    """
    unique_clues: dict[str, Clue] = {}

    for clue in all_clues:
        # Dedup key uses category (ground truth from transcript) rather than
        # board_row/board_col (which are hallucinated — LLM can't see the board).
        # 2000ms bucket: WhisperX has ±50-200ms jitter, overlapping chunks can
        # produce the same clue with timestamps differing by 200-500ms.
        # Host reads a clue in 3-15s, so a 2s bucket cannot merge distinct clues
        # within the same category.
        time_bucket = round(clue.host_start_timestamp_ms / 2000.0) * 2000
        cat_normalized = clue.category.lower().strip()
        key = f"{time_bucket}_{clue.round}_{cat_normalized}"

        if key not in unique_clues:
            unique_clues[key] = clue
        else:
            existing = unique_clues[key]
            replaced = False
            reason = ""
            if len(clue.attempts) > len(existing.attempts):
                unique_clues[key] = clue
                replaced = True
                reason = f"more_attempts ({len(clue.attempts)} > {len(existing.attempts)})"
            elif len(clue.attempts) == len(existing.attempts) and len(clue.clue_text) > len(existing.clue_text):
                unique_clues[key] = clue
                replaced = True
                reason = f"longer_text ({len(clue.clue_text)} > {len(existing.clue_text)})"

            logger.debug(
                "Dedup collision",
                key=key,
                replaced=replaced,
                reason=reason if replaced else "kept_existing",
                kept_text_preview=unique_clues[key].clue_text[:50],
            )

    total_dupes = len(all_clues) - len(unique_clues)
    if total_dupes > 0:
        logger.info(
            "Deduplication summary",
            input_clues=len(all_clues),
            unique_clues=len(unique_clues),
            duplicates_removed=total_dupes,
            bucket_width_ms=2000,
        )

    return list(unique_clues.values())
