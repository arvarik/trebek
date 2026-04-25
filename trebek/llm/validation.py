from trebek.schemas import Episode, Clue


def _validate_extraction_integrity(episode: Episode) -> list[str]:
    warnings: list[str] = []

    j_clues = [c for c in episode.clues if c.round == "Jeopardy"]
    dj_clues = [c for c in episode.clues if c.round == "Double Jeopardy"]

    if len(j_clues) > 30:
        warnings.append(f"Jeopardy round has {len(j_clues)} clues (max 30)")
    if len(dj_clues) > 30:
        warnings.append(f"Double Jeopardy round has {len(dj_clues)} clues (max 30)")
    if len(j_clues) == 0:
        warnings.append("No Jeopardy round clues extracted")
    if len(dj_clues) == 0:
        warnings.append("No Double Jeopardy round clues extracted")

    for round_name, round_clues in [("Jeopardy", j_clues), ("Double Jeopardy", dj_clues)]:
        sorted_round = sorted(round_clues, key=lambda c: c.host_start_timestamp_ms)
        for i in range(1, len(sorted_round)):
            if sorted_round[i].host_start_timestamp_ms < sorted_round[i - 1].host_finish_timestamp_ms:
                warnings.append(
                    f"{round_name} clue {i + 1} overlaps with clue {i}: "
                    f"start {sorted_round[i].host_start_timestamp_ms} < "
                    f"prev finish {sorted_round[i - 1].host_finish_timestamp_ms}"
                )
                break

    dd_j = sum(1 for c in j_clues if c.is_daily_double)
    dd_dj = sum(1 for c in dj_clues if c.is_daily_double)
    total_dd = dd_j + dd_dj

    if total_dd > 3:
        warnings.append(f"Found {total_dd} Daily Doubles (expected max 3)")
    if dd_j > 1:
        warnings.append(f"Found {dd_j} Daily Doubles in Jeopardy round (expected 1)")
    if dd_dj > 2:
        warnings.append(f"Found {dd_dj} Daily Doubles in Double Jeopardy (expected 2)")

    contestant_names = {c.name.lower().strip() for c in episode.contestants}
    unknown_buzzers = set()
    for clue in episode.clues:
        for attempt in clue.attempts:
            if attempt.speaker.lower().strip() not in contestant_names:
                unknown_buzzers.add(attempt.speaker)

    if unknown_buzzers:
        warnings.append(f"Unknown contestants in buzz attempts: {unknown_buzzers}")

    for clue in episode.clues:
        if clue.round in ("Jeopardy", "Double Jeopardy"):
            if not (1 <= clue.board_row <= 5):
                warnings.append(f"Clue '{clue.clue_text[:30]}' has invalid board_row: {clue.board_row}")
            if not (1 <= clue.board_col <= 6):
                warnings.append(f"Clue '{clue.clue_text[:30]}' has invalid board_col: {clue.board_col}")

    return warnings


def _deduplicate_clues(all_clues: list[Clue]) -> list[Clue]:
    unique_clues: dict[str, Clue] = {}

    for clue in all_clues:
        time_bucket = round(clue.host_start_timestamp_ms / 100.0) * 100
        key = f"{time_bucket}_{clue.round}_{clue.board_row}_{clue.board_col}"

        if key not in unique_clues:
            unique_clues[key] = clue
        else:
            existing = unique_clues[key]
            if len(clue.attempts) > len(existing.attempts):
                unique_clues[key] = clue
            elif len(clue.attempts) == len(existing.attempts) and len(clue.clue_text) > len(existing.clue_text):
                unique_clues[key] = clue

    return list(unique_clues.values())
