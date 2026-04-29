"""
Speaker name normalization for post-extraction cleanup.

Resolves the many inconsistent speaker name variants the LLM produces
(raw diarization IDs, first names, abbreviations) to canonical contestant
full names, preventing FK constraint failures in the relational DB commit.

This is the most complex pure function in the codebase — see
``tests/test_speaker_normalization.py`` for comprehensive edge-case coverage.
"""

import structlog
from typing import Any, Dict, Optional

from trebek.schemas import Clue

logger = structlog.get_logger()


def _normalize_speaker_names(
    clues: "list[Clue]",
    speaker_mapping: "Dict[str, str]",
    contestant_names: "list[str]",
    host_name: str = "",
    score_adjustments: "Optional[list[Any]]" = None,
    fj_wagers: "Optional[list[Any]]" = None,
) -> None:
    """
    Post-extraction normalization: maps all speaker name variants in buzz attempts
    to canonical contestant full names from the episode metadata.

    The LLM inconsistently uses:
      - Raw diarization IDs: SPEAKER_00, SPEAKER_02
      - Pass 1 mapped names: Rachel, Lawrence, Kamau
      - Full names: Harrison Whitaker

    This function resolves them all to the canonical full names (e.g. Rachel Bernstein)
    to prevent FK constraint failures in the relational DB commit.
    """
    if score_adjustments is None:
        score_adjustments = []
    if fj_wagers is None:
        fj_wagers = []

    # Build a comprehensive lookup: variant → canonical full name
    variant_map: Dict[str, str] = {}

    # 1. Exact matches (case-insensitive)
    for name in contestant_names:
        variant_map[name.lower()] = name

    # 2. Every individual word in the name → canonical name
    #    Handles: "Kamau" → "W. Kamau Bell", "Rachel" → "Rachel Bernstein",
    #    "Subba" → "Lawrence Subba", "DeFrank" → "Alex DeFrank"
    for name in contestant_names:
        for part in name.split():
            part_lower = part.lower().rstrip(".")  # strip trailing dots from initials like "W."
            # Skip very short parts (initials like "W") to avoid false positives
            if len(part_lower) <= 1:
                continue
            if part_lower in variant_map and variant_map[part_lower] != name:
                # Name part collision — two contestants share a name part (e.g. two "James")
                # Remove the ambiguous mapping so the LLM must use full names
                logger.warning(
                    "Speaker normalization: ambiguous name part, requiring full name",
                    part=part_lower,
                    contestant_a=variant_map[part_lower],
                    contestant_b=name,
                )
                del variant_map[part_lower]
            elif part_lower not in variant_map:
                variant_map[part_lower] = name

    # 3. SPEAKER_XX → Pass 1 name → contestant full name
    #    Skip the host — they shouldn't appear in buzz attempts
    host_lower = host_name.lower().strip() if host_name else ""
    for speaker_id, mapped_name in speaker_mapping.items():
        sid = speaker_id.lower()
        # Skip the host entirely — they don't buzz in
        if mapped_name.lower().strip() == host_lower:
            continue
        # Try to resolve the Pass 1 name to a full contestant name
        resolved = variant_map.get(mapped_name.lower())
        if not resolved:
            # Try each word of the mapped name
            for part in mapped_name.split():
                part_lower = part.lower().rstrip(".")
                if len(part_lower) > 1:
                    resolved = variant_map.get(part_lower)
                    if resolved:
                        break
        if not resolved:
            # Substring containment: does the mapped name appear IN any contestant name?
            mapped_lower = mapped_name.lower()
            for cname in contestant_names:
                if mapped_lower in cname.lower():
                    resolved = cname
                    break
        if resolved:
            variant_map[sid] = resolved
            # Also map the Pass 1 name itself
            if mapped_name.lower() not in variant_map:
                variant_map[mapped_name.lower()] = resolved
        else:
            # Pass 1 name doesn't match any contestant — map SPEAKER_XX to it directly
            variant_map[sid] = mapped_name

    # 4. Abbreviated speaker IDs: S0, S00, S01, s0, s00, etc.
    #    The system prompt tells the LLM to use S0=SPEAKER_00, S1=SPEAKER_01, etc.
    #    The LLM sometimes outputs these abbreviated forms in buzz attempt speaker fields.
    for speaker_id, mapped_name in speaker_mapping.items():
        if mapped_name.lower().strip() == host_lower:
            continue
        # Extract the numeric part from SPEAKER_XX
        if speaker_id.upper().startswith("SPEAKER_"):
            num = speaker_id[len("SPEAKER_") :]
            resolved = variant_map.get(speaker_id.lower())
            if resolved:
                # Map all abbreviated variants
                variant_map[f"s{num}".lower()] = resolved  # s00, s01
                variant_map[f"s{int(num)}".lower()] = resolved  # s0, s1
                # Also map zero-padded variants
                variant_map[f"s{int(num):02d}"] = resolved  # s00, s01

    unmapped: set[str] = set()

    # Normalize buzz attempt speakers
    for clue in clues:
        for attempt in clue.attempts:
            canonical = variant_map.get(attempt.speaker.lower())
            if not canonical:
                # Last resort: substring containment check
                speaker_lower = attempt.speaker.lower()
                for cname in contestant_names:
                    if speaker_lower in cname.lower() or cname.lower().startswith(speaker_lower):
                        canonical = cname
                        break
            if canonical:
                attempt.speaker = canonical
            else:
                unmapped.add(attempt.speaker)

        # Also normalize wagerer_name on daily doubles
        if clue.wagerer_name:
            canonical = variant_map.get(clue.wagerer_name.lower())
            if not canonical:
                # Fuzzy match wagerer too
                canonical = _fuzzy_match_contestant(clue.wagerer_name, contestant_names)
            if canonical:
                clue.wagerer_name = canonical

    # Normalize score_adjustment contestant names (they also build FK contestant_ids)
    for adj in score_adjustments:
        if hasattr(adj, "contestant") and adj.contestant:
            canonical = variant_map.get(adj.contestant.lower())
            if not canonical:
                canonical = _fuzzy_match_contestant(adj.contestant, contestant_names)
            if canonical:
                adj.contestant = canonical

    # Normalize FinalJepWager contestant names (for future FK safety)
    for wager in fj_wagers:
        if hasattr(wager, "contestant") and wager.contestant:
            canonical = variant_map.get(wager.contestant.lower())
            if not canonical:
                canonical = _fuzzy_match_contestant(wager.contestant, contestant_names)
            if canonical:
                wager.contestant = canonical

    # ── Hard cleanup sweep ───────────────────────────────────────────
    # After soft normalization, any remaining unmapped speakers will crash
    # the database FK constraints. Fuzzy-match or drop them.
    if True:  # Always run hard cleanup to catch any soft-mapped but invalid names
        contestant_set_lower = {c.lower() for c in contestant_names}
        dropped_total = 0
        fuzzy_resolved = 0

        # Build a set of known host-associated IDs to exclude from fuzzy matching.
        # Without this, abbreviated host IDs like "S00" could fuzzy-match to a
        # contestant name (e.g. "S00" → "Dan" at edit distance 2).
        host_ids: set[str] = set()
        if host_lower:
            host_ids.add(host_lower)
            for speaker_id, mapped_name in speaker_mapping.items():
                if mapped_name.lower().strip() == host_lower:
                    host_ids.add(speaker_id.lower())
                    if speaker_id.upper().startswith("SPEAKER_"):
                        num = speaker_id[len("SPEAKER_") :]
                        host_ids.add(f"s{num}".lower())
                        host_ids.add(f"s{int(num)}".lower())
                        host_ids.add(f"s{int(num):02d}")

        for clue in clues:
            surviving_attempts = []
            for attempt in clue.attempts:
                if attempt.speaker.lower() in contestant_set_lower:
                    surviving_attempts.append(attempt)
                    continue

                # Already a canonical name (case-exact match)?
                if attempt.speaker in contestant_names:
                    surviving_attempts.append(attempt)
                    continue

                # Skip host-associated IDs entirely — host doesn't buzz in
                if attempt.speaker.lower() in host_ids:
                    logger.warning(
                        "Speaker normalization: dropping host speaker from buzz attempts",
                        speaker=attempt.speaker,
                        clue_category=clue.category,
                    )
                    dropped_total += 1
                    continue

                # Try fuzzy matching (edit distance)
                fuzzy_match = _fuzzy_match_contestant(attempt.speaker, contestant_names)
                if fuzzy_match:
                    logger.info(
                        "Speaker normalization: fuzzy-resolved unmapped speaker",
                        original=attempt.speaker,
                        resolved_to=fuzzy_match,
                    )
                    attempt.speaker = fuzzy_match
                    surviving_attempts.append(attempt)
                    fuzzy_resolved += 1
                else:
                    logger.warning(
                        "Speaker normalization: dropping unresolvable buzz attempt",
                        speaker=attempt.speaker,
                        clue_category=clue.category,
                        clue_selection_order=clue.selection_order,
                    )
                    dropped_total += 1

            clue.attempts = surviving_attempts

        logger.warning(
            "Speaker normalization: hard cleanup complete",
            unmapped_speakers=unmapped,
            fuzzy_resolved=fuzzy_resolved,
            attempts_dropped=dropped_total,
            variant_map={k: v for k, v in variant_map.items() if not k.startswith("speaker_")},
        )


def _fuzzy_match_contestant(speaker: str, contestant_names: "list[str]", max_distance: int = 3) -> "Optional[str]":
    """
    Attempts to fuzzy-match a speaker name to a contestant using edit distance.

    Returns the closest contestant name if within max_distance, else None.
    Uses a simple Levenshtein implementation to avoid adding a dependency.
    """
    speaker_lower = speaker.lower().strip()
    best_match: "Optional[str]" = None
    best_distance = max_distance + 1

    for cname in contestant_names:
        # Try matching against full name and each name part
        candidates = [cname.lower()] + [p.lower() for p in cname.split() if len(p) > 1]
        for candidate in candidates:
            dist = _levenshtein(speaker_lower, candidate)
            if dist < best_distance:
                best_distance = dist
                best_match = cname

    return best_match if best_distance <= max_distance else None


def _levenshtein(s1: str, s2: str) -> int:
    """Minimal Levenshtein distance implementation (no external deps)."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def _reconcile_speaker_mapping(
    speaker_mapping: "Dict[str, str]",
    contestant_names: "list[str]",
    host_name: str = "",
) -> "Dict[str, str]":
    """
    Reconcile Pass 1 audio-derived speaker names against Pass 2 contestant names.

    Pass 1 (audio anchoring via Flash) produces speaker names from vocal timbre
    recognition — fast and cheap but phonetically approximate (e.g., "Paulo").
    Pass 2 (full transcript analysis via Pro) produces definitive contestant
    names (e.g., "Paolo Pasco").

    Without reconciliation, the clue extraction prompt contains contradictory
    instructions: the system prompt says ``S3 = Paulo`` while the schema
    constrains speakers to ``Literal["Paolo Pasco"]``. This causes the LLM to
    output inconsistent speaker names, which the FK pre-commit filter then drops.

    This function fuzzy-matches each Pass 1 name against the Pass 2 contestant
    list and replaces mismatches with the canonical name, ensuring the clue
    extraction prompt is self-consistent.
    """
    from trebek.config import KNOWN_HOSTS

    reconciled: Dict[str, str] = {}
    host_lower = host_name.lower().strip() if host_name else ""
    contestant_lower_map = {c.lower().strip(): c for c in contestant_names}
    known_hosts_lower = {h.lower() for h in KNOWN_HOSTS}

    for speaker_id, pass1_name in speaker_mapping.items():
        p1_lower = pass1_name.lower().strip()

        # 1. Exact match to contestant → keep
        if p1_lower in contestant_lower_map:
            reconciled[speaker_id] = contestant_lower_map[p1_lower]
            continue

        # 2. Known host → keep
        if p1_lower in known_hosts_lower or p1_lower == host_lower:
            reconciled[speaker_id] = pass1_name
            continue

        # 3. Substring match (e.g., "Ame" matches "Ame Fluitt")
        substring_match = None
        for cname in contestant_names:
            if p1_lower in cname.lower() or cname.lower().startswith(p1_lower):
                substring_match = cname
                break
        if substring_match:
            if substring_match != pass1_name:
                logger.info(
                    "Speaker reconciliation: substring match",
                    pass1_name=pass1_name,
                    resolved_to=substring_match,
                    speaker_id=speaker_id,
                )
            reconciled[speaker_id] = substring_match
            continue

        # 4. Fuzzy match (Levenshtein ≤ 3)
        fuzzy_match = _fuzzy_match_contestant(pass1_name, contestant_names, max_distance=3)
        if fuzzy_match:
            logger.warning(
                "Speaker reconciliation: fuzzy-resolved Pass 1 name",
                pass1_name=pass1_name,
                resolved_to=fuzzy_match,
                speaker_id=speaker_id,
            )
            reconciled[speaker_id] = fuzzy_match
            continue

        # 5. No match — keep original (will be handled downstream)
        logger.warning(
            "Speaker reconciliation: unresolvable Pass 1 name",
            pass1_name=pass1_name,
            speaker_id=speaker_id,
            contestant_names=contestant_names,
        )
        reconciled[speaker_id] = pass1_name

    return reconciled


def _resolve_host_from_pass1(
    speaker_mapping: "Dict[str, str]",
) -> "Optional[str]":
    """
    Extract the host identity from the Pass 1 speaker mapping.

    Checks if any Pass 1 mapped name matches a known J! host. Returns the
    canonical host name if found, else None.
    """
    from trebek.config import KNOWN_HOSTS

    known_lower = {h.lower(): h for h in KNOWN_HOSTS}
    for _speaker_id, name in speaker_mapping.items():
        name_lower = name.lower().strip()
        if name_lower in known_lower:
            return known_lower[name_lower]
        # Check partial match (e.g., "Ken" matches "Ken Jennings")
        for known_name in KNOWN_HOSTS:
            parts = known_name.lower().split()
            if name_lower in parts:
                return known_name
    return None
