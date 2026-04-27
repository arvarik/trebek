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
            if canonical:
                clue.wagerer_name = canonical

    # Normalize score_adjustment contestant names (they also build FK contestant_ids)
    for adj in score_adjustments:
        if hasattr(adj, "contestant") and adj.contestant:
            canonical = variant_map.get(adj.contestant.lower())
            if canonical:
                adj.contestant = canonical

    # Normalize FinalJeopardyWager contestant names (for future FK safety)
    for wager in fj_wagers:
        if hasattr(wager, "contestant") and wager.contestant:
            canonical = variant_map.get(wager.contestant.lower())
            if canonical:
                wager.contestant = canonical

    if unmapped:
        logger.warning(
            "Speaker normalization: unmapped speakers remain",
            unmapped=unmapped,
            variant_map={k: v for k, v in variant_map.items() if not k.startswith("speaker_")},
        )
    else:
        logger.info(
            "Speaker normalization complete — all speakers mapped to contestant full names",
            total_attempts=sum(len(c.attempts) for c in clues),
        )
