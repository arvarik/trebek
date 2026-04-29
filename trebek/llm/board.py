"""
Board structure constants and clue value → row inference.

A Jeopardy board is a fixed grid with known value progressions.
Instead of asking the LLM to guess board_row (it can't see the board),
we parse the dollar value from the host's clue selection announcement
(e.g., "Let's try Songwriters for $800") and deterministically map it
to a row using the known value grids.

Supports multiple board formats:
- Standard (post-2001): $200-$1000 / $400-$2000
- Celebrity Jeopardy: $100-$500 / $200-$1000 / $300-$1500 (Triple J!)
- Classic (pre-2001): $100-$500 / $200-$1000
"""

import re
import structlog
from dataclasses import dataclass
from typing import Optional

logger = structlog.get_logger()

# ── Board Format Definitions ────────────────────────────────────────
# Each format maps (round, value) → row. The format is auto-detected
# from the meta extraction's `is_tournament` flag and category count.


@dataclass(frozen=True)
class BoardFormat:
    """Defines the value grid for a specific J! board format."""

    name: str
    j_values: tuple[int, ...]
    dj_values: tuple[int, ...]
    # Triple J! exists only in Celebrity Jeopardy primetime specials
    tj_values: tuple[int, ...] = ()
    categories_per_round: int = 6
    clues_per_category: int = 5


STANDARD_BOARD = BoardFormat(
    name="standard",
    j_values=(200, 400, 600, 800, 1000),
    dj_values=(400, 800, 1200, 1600, 2000),
)

CELEBRITY_BOARD = BoardFormat(
    name="celebrity",
    j_values=(100, 200, 300, 400, 500),
    dj_values=(200, 400, 600, 800, 1000),
    tj_values=(300, 600, 900, 1200, 1500),
)

CLASSIC_BOARD = BoardFormat(
    name="classic",
    j_values=(100, 200, 300, 400, 500),
    dj_values=(200, 400, 600, 800, 1000),
)

# Ordered from most common → least common for detection priority
_ALL_FORMATS = [STANDARD_BOARD, CELEBRITY_BOARD, CLASSIC_BOARD]


def detect_board_format(
    is_tournament: bool,
    j_categories: list[str],
    dj_categories: list[str],
    transcript_text: str = "",
) -> BoardFormat:
    """Auto-detect the board format from episode metadata.

    Detection heuristics (in priority order):
    1. If transcript contains "$100" values but NOT "$200" in J!-context → classic/celebrity
    2. If triple J! categories exist → celebrity
    3. Default → standard (post-2001)
    """
    # Celebrity J! uses $100-$500 in J! round and has 3 rounds
    # Standard uses $200-$1000 in J! round
    if transcript_text:
        # Look for value patterns in first 40% of transcript (J! round area)
        j_section = transcript_text[: len(transcript_text) * 2 // 5]
        has_100_values = bool(re.search(r"\$100\b", j_section))
        has_200_values = bool(re.search(r"\$200\b", j_section))

        if has_100_values and not has_200_values:
            logger.info("Detected non-standard board format (classic/$100 values)")
            return CLASSIC_BOARD

    # For now, all standard and tournament games use the same value grid
    return STANDARD_BOARD


# ── Value-to-Row Parsing ────────────────────────────────────────────

# Regex: captures "$800" from "Let's try Songwriters for $800"
# Also handles "the $1,200 clue", "$1200", and "for 800"
_VALUE_PATTERN = re.compile(r"\$?([\d,]+)")


def _parse_dollar_value(text: str) -> Optional[int]:
    """Extract a dollar value from text like '$800', '$1,200', '800'."""
    match = _VALUE_PATTERN.search(text)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def infer_board_row(
    clue_value: Optional[int],
    round_name: str,
    board_format: BoardFormat = STANDARD_BOARD,
) -> int:
    """Map a clue's dollar value to its board row (1-5).

    Returns the row index (1-indexed) or falls back to a middle row (3)
    if the value doesn't match any known grid position.
    """
    if clue_value is None:
        return 3  # Fallback: middle row

    if round_name == "J!":
        values = board_format.j_values
    elif round_name == "Double J!":
        values = board_format.dj_values
    else:
        return 1  # FJ / Tiebreaker: row 1

    if clue_value in values:
        return values.index(clue_value) + 1

    # Fuzzy match: find the closest value (handles rounding errors in speech)
    closest_idx = min(range(len(values)), key=lambda i: abs(values[i] - clue_value))
    if abs(values[closest_idx] - clue_value) <= 100:
        logger.debug(
            "Fuzzy-matched clue value to board row",
            raw_value=clue_value,
            matched_value=values[closest_idx],
            row=closest_idx + 1,
        )
        return closest_idx + 1

    logger.warning(
        "Clue value doesn't match any board row",
        value=clue_value,
        round=round_name,
        format=board_format.name,
        expected=values,
    )
    return 3  # Fallback: middle row


def infer_board_row_from_selection_text(
    selection_text: str,
    round_name: str,
    board_format: BoardFormat = STANDARD_BOARD,
) -> int:
    """Parse the board row from a clue selection announcement.

    The host or contestant says something like:
    - "Let's try Songwriters for $800"
    - "I'll take Around North America for $1,000"
    - "Songwriters, $600"

    We extract the dollar value and map it to the correct row.
    """
    value = _parse_dollar_value(selection_text)
    return infer_board_row(value, round_name, board_format)


# ── Round Manifest for Gap Detection ────────────────────────────────
# A standard game expects 6 categories × 5 clues per round.
# Tournament games sometimes have different structures — the
# category count from meta extraction is used as ground truth.


@dataclass
class RoundManifest:
    """Expected board structure for a single round."""

    round_name: str
    categories: list[str]
    clues_per_category: int = 5

    @property
    def expected_total(self) -> int:
        return len(self.categories) * self.clues_per_category


def build_manifests(
    j_categories: list[str],
    dj_categories: list[str],
    board_format: BoardFormat = STANDARD_BOARD,
) -> tuple["RoundManifest", "RoundManifest"]:
    """Build round manifests from meta extraction categories.

    Uses actual extracted category count (not hardcoded 6) to handle
    tournament and special episode formats gracefully.
    """
    return (
        RoundManifest(
            round_name="J!",
            categories=j_categories,
            clues_per_category=board_format.clues_per_category,
        ),
        RoundManifest(
            round_name="Double J!",
            categories=dj_categories,
            clues_per_category=board_format.clues_per_category,
        ),
    )
