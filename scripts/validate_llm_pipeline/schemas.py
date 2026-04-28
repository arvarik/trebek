"""
Pydantic schemas (inlined from trebek) and result tracking for the standalone validator.

These are intentionally duplicated from trebek.schemas and trebek.llm.schemas
so this package can run on remote machines with zero trebek imports.
"""

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
    round: Literal["J!", "Double J!", "Final J!", "Tiebreaker"]
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
    final_jep: FJ
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
