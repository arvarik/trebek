from typing import Optional, Literal
from pydantic import BaseModel, Field
from trebek.schemas import Contestant, FinalJeopardy, ScoreAdjustment


class PartialEpisodeMeta(BaseModel):
    reasoning_scratchpad: str = Field(description="Write a chronological narrative of the events...")
    episode_date: str
    host_name: str
    is_tournament: bool
    contestants: list[Contestant]
    final_jeopardy: FinalJeopardy
    score_adjustments: list[ScoreAdjustment]


class BuzzAttemptExtraction(BaseModel):
    attempt_order: int
    speaker: str
    response_given: str
    is_correct: bool
    buzz_line_id: str
    is_lockout_inferred: bool


class ClueExtraction(BaseModel):
    round: Literal["Jeopardy", "Double Jeopardy", "Final Jeopardy", "Tiebreaker"]
    category: str
    board_row: int
    board_col: int
    is_daily_double: bool
    requires_visual_context: bool
    host_read_start_line_id: str
    host_read_end_line_id: str
    daily_double_wager: Optional[str] = Field(
        description="The wagered amount as a string (e.g. '2000', 'True Daily Double'), or null if not a Daily Double."
    )
    wagerer_name: Optional[str] = Field(
        description="Name of the contestant who found the Daily Double, or null if not a Daily Double."
    )
    correct_response: str
    attempts: list[BuzzAttemptExtraction] = Field(
        description="Chronological list of buzz attempts. Empty list for Triple Stumpers."
    )


class PartialClues(BaseModel):
    clues: list[ClueExtraction]


class EpisodeSkeleton(BaseModel):
    reasoning_scratchpad: str = Field(description="...")
    jeopardy_categories: list[str]
    double_jeopardy_categories: list[str]
    total_jeopardy_clues_played: int
    total_double_jeopardy_clues_played: int
    daily_double_count: int
