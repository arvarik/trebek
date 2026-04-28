"""
Core Pydantic schemas for J! episode data.

Defines the canonical data model for episodes, clues, contestants,
buzz attempts, score adjustments, Final J!, and job telemetry.
These schemas are used throughout the pipeline for validation and serialization.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union


class BuzzAttempt(BaseModel):
    attempt_order: int = Field(description="1 for first buzz, 2/3 for rebounds.")
    speaker: str = Field(description="The contestant who rang in.")
    response_given: str = Field(description="The literal text of what they guessed, capturing stutters/ums.")
    is_correct: bool = Field(description="Whether the host ruled it correct.")
    buzz_timestamp_ms: float = Field(description="The exact timestamp when the contestant buzzed in.")
    response_start_timestamp_ms: float = Field(description="The exact timestamp when the contestant began speaking.")
    is_lockout_inferred: bool = Field(
        description="True if they appeared to jump the gun and suffered the 0.25s penalty."
    )


class Clue(BaseModel):
    round: Literal["J!", "Double J!", "Final J!", "Tiebreaker"]
    category: str
    board_row: int = Field(description="1 to 5 (e.g., $200 to $1000)")
    board_col: int = Field(description="1 to 6")
    selection_order: int = Field(description="Chronological order (1 to 60) this clue was picked.")
    is_daily_double: bool
    requires_visual_context: bool = Field(
        description="True if the transcript implies contestants are looking at a picture/video clue. Triggers multimodal extraction."
    )
    host_start_timestamp_ms: float = Field(
        description="The exact WhisperX timestamp when the host starts reading the clue."
    )
    host_finish_timestamp_ms: float = Field(
        description="The exact WhisperX timestamp when the host finishes reading the clue."
    )
    clue_syllable_count: int = Field(description="Calculated syllable count for the clue text.")
    daily_double_wager: Optional[Union[int, Literal["True Daily Double"]]] = Field(
        default=None, description="The amount wagered. Can be an integer or the literal string 'True Daily Double'."
    )
    wagerer_name: Optional[str] = Field(default=None, description="Name of the contestant who found the Daily Double.")
    clue_text: str = Field(description="The text read by the host (the answer).")
    correct_response: str = Field(description="The accepted correct response (the question).")

    attempts: List[BuzzAttempt] = Field(
        default_factory=list, description="Chronological list of buzz attempts. Empty list for Triple Stumpers."
    )


class ScoreAdjustment(BaseModel):
    contestant: str
    points_adjusted: int = Field(description="Positive or negative integer adjustment.")
    reason: str = Field(description="The host's explanation for the score change.")
    effective_after_clue_selection_order: int = Field(
        description="Chronological index of clue selection after which this adjustment strictly applies."
    )


class FinalJepWager(BaseModel):
    contestant: str
    wager: int
    response: str
    is_correct: bool


class FinalJep(BaseModel):
    category: str
    clue_text: str
    wagers_and_responses: List[FinalJepWager]


class Contestant(BaseModel):
    name: str
    podium_position: int = Field(description="1 (left), 2 (center), or 3 (right).", ge=1, le=3)
    occupational_category: str = Field(description="Classified by LLM (e.g., 'Academia', 'STEM', 'Law').")
    is_returning_champion: bool
    description: str = Field(description="Hometown/occupational context from the interview segment.")


class Episode(BaseModel):
    episode_date: str
    host_name: str = Field(description="Name of the host for this episode.")
    is_tournament: bool = Field(description="True if this is a tournament episode.")
    contestants: List[Contestant]
    clues: List[Clue]
    final_jep: FinalJep
    score_adjustments: List[ScoreAdjustment]


class JobTelemetry(BaseModel):
    episode_id: str
    peak_vram_mb: Optional[float] = Field(default=None, ge=0.0)
    avg_gpu_utilization_pct: Optional[float] = Field(default=None, ge=0.0)
    stage_ingestion_ms: Optional[float] = Field(default=None, ge=0.0)
    stage_gpu_extraction_ms: Optional[float] = Field(default=None, ge=0.0)
    stage_commercial_filtering_ms: Optional[float] = Field(default=None, ge=0.0)
    stage_structured_extraction_ms: Optional[float] = Field(default=None, ge=0.0)
    stage_multimodal_ms: Optional[float] = Field(default=None, ge=0.0)
    stage_vectorization_ms: Optional[float] = Field(default=None, ge=0.0)
    gemini_total_input_tokens: Optional[int] = Field(default=None, ge=0)
    gemini_total_output_tokens: Optional[int] = Field(default=None, ge=0)
    gemini_total_cached_tokens: Optional[int] = Field(default=None, ge=0)
    gemini_total_cost_usd: Optional[float] = Field(default=None, ge=0.0)
    gemini_api_latency_ms: Optional[float] = Field(default=None, ge=0.0)
    pydantic_retry_count: Optional[int] = Field(default=None, ge=0)
