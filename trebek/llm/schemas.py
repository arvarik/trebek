from typing import Optional, Literal
from pydantic import BaseModel, Field
from trebek.schemas import Contestant, FinalJep, ScoreAdjustment


class PartialEpisodeMeta(BaseModel):
    episode_date: str
    host_name: str
    is_tournament: bool
    contestants: list[Contestant]
    jeopardy_categories: list[str]
    double_jep_categories: list[str]
    final_jep: FinalJep
    score_adjustments: list[ScoreAdjustment]


class BuzzAttemptExtraction(BaseModel):
    attempt_order: int
    speaker: str
    response_given: str = Field(
        description="The literal text of what the contestant said, verbatim from the transcript."
    )
    is_correct: bool
    buzz_line_id: str
    is_lockout_inferred: bool


class ClueExtraction(BaseModel):
    round: Literal["J!", "Double J!", "Final J!", "Tiebreaker"]
    category: str
    board_row: int
    board_col: int
    is_daily_double: bool
    requires_visual_context: bool
    host_read_start_line_id: str
    host_read_end_line_id: str
    daily_double_wager: Optional[str] = Field(
        default=None,
        description="The wagered amount as a string (e.g. '2000', 'True Daily Double'), or null if not a Daily Double.",
    )
    wagerer_name: Optional[str] = Field(
        default=None, description="Name of the contestant who found the Daily Double, or null if not a Daily Double."
    )
    correct_response: str = Field(
        description=(
            "The correct response in J! question format. MUST start with 'What is', 'Who is', 'What are', "
            "'Who are', 'Where is', etc. For example: 'What is Paris?' not just 'Paris'. "
            "If the contestant gave the correct response, use their exact phrasing. "
            "If no one answered correctly, construct the proper question form."
        )
    )
    attempts: list[BuzzAttemptExtraction] = Field(
        default_factory=list, description="Chronological list of buzz attempts. Empty list for Triple Stumpers."
    )


class PartialClues(BaseModel):
    clues: list[ClueExtraction]


class EpisodeSkeleton(BaseModel):
    jeopardy_categories: list[str]
    double_jep_categories: list[str]
    total_jep_clues_played: int
    total_double_jep_clues_played: int
    daily_double_count: int


def create_dynamic_clue_schema(categories: list[str], contestants: list[str]) -> type[BaseModel]:
    from pydantic import create_model

    # Provide fallbacks to avoid Empty Literal error if lists are somehow empty
    cat_tuple = tuple(categories) if categories else ("Unknown Category",)
    cont_tuple = tuple(contestants) if contestants else ("Unknown Contestant",)

    CategoryLiteral = Literal[cat_tuple]  # type: ignore
    SpeakerLiteral = Literal[cont_tuple]  # type: ignore

    DynamicBuzzAttempt = create_model(
        "BuzzAttemptExtraction",
        __base__=BuzzAttemptExtraction,
        speaker=(SpeakerLiteral, ...),
    )

    DynamicClueExtraction = create_model(
        "ClueExtraction",
        __base__=ClueExtraction,
        category=(CategoryLiteral, ...),
        wagerer_name=(
            Optional[SpeakerLiteral],
            Field(
                default=None,
                description="Name of the contestant who found the Daily Double, or null if not a Daily Double.",
            ),
        ),
        attempts=(
            list[DynamicBuzzAttempt],  # type: ignore
            Field(
                default_factory=lambda: [],
                description="Chronological list of buzz attempts. Empty list for Triple Stumpers.",
            ),
        ),
    )

    DynamicPartialClues = create_model(
        "PartialClues",
        __base__=PartialClues,
        clues=(list[DynamicClueExtraction], ...),  # type: ignore
    )

    return DynamicPartialClues
