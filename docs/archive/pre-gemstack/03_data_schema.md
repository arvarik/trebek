# Data Schema

## 1. Designing for the Reality of the Game
The data schema defines the structure extracted by Gemini 3.1 Pro via Pydantic Structured Outputs. We are building a **High-Fidelity Cognitive, Temporal, and Spatiotemporal Capture Engine**. To train predictive ML models, we need an Event-Sourced Relational Model that captures the "Running State" at the millisecond a decision is made.

## 2. Event-Sourced Relational Model (SQLite for ML)

This is the production-grade database architecture designed to scale directly to PyTorch DataLoaders or pandas DataFrames.

### Core Metadata (The Anchors)
```sql
CREATE TABLE episodes (
    episode_id TEXT PRIMARY KEY,
    air_date DATE,
    host_name TEXT, -- ML Feature: Different hosts have different reading WPM paces
    is_tournament BOOLEAN -- Tournaments drastically alter wagering risk-theory math
);

CREATE TABLE contestants (
    contestant_id TEXT PRIMARY KEY,
    name TEXT,
    occupational_category TEXT, -- Classified by LLM (e.g., "Academia", "STEM", "Law")
    is_returning_champion BOOLEAN
);

CREATE TABLE episode_performances (
    episode_id TEXT REFERENCES episodes(episode_id),
    contestant_id TEXT REFERENCES contestants(contestant_id),
    podium_position INTEGER CHECK(podium_position IN (1, 2, 3)), -- Physical distance impacts micro-latency
    coryat_score INTEGER, -- ML Target: Pure trivia score (ignores all wagering luck entirely)
    final_score INTEGER,
    forrest_bounce_index REAL, -- ML Feature: How aggressively they "hunt" the board vs playing top-down (0.0 to 1.0)
    PRIMARY KEY (episode_id, contestant_id)
);
```

### The Game Board & Linguistic Matrix
```sql
CREATE TABLE clues (
    clue_id TEXT PRIMARY KEY,
    episode_id TEXT REFERENCES episodes(episode_id),
    round TEXT CHECK(round IN ('Jeopardy', 'Double', 'Final', 'Tiebreaker')),
    category TEXT,
    board_row INTEGER, -- 1 to 5 (e.g., $200 to $1000)
    board_col INTEGER, -- 1 to 6
    selection_order INTEGER, -- Chronological order (1 to 60) this clue was picked
    
    clue_text TEXT,
    correct_response TEXT,
    is_daily_double BOOLEAN,
    is_triple_stumper BOOLEAN, -- True if no one answered correctly
    daily_double_wager TEXT, -- Can be integer or 'True Daily Double'
    wagerer_name TEXT, -- Name of the contestant who found the Daily Double
    
    -- Extracted Linguistic & Temporal Features
    requires_visual_context BOOLEAN,
    host_start_timestamp_ms REAL,
    host_finish_timestamp_ms REAL,
    clue_syllable_count INTEGER,
    host_speech_rate_wpm REAL, -- Derived: (Syllables / duration)
    
    -- Board Control
    selector_had_board_control BOOLEAN, -- True if the selector held board control (isolates Forrest Bouncing)

    -- RAG Embeddings via sqlite-vec
    clue_embedding FLOAT[1536], -- Vectorized: Category + Clue
    response_embedding FLOAT[1536], -- Vectorized: Correct Response
    semantic_lateral_distance REAL -- Cosine distance between clue and response embeddings
);
```

### Behavioral & Acoustic Physics (The ML Goldmine)
```sql
CREATE TABLE buzz_attempts (
    attempt_id TEXT PRIMARY KEY,
    clue_id TEXT REFERENCES clues(clue_id),
    contestant_id TEXT REFERENCES contestants(contestant_id),
    
    -- Temporal & Reflex Features
    attempt_order INTEGER, -- 1st buzz, or 2nd/3rd (Rebound)
    buzz_timestamp_ms REAL,
    podium_light_timestamp_ms REAL, -- Visual lockout disengage timestamp
    true_buzzer_latency_ms REAL, -- Derived: buzz_timestamp - podium_light_timestamp
    is_lockout_inferred BOOLEAN, -- Did they jump the gun and suffer the 0.25s penalty?
    
    -- Acoustic Confidence Features
    response_given TEXT,
    is_correct BOOLEAN,
    response_start_timestamp_ms REAL, 
    brain_freeze_duration_ms REAL, -- Derived: Delay between buzzing in and the first vocal phoneme
    true_acoustic_confidence_score REAL, -- Deterministic via WhisperX logprobs
    disfluency_count INTEGER -- Deterministic count of "ums", "uhs", and vocal stutters
);

CREATE TABLE wagers (
    wager_id TEXT PRIMARY KEY,
    clue_id TEXT REFERENCES clues(clue_id),
    contestant_id TEXT REFERENCES contestants(contestant_id),
    
    -- Game Theory Features
    running_score_at_time INTEGER,
    opponent_1_score INTEGER,
    opponent_2_score INTEGER,
    actual_wager INTEGER,
    
    game_theory_optimal_wager INTEGER, -- The mathematically perfect Kelly Criterion wager
    wager_irrationality_delta INTEGER -- abs(actual_wager - game_theory_optimal_wager)
);
```

## 3. Pydantic Definitions (Extraction Schema)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class BuzzAttempt(BaseModel):
    """
    Captures a single attempt at answering a clue.
    A Clue will have a list of these. If the list is empty, it's a 'Triple Stumper'.
    If the list has multiple entries, it represents 'Rebounds' (someone missed, another rang in).
    """
    attempt_order: int = Field(description="1 for first buzz, 2/3 for rebounds.")
    speaker: str = Field(description="The contestant who rang in.")
    response_given: str = Field(description="The literal text of what they guessed, capturing stutters/ums.")
    is_correct: bool = Field(description="Whether the host ruled it correct.")
    buzz_timestamp_ms: float = Field(description="The exact timestamp when the contestant buzzed in.")
    response_start_timestamp_ms: float = Field(description="The exact timestamp when the contestant began speaking.")
    is_lockout_inferred: bool = Field(description="True if they appeared to jump the gun and suffered the 0.25s penalty.")

class Clue(BaseModel):
    """
    Represents a single square on the board.
    """
    round: Literal["Jeopardy", "Double Jeopardy", "Final Jeopardy", "Tiebreaker"]
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
        description="The amount wagered. Can be an integer or the literal string 'True Daily Double'."
    )
    wagerer_name: Optional[str] = Field(description="Name of the contestant who found the Daily Double.")
    clue_text: str = Field(description="The text read by the host (the answer).")
    correct_response: str = Field(description="The accepted correct response (the question).")
    
    attempts: List[BuzzAttempt] = Field(
        default_factory=list, 
        description="Chronological list of buzz attempts. Empty list for Triple Stumpers."
    )

class ScoreAdjustment(BaseModel):
    """
    Captures moments where the judges intervene mid-game to overturn or correct a previous ruling.
    """
    contestant: str
    points_adjusted: int = Field(description="Positive or negative integer adjustment.")
    reason: str = Field(description="The host's explanation for the score change.")
    effective_after_clue_selection_order: int = Field(
        description="Chronological index of clue selection after which this adjustment strictly applies."
    )

class FinalJeopardyWager(BaseModel):
    contestant: str
    wager: int
    response: str
    is_correct: bool

class FinalJeopardy(BaseModel):
    category: str
    clue_text: str
    wagers_and_responses: List[FinalJeopardyWager]

class Contestant(BaseModel):
    name: str
    podium_position: int = Field(description="1 (left), 2 (center), or 3 (right).")
    occupational_category: str = Field(description="Classified by LLM (e.g., 'Academia', 'STEM', 'Law').")
    is_returning_champion: bool
    description: str = Field(description="Hometown/occupational context from the interview segment.")

class Episode(BaseModel):
    """
    The root schema output by Gemini 3.1 Pro.
    """
    episode_date: str
    host_name: str = Field(description="Name of the host for this episode.")
    is_tournament: bool = Field(description="True if this is a tournament episode.")
    contestants: List[Contestant]
    clues: List[Clue]
    final_jeopardy: FinalJeopardy
    
    score_adjustments: List[ScoreAdjustment]
```

## 4. Key Schema Features
1. **The `attempts` List:** Prevents schema crashes on null answers (Triple Stumpers) and accurately tracks rebounds (where multiple contestants get a chance to answer).
2. **Deterministic Math:** The LLM is explicitly forbidden from calculating math or running scores, preventing probabilistic hallucinations. It ONLY extracts atomic events (e.g., "Contestant A answered correctly for $400"). A strict Python State Machine calculates all scores, eliminating bad data.
3. **`ScoreAdjustment`:** Captures retroactive judge rulings, which happen frequently after commercial breaks and drastically alter the math of the game.
4. **2-Pass Speaker Anchoring Strategy:** To prevent LLM hallucinations, a fast Gemini Flash pass strictly isolates the host's interview segment to generate a rigid `{SPEAKER_XX: "Name"}` mapping dictionary. This locked dictionary is injected into the System Prompt of the massive Stage 5 structured extraction.
5. **Machine Learning Feature Extraction:** Captures critical metrics like acoustic confidence, brain freeze duration, and board-hunting patterns directly into the schema to enable advanced behavioral and physics-based ML modeling.