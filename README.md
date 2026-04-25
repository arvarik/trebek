<div align="center">
  <h1>Trebek 🎙️</h1>
  <p><b>A highly resilient, fault-tolerant data extraction pipeline daemon for transcribing and extracting structured game events from Jeopardy! episodes.</b></p>
  <p>
    <a href="https://github.com/arvarik/trebek/actions/workflows/ci.yml">
      <img alt="CI" src="https://github.com/arvarik/trebek/actions/workflows/ci.yml/badge.svg" />
    </a>
    <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white" />
    <img alt="SQLite" src="https://img.shields.io/badge/database-SQLite%20(WAL)-003B57?logo=sqlite&logoColor=white" />
    <img alt="Pydantic v2" src="https://img.shields.io/badge/pydantic-v2-e92063?logo=pydantic&logoColor=white" />
    <img alt="Google Gemini" src="https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google&logoColor=white" />
    <img alt="WhisperX" src="https://img.shields.io/badge/ASR-WhisperX-green" />
    <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" />
    <img alt="Mypy" src="https://img.shields.io/badge/types-Mypy-blue.svg" />
    <img alt="License" src="https://img.shields.io/badge/license-AGPL_3.0-green" />
  </p>
</div>

---

Trebek is an advanced orchestration system that bridges **local GPU compute** (WhisperX, Pyannote), **Cloud LLMs** (Google Gemini 1.5 Pro/Flash, Vision), and a **deterministic Python state machine** into a single, continuously running pipeline daemon. Its core purpose is to extract highly accurate, chronological, and structurally validated data from raw Jeopardy! video episodes into a normalized relational format designed for **RAG semantic searches** and **game-theoretic analysis**.

The resulting dataset captures not just the questions and answers, but the full *cognitive fingerprint* of each game: true buzzer reaction times, speech disfluency counts, wager irrationality deltas, board control patterns, and semantic lateral distances between clues and responses.

## Table of Contents

- [Why Trebek?](#why-trebek)
- [Core Features](#-core-features)
- [System Architecture](#-system-architecture)
- [Pipeline Stages](#-pipeline-stages)
- [Data Model](#-data-model)
- [ML/AI Integration](#-mlai-integration)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Development](#-development)
- [Project Structure](#-project-structure)
- [Safety Invariants](#-safety-invariants)
- [Design Philosophy](#-design-philosophy)

---

## Why Trebek?

Existing Jeopardy! datasets are typically scraped text archives — static lists of clues and responses with no temporal, behavioral, or strategic context. Trebek fills this gap by processing the *raw video*, producing a dataset that includes:

- **Millisecond-precision buzzer latencies** calculated from cross-referencing visual podium illumination timestamps with acoustic buzz detection.
- **Disfluency tracking** (ums, uhs, stutters) via WhisperX word-level logprobs, not LLM hallucinations.
- **Game-theory optimal wager calculations** compared against actual contestant wagers to quantify irrationality.
- **Semantic lateral distance** between clues and responses, distinguishing wordplay from direct factual recall.
- **Forrest Bounce detection** and board control analysis for strategic game modeling.

The target audience is **ML engineers, data scientists, and researchers** who need a surgically clean, event-sourced dataset of human decision-making under televised pressure for predictive modeling.

---

## ✨ Core Features

### Database-Backed Queueing (True Resumability)
Uses a persistent SQLite `pipeline_state` table to manage jobs across all stages of execution. The daemon can be interrupted at any point — via `SIGINT`, `SIGTERM`, or a crash — and will seamlessly resume exactly where it left off. No data is lost. No re-processing is required.

### VRAM Fragmentation Immunity
Local GPU operations (PyTorch/WhisperX) are sandboxed in a `ProcessPoolExecutor` with `max_tasks_per_child=1`. Worker processes forcefully die after every episode, which defragments 100% of VRAM. This makes the system immune to PyTorch's internal memory fragmentation during multi-day inference runs — a problem `torch.cuda.empty_cache()` alone cannot solve.

### Multi-Pass LLM Architecture
- **Pass 1 (Gemini Flash):** Fast speaker anchoring. Extracts a rigid `{SPEAKER_XX: "Name"}` mapping from the host interview segment to prevent hallucinations in later passes.
- **Pass 2 (Gemini Pro):** Massive structured extraction of clues, buzzes, and wagers into strict JSON. Includes a **Pydantic self-healing retry loop** — if the LLM output fails schema validation, the `ValidationError` is injected back into the prompt for automatic correction (up to 2 retries).
- **Pass 3 (Gemini Pro Vision):** Multimodal augmentation for visual clue reconstruction and exact podium lockout illumination frame detection.

### Deterministic State Machine
A pure Python `TrebekStateMachine` replays extracted atomic game events chronologically to:
- Calculate **perfect running scores** (never trusting LLMs to do arithmetic).
- Resolve **"True Daily Double"** wagers at runtime against current scores.
- Apply **chronologically anchored score adjustments** (judge reversals) at exactly the right moment.
- Track **board control** shifts and detect **Forrest Bounce** patterns.

### Physics Engine (True Buzzer Latency)
Cross-references visual podium illumination timestamps (from Gemini Vision) with WhisperX's acoustic word-level boundaries to calculate true contestant reaction speeds, independent of host cadence variance. Also computes:
- Acoustic confidence scores from raw WhisperX logprobs.
- Deterministic disfluency counts (ums/uhs) from acoustic data, not LLM guesses.
- Semantic lateral distance via cosine distance on text embeddings.

### Actor-Pattern Database Writer
All SQLite writes are routed through a single `DatabaseWriter` actor — an asyncio task owning an internal `asyncio.Queue`. This serializes concurrent write requests, preventing `database is locked` exceptions. Every enqueued operation returns an `asyncio.Future` protected by `asyncio.wait_for()` to prevent silent deadlocks.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    TrebekPipelineOrchestrator                    │
│                      (asyncio event loop)                       │
├─────────┬─────────────┬───────────────┬─────────────────────────┤
│ Ingest  │  Extractor  │  LLM Worker   │  State Machine Worker   │
│ Worker  │   Worker    │               │                         │
│         │             │               │                         │
│ polls   │ dispatches  │ Gemini Flash  │ TrebekStateMachine      │
│ input/  │ to GPU      │ Gemini Pro    │ Score verification      │
│ dir     │ subprocess  │ Self-healing  │ Board control           │
│         │             │ retry loop    │ Wager math              │
└────┬────┴──────┬──────┴───────┬───────┴──────────┬──────────────┘
     │           │              │                  │
     ▼           ▼              ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DatabaseWriter (Actor)                       │
│              asyncio.Queue → sqlite3.Connection                 │
│         PRAGMA foreign_keys=ON | journal_mode=WAL               │
│       PRAGMA busy_timeout=5000 | auto_vacuum=INCREMENTAL        │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SQLite Database                          │
│  pipeline_state │ episodes │ contestants │ clues │ buzz_attempts│
│  wagers │ score_adjustments │ episode_performances              │
└─────────────────────────────────────────────────────────────────┘
```

### Concurrency Model

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **I/O Orchestration** | `asyncio` event loop | Manages state polling, signal handling, and worker coordination |
| **GPU Isolation** | `ProcessPoolExecutor` (`spawn`, `max_tasks_per_child=1`) | Subprocess dies after every task → 100% VRAM reclamation |
| **Write Serialization** | Actor pattern (`asyncio.Queue` + `Future`) | Prevents SQLite `database is locked` errors |
| **CPU Offloading** | `asyncio.to_thread()` | Offloads heavy Pydantic JSON validation off the event loop |
| **IPC Optimization** | Filepath strings over `.json.gz` | Avoids pickling overhead of massive JSON structures across process boundaries |

---

## 📊 Pipeline Stages

The pipeline processes each episode through a rigorous sequence of stages, with the `pipeline_state` table acting as a persistent, crash-safe queue:

| Stage | Name | Status Transition | Engine | Description |
|-------|------|-------------------|--------|-------------|
| 1 | **Ingestion** | → `PENDING` | Filesystem polling | `.mp4` files detected in `input_dir` are registered in `pipeline_state` |
| 2–3 | **GPU Extraction** | `PENDING` → `TRANSCRIBING` → `TRANSCRIPT_READY` | FFmpeg + WhisperX + Pyannote | Audio extraction, Large-v3 float16 transcription, forced alignment diarization. Output: `.json.gz` |
| 4 | **Commercial Filtering** | `TRANSCRIPT_READY` → `CLEANED` | Gemini Flash | Hardware-accelerated advertisement removal while preserving exact word-level timings |
| 5 | **Structured Extraction** | `CLEANED` → `SAVING` | Gemini Flash + Pro | Pass 1: Speaker anchoring. Pass 2: Full game event extraction with Pydantic self-healing |
| 6 | **Multimodal Augmentation** | (inline) | Gemini Pro Vision | Visual clue reconstruction and podium illumination timestamp extraction |
| 7 | **State Verification** | `SAVING` → `VECTORIZING` | `TrebekStateMachine` | Deterministic replay validates score sequences, adjustments, and board control logic |
| 8–9 | **Relational & Semantic Commit** | `VECTORIZING` → `COMPLETED` | `DatabaseWriter` + `sqlite-vec` | Normalized INSERT into relational tables + vector embedding for semantic search |

If any stage fails, the episode status is set to `FAILED` and logged for manual review. The daemon continues processing other episodes.

---

## 🗂️ Data Model

The SQLite schema is designed as a **normalized relational model** optimized for analytical queries:

### Core Tables

```
pipeline_state          The persistent job queue
├── episode_id (PK)
├── status              PENDING → TRANSCRIBING → TRANSCRIPT_READY → CLEANED → SAVING → VECTORIZING → COMPLETED
├── transcript_path     Filepath to .json.gz GPU output
├── created_at
└── updated_at

episodes                High-level episode metadata
├── episode_id (PK)
├── air_date
├── host_name
└── is_tournament

contestants             Unique contestant profiles
├── contestant_id (PK)
├── name
├── occupational_category    LLM-classified (e.g., 'Academia', 'STEM', 'Law')
└── is_returning_champion

episode_performances    Per-episode contestant stats
├── episode_id (FK)
├── contestant_id (FK)
├── podium_position     1 (left), 2 (center), 3 (right)
├── coryat_score        Score without Daily Doubles and Final Jeopardy
├── final_score
└── forrest_bounce_index

clues                   The board matrix with temporal and semantic markers
├── clue_id (PK)
├── episode_id (FK)
├── round               CHECK('Jeopardy', 'Double', 'Final', 'Tiebreaker')
├── category / board_row / board_col / selection_order
├── clue_text / correct_response
├── is_daily_double / daily_double_wager / wagerer_name
├── host_start_timestamp_ms / host_finish_timestamp_ms
├── clue_syllable_count / host_speech_rate_wpm
├── requires_visual_context
├── clue_embedding (BLOB)     Vector for semantic search
├── response_embedding (BLOB)
└── semantic_lateral_distance  Cosine distance: wordplay vs. factual recall

buzz_attempts           Behavioral physics per buzz-in
├── attempt_id (PK)
├── clue_id (FK) / contestant_id (FK)
├── attempt_order       1st buzz, 2nd/3rd for rebounds
├── buzz_timestamp_ms / podium_light_timestamp_ms
├── true_buzzer_latency_ms   Reaction time (visual - acoustic)
├── is_lockout_inferred      0.25s penalty detection
├── response_given / is_correct
├── brain_freeze_duration_ms
├── true_acoustic_confidence_score   From WhisperX logprobs
├── disfluency_count
└── phonetic_similarity_score

wagers                  Game theory analysis
├── wager_id (PK)
├── clue_id (FK) / contestant_id (FK)
├── running_score_at_time / opponent scores
├── actual_wager
├── game_theory_optimal_wager
└── wager_irrationality_delta

score_adjustments       Chronological host/judge corrections
├── adjustment_id (PK)
├── episode_id (FK) / contestant_id (FK)
├── points_adjusted
├── reason
└── effective_after_clue_selection_order
```

### Pydantic Data Contracts

All LLM extraction outputs are validated against strict Pydantic v2 models defined in `src/schemas.py`. Key models include:

| Model | Description |
|-------|-------------|
| `Episode` | Top-level container: contestants, clues, final jeopardy, score adjustments |
| `Clue` | Board position, temporal bounds, Daily Double metadata, buzz attempts |
| `BuzzAttempt` | Per-buzz reaction data: timestamps, lockout inference, response text |
| `Contestant` | Name, podium position, occupation category, champion status |
| `FinalJeopardy` | Category, clue text, per-contestant wagers and responses |
| `ScoreAdjustment` | Chronologically anchored point corrections with reasons |

---

## 🤖 ML/AI Integration

| Provider | Model | Stage | Application |
|----------|-------|-------|-------------|
| **Local GPU** | WhisperX / Pyannote | 2–3 | Large-v3 float16 transcription, forced alignment, speaker diarization |
| **Google** | Gemini 1.5 Flash | 4–5 | Speaker anchoring and commercial filtering (high-speed structured mapping) |
| **Google** | Gemini 1.5 Pro | 5 | Massive structured extraction with Pydantic self-healing retry loop |
| **Google** | Gemini Pro Vision | 6 | Visual clue reconstruction, podium lockout illumination frame detection |
| **Local GPU** | Ollama / Llama-3-8B | 5 (fallback) | Offline structured extraction for environments without Gemini API access |
| **Local/API** | Text Embeddings | 9 | Cosine distance calculation for `semantic_lateral_distance` |

---

## 🛠️ Installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python** | 3.9 or higher |
| **FFmpeg** | Required for audio extraction from video files |
| **NVIDIA GPU** | Minimum 16GB VRAM recommended (optimized for RTX 4060 Ti / 5060 Ti) |
| **CUDA Toolkit** | Required for WhisperX GPU acceleration |
| **SQLite** | 3.35+ (for `RETURNING` clause support in atomic polling) |
| **Gemini API Key** | Required for LLM extraction stages |

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arvarik/trebek.git
   cd trebek
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the package with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install GPU dependencies** (if processing locally):
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install whisperx pyannote.audio
   ```

---

## ⚙️ Configuration

Trebek uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration, automatically loading values from environment variables or a `.env` file in the project root.

Create a `.env` file:

```env
# ─── Core Paths ───
db_path=trebek.db                     # Path to the SQLite database
output_dir=gpu_outputs                # Directory for intermediate pipeline outputs (.json.gz)
input_dir=input_videos                # Directory to poll for new .mp4 files

# ─── API Keys ───
GEMINI_API_KEY=your_api_key_here      # Google Gemini API key

# ─── Logging ───
log_level=INFO                        # DEBUG, INFO, WARNING, ERROR

# ─── GPU Constraints ───
gpu_vram_target_gb=16                 # Target VRAM ceiling (4–24 GB)
whisper_batch_size=16                 # WhisperX batch size (tuned for 16GB VRAM)
whisper_compute_type=float16          # float16 or float32
```

### Configuration Validation

The `Settings` class enforces runtime constraints via Pydantic field validators:

| Setting | Constraint | Default |
|---------|-----------|---------|
| `gpu_vram_target_gb` | Must be between 4 and 24 (inclusive) | `16` |
| `whisper_compute_type` | Must be `float16` or `float32` | `float16` |
| `whisper_batch_size` | Must be greater than 0 | `16` |

Invalid configurations will raise a `ValidationError` at startup, preventing the daemon from running with unsafe GPU parameters.

---

## 🚀 Usage

Trebek is designed to run as a **continuous daemon**. Once started, it will poll the configured `input_dir` for new `.mp4` files and orchestrate the full extraction pipeline automatically.

### Start the Pipeline

```bash
python src/main.py
```

### Process Episodes

1. Place `.mp4` video files into the `input_videos/` directory (or your configured `input_dir`).
2. The ingestion worker will detect new files within 5 seconds and register them as `PENDING`.
3. Each episode flows through the pipeline stages automatically.
4. Monitor progress via structured JSON logs (stdout) or query the `pipeline_state` table directly.

### Graceful Shutdown

Send `SIGINT` (Ctrl+C) or `SIGTERM` to the process. The daemon will:
1. Stop accepting new work.
2. Cancel all running async tasks.
3. Wait for the GPU subprocess to complete its current task.
4. Flush and close the database connection.
5. Log a clean shutdown confirmation.

### Querying Results

After processing, query the normalized SQLite database directly:

```sql
-- Find the fastest buzzer in a specific episode
SELECT c.name, ba.true_buzzer_latency_ms
FROM buzz_attempts ba
JOIN contestants c ON ba.contestant_id = c.contestant_id
WHERE ba.is_correct = 1
ORDER BY ba.true_buzzer_latency_ms ASC
LIMIT 5;

-- Identify irrational Daily Double wagers
SELECT c.name, w.actual_wager, w.game_theory_optimal_wager, w.wager_irrationality_delta
FROM wagers w
JOIN contestants c ON w.contestant_id = c.contestant_id
WHERE ABS(w.wager_irrationality_delta) > 500
ORDER BY ABS(w.wager_irrationality_delta) DESC;

-- Semantic search for wordplay-heavy categories
SELECT category, AVG(semantic_lateral_distance) as avg_distance
FROM clues
GROUP BY category
ORDER BY avg_distance DESC
LIMIT 10;
```

---

## 🧪 Development

### Toolchain

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **pytest** | Test runner (with `pytest-asyncio` for async tests) | `pyproject.toml` |
| **ruff** | Linter and import sorter | Line length: 120, target: Python 3.9 |
| **mypy** | Static type checker | Strict mode enabled |

### Commands

```bash
# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run the linter
ruff check .

# Run the type checker
mypy src/
```

### Test Coverage

The test suite validates critical system contracts:

| Test Module | Coverage Area |
|-------------|---------------|
| `test_state_machine.py` | Score calculation, board control, chronological adjustments, True Daily Double resolution |
| `test_core_database.py` | Actor-pattern write execution, atomic polling (`RETURNING` clause) |
| `test_schema_integrity.py` | Foreign key enforcement, CHECK constraints, NOT NULL constraints |
| `test_config_validation.py` | GPU VRAM bounds, compute type validation, batch size validation |
| `test_schemas.py` | Pydantic model constraints: podium positions, Daily Double wager union types |
| `test_gpu_orchestrator.py` | Subprocess lifecycle, `.json.gz` output generation, mock binary integration |
| `test_llm_pipeline.py` | Speaker anchoring Pass 1 with mocked Gemini client |

---

## 📁 Project Structure

```
trebek/
├── src/
│   ├── main.py               # Pipeline orchestrator daemon (asyncio event loop, workers, signal handling)
│   ├── config.py              # Pydantic Settings with GPU constraint validators
│   ├── schemas.py             # Pydantic v2 data contracts (Episode, Clue, BuzzAttempt, etc.)
│   ├── schema.sql             # SQLite DDL: 8 tables with foreign keys, CHECK constraints, PRAGMAs
│   ├── core_database.py       # Actor-pattern DatabaseWriter with deadlock protection
│   ├── gpu_orchestrator.py    # ProcessPoolExecutor with spawn context and SIGKILL safety
│   ├── llm_pipeline.py        # Multi-pass Gemini extraction with self-healing retry loop
│   ├── state_machine.py       # Deterministic game state replay (scores, adjustments, board control)
│   └── physics_engine.py      # Buzzer latency, acoustic metrics, semantic distance, Vision client
├── tests/
│   ├── conftest.py            # Shared fixtures (in-memory SQLite with schema)
│   ├── mock_bin/              # Mock ffmpeg/whisperx binaries for GPU orchestrator tests
│   ├── test_state_machine.py
│   ├── test_core_database.py
│   ├── test_schema_integrity.py
│   ├── test_config_validation.py
│   ├── test_schemas.py
│   ├── test_gpu_orchestrator.py
│   └── test_llm_pipeline.py
├── docs/                      # Design documents, plans, explorations, and archived specs
├── .agent/                    # Agent lifecycle metadata (architecture, philosophy, status, style)
├── pyproject.toml             # Build system, dependencies, tool configuration
├── .gitignore
└── README.md
```

---

## 🔒 Safety Invariants

These are **non-negotiable** constraints that must be preserved across all contributions:

1. **GPU Subprocess Isolation.** All PyTorch/WhisperX operations **must** execute inside a `ProcessPoolExecutor` with `max_tasks_per_child=1`. Workers must die after every task to guarantee VRAM defragmentation. Never use `torch.cuda.empty_cache()` as a substitute.

2. **Database Write Serialization.** All SQLite write operations **must** be routed through the `DatabaseWriter` actor queue. Direct `conn.execute()` calls from workers will cause `database is locked` errors under concurrent load.

3. **Event Loop Protection.** Heavy CPU-bound operations (specifically `Episode.model_validate_json`) **must** be offloaded to a background thread via `asyncio.to_thread()`. Blocking the main event loop will trigger watchdog heartbeat timeouts.

4. **IPC Boundary Hygiene.** Never pass large JSON structures across process boundaries (IPC pickling). Write data to disk as compressed `.json.gz` and pass the filepath string instead.

5. **LLM Fact Extraction Only.** LLMs **must never** perform running score math or wager calculations. They extract facts; the `TrebekStateMachine` executes all arithmetic deterministically.

6. **Chronological Score Adjustments.** Score adjustments **must** be applied at exactly the correct `selection_order` index — not before, not after.

7. **Persistent Queue Only.** The SQLite `pipeline_state` table **must** act as the inter-stage queue. Never use `asyncio.Queue` for passing work between pipeline stages.

---

## 💡 Design Philosophy

### Database-Driven State Machine over Memory
True resumability and crash immunity are paramount. Zero data loss during multi-day inference runs requires database-backed queueing, not fragile in-memory queues. The pipeline can be killed at any point and will resume cleanly.

### Deterministic Math over LLM Approximations
LLMs are hallucination-prone when performing arithmetic. They extract pure facts from transcripts; deterministic Python state machines execute the score tracking, True Daily Double resolution, and game-theory optimal wager calculations.

### Hardware Isolation is Safety
VRAM fragmentation is inevitable in long-running PyTorch processes. Forceful memory reclamation via ephemeral subprocesses (`max_tasks_per_child=1`) guarantees stability over multi-day batch runs processing hundreds of episodes.

### What Trebek Is NOT
- **Not a real-time application.** This is a batch-processing, heavy-compute daemon pipeline, not an interactive or real-time streaming service.
- **Not an API server.** It operates via filesystem polling and SQLite state management, not over HTTP endpoints.
- **Not a keyword matcher.** The dataset relies on vectorized embeddings (`sqlite-vec`) for semantic evaluation of clues, isolating wordplay from direct factual recall.

---

<div align="center">
  <sub>Built for ML researchers who believe the best datasets are the ones you extract yourself.</sub>
</div>
