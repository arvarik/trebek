<div align="center">
  <h1>Trebek 🎙️</h1>
  <p><b>A highly resilient, fault-tolerant data extraction pipeline for transcribing and extracting structured game events from Jeopardy! episodes.</b></p>
  <p>
    <a href="https://github.com/arvarik/trebek/actions/workflows/ci.yml">
      <img alt="CI" src="https://github.com/arvarik/trebek/actions/workflows/ci.yml/badge.svg" />
    </a>
    <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" />
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

Trebek is an advanced orchestration system that bridges **local GPU compute** (WhisperX, Pyannote), **Cloud LLMs** (Google Gemini 3.1 Pro, Gemini 3.1 Flash-Lite), and a **deterministic Python state machine** into a single, continuously running pipeline daemon. It extracts highly accurate, chronological, and structurally validated data from raw Jeopardy! video episodes into a normalized relational format designed for **RAG semantic searches** and **game-theoretic analysis**.

The resulting dataset captures not just the questions and answers, but the full *cognitive fingerprint* of each game: true buzzer reaction times, speech disfluency counts, wager irrationality deltas, board control patterns, and semantic lateral distances between clues and responses.

## Table of Contents

- [Why Trebek?](#why-trebek)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Pipeline Stages](#pipeline-stages)
- [Data Model](#data-model)
- [ML/AI Integration](#mlai-integration)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Project Structure](#project-structure)
- [Safety Invariants](#safety-invariants)
- [Design Philosophy](#design-philosophy)

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

### Warm Worker Architecture & VRAM Management
Local GPU operations (PyTorch/WhisperX) utilize a "Warm Worker" architecture where the model weights remain resident in VRAM across tasks, eliminating cold-start latency. Memory fragmentation is actively managed via explicit tensor deletion, garbage collection (`gc.collect()`), and `torch.cuda.empty_cache()`. If a CUDA Out-Of-Memory (OOM) error occurs, the orchestrator seamlessly catches it and restarts the worker pool to recover. Shutdowns are handled gracefully via `SIGTERM` signals using `psutil`, falling back to `SIGKILL` only for stubborn zombies.

### Multi-Pass LLM Architecture
- **Pass 1 (Gemini 3.1 Flash-Lite):** Fast speaker anchoring. Extracts a rigid `{SPEAKER_XX: "Name"}` mapping from the host interview segment to prevent hallucinations in later passes.
- **Pass 2 (Gemini 3.1 Pro):** Massive structured extraction of clues, buzzes, and wagers into strict JSON. Includes a **Pydantic self-healing retry loop** — if the LLM output fails schema validation, the `ValidationError` is injected back into the prompt for automatic correction (up to 2 retries).
- **Pass 3 (Gemini 3.1 Pro):** Multimodal augmentation for visual clue reconstruction and exact podium lockout illumination frame detection.

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
All SQLite writes are routed through a single `DatabaseWriter` actor — an asyncio task owning an internal `asyncio.Queue`. This serializes concurrent write requests, preventing `database is locked` exceptions. It supports high-throughput atomic transactions via `execute_transaction`, allowing multiple related queries to be batched efficiently, eliminating micro-transaction disk stalls. Every enqueued operation returns an `asyncio.Future` protected by `asyncio.wait_for()` to prevent silent deadlocks.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────┐
│        TrebekPipelineOrchestrator             │
│            (asyncio event loop)               │
├──────────┬──────────┬──────────┬──────────────┤
│ Ingest   │ GPU      │ LLM     │ State Machine│
│ Worker   │ Worker   │ Worker  │ Worker       │
│          │          │         │              │
│ polls    │ FFmpeg + │ Flash-  │ Score verify │
│ input/   │ WhisperX │ Lite +  │ Board ctrl   │
│ dir      │ Pyannote │ Pro     │ Wager math   │
└────┬─────┴────┬─────┴────┬────┴───────┬──────┘
     │          │          │            │
     ▼          ▼          ▼            ▼
┌──────────────────────────────────────────────┐
│         DatabaseWriter (Actor)                │
│      asyncio.Queue → sqlite3.Connection       │
│  journal_mode=WAL | foreign_keys=ON           │
│  busy_timeout=5000 | auto_vacuum=INCREMENTAL  │
└──────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│            SQLite Database                    │
│  pipeline_state │ episodes │ contestants      │
│  clues │ buzz_attempts │ wagers               │
│  score_adjustments │ episode_performances     │
│  job_telemetry                                │
└──────────────────────────────────────────────┘
```

### Concurrency Model

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **I/O Orchestration** | `asyncio` Event-Driven | `asyncio.Event` objects replace polling latency, orchestrating stages efficiently |
| **GPU Isolation** | `ProcessPoolExecutor` (`spawn`) | Warm Worker architecture with active GC and automatic OOM-recovery pool restarts |
| **Write Serialization** | Actor pattern (`Queue` + `Future`) | Prevents SQLite `database is locked` errors |
| **CPU Offloading** | `asyncio.to_thread()` | Offloads Pydantic JSON validation off the event loop |
| **IPC Optimization** | Filepath strings over `.json.gz` | Avoids pickling large JSON across process boundaries |

---

## 📊 Pipeline Stages

The pipeline processes each episode through a rigorous sequence of stages, with the `pipeline_state` table acting as a persistent, crash-safe queue:

| Stage | Name | Engine | Description |
|-------|------|--------|-------------|
| 1 | **Ingestion** | Filesystem polling | New video files registered as `PENDING` |
| 2–3 | **GPU Extraction** | FFmpeg + WhisperX | Audio extraction, transcription, diarization |
| 4 | **Commercial Filtering** | Gemini Flash-Lite | Ad removal preserving word-level timings |
| 5 | **Structured Extraction** | Flash-Lite + Pro | Speaker anchoring → full event extraction |
| 6 | **Multimodal Augmentation** | Gemini Pro | Visual clue + podium illumination detection |
| 7 | **State Verification** | `TrebekStateMachine` | Deterministic score/adjustment validation |
| 8–9 | **Relational & Semantic Commit** | `DatabaseWriter` | Normalized INSERT + vector embeddings |

If any stage fails, the episode status is set to `FAILED` and logged for manual review. The daemon continues processing other episodes.

---

## 🗂️ Data Model

The SQLite schema is designed as a **normalized relational model** optimized for analytical queries:

### Core Tables

```
pipeline_state
├── episode_id (PK)
├── status           PENDING → TRANSCRIBING →
│                    TRANSCRIPT_READY → CLEANED →
│                    SAVING → VECTORIZING → COMPLETED
├── transcript_path  Filepath to .json.gz output
├── created_at
└── updated_at

episodes
├── episode_id (PK)
├── air_date
├── host_name
└── is_tournament

contestants
├── contestant_id (PK)
├── name
├── occupational_category
└── is_returning_champion

episode_performances
├── episode_id (FK)
├── contestant_id (FK)
├── podium_position   1 (left), 2 (center), 3 (right)
├── coryat_score
├── final_score
└── forrest_bounce_index

clues
├── clue_id (PK)
├── episode_id (FK)
├── round              Jeopardy / Double / Final
├── category / board_row / board_col
├── selection_order
├── clue_text / correct_response
├── is_daily_double / daily_double_wager
├── host_start_timestamp_ms
├── host_finish_timestamp_ms
├── clue_syllable_count
├── requires_visual_context
├── clue_embedding (BLOB)
├── response_embedding (BLOB)
└── semantic_lateral_distance

buzz_attempts
├── attempt_id (PK)
├── clue_id (FK) / contestant_id (FK)
├── attempt_order
├── buzz_timestamp_ms
├── podium_light_timestamp_ms
├── true_buzzer_latency_ms
├── is_lockout_inferred
├── response_given / is_correct
├── brain_freeze_duration_ms
├── true_acoustic_confidence_score
├── disfluency_count
└── phonetic_similarity_score

wagers
├── wager_id (PK)
├── clue_id (FK) / contestant_id (FK)
├── running_score_at_time
├── actual_wager
├── game_theory_optimal_wager
└── wager_irrationality_delta

score_adjustments
├── adjustment_id (PK)
├── episode_id (FK) / contestant_id (FK)
├── points_adjusted
├── reason
└── effective_after_clue_selection_order

job_telemetry
├── episode_id (FK)
├── peak_vram_mb / avg_gpu_utilization_pct
├── gemini_total_input_tokens
├── gemini_total_output_tokens
├── gemini_total_cached_tokens
├── gemini_total_cost_usd
├── stage_ingestion_ms
├── stage_gpu_extraction_ms
├── stage_structured_extraction_ms
├── stage_vectorization_ms
├── gemini_api_latency_ms
└── pydantic_retry_count
```

### Pydantic Data Contracts

All LLM extraction outputs are validated against strict Pydantic v2 models defined in `trebek/schemas.py`. Key models include:

| Model | Description |
|-------|-------------|
| `Episode` | Top-level container: contestants, clues, final jeopardy, score adjustments |
| `Clue` | Board position, temporal bounds, Daily Double metadata, buzz attempts |
| `BuzzAttempt` | Per-buzz reaction data: timestamps, lockout inference, response text |
| `Contestant` | Name, podium position, occupation category, champion status |
| `FinalJeopardy` | Category, clue text, per-contestant wagers and responses |
| `ScoreAdjustment` | Chronologically anchored point corrections with reasons |
| `JobTelemetry` | Hardware signatures, token usage, latency, and cost tracking |

---

## 🤖 ML/AI Integration

| Provider | Model | Stage | Application |
|----------|-------|-------|-------------|
| **Local GPU** | WhisperX / Pyannote | 2–3 | Large-v3 float16 transcription, diarization |
| **Google** | Gemini 3.1 Flash-Lite | 4–5 | Speaker anchoring, commercial filtering |
| **Google** | Gemini 3.1 Pro | 5 | Structured extraction + Pydantic self-healing |
| **Google** | Gemini 3.1 Pro | 6 | Visual clue + podium illumination detection |
| **Local/API** | Text Embeddings | 9 | Cosine distance for `semantic_lateral_distance` |

---

## 🛠️ Installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python** | 3.11 or higher |
| **FFmpeg** | Required for audio extraction from video files |
| **NVIDIA GPU** | 16GB VRAM recommended (RTX 4060 Ti / 5060 Ti) |
| **CUDA Toolkit** | Required for WhisperX GPU acceleration |
| **SQLite** | 3.35+ (for `RETURNING` clause support) |
| **Gemini API Key** | Required for LLM extraction stages |

### Quick Start

```bash
# 1. Install the package
pip install trebek

# 2. Create your config
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
# Get a free key at https://aistudio.google.com/apikey

# 3. Run the pipeline
trebek
```

### GPU Dependencies (Optional)

For native GPU processing without Docker:

```bash
pip install torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
pip install whisperx pyannote.audio
```

If you prefer not to install these heavy dependencies, use the built-in Docker wrapper instead (see below).

### 🐳 Docker Hybrid Execution (Recommended)

To completely bypass complex PyTorch and CUDA dependency issues on your host, Trebek includes a seamless Docker orchestrator.

**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host.

**Usage:**
Simply append the `--docker` flag to any `trebek` command. The CLI will automatically spin up the official GPU-enabled container, mapping your working directory and `.env` variables seamlessly:

```bash
trebek --docker
trebek --docker --once --input-dir ./videos
```

> **⚠️ WARNING — SQLite WAL Mode & Network Drives**
> Trebek uses SQLite Write-Ahead Logging (WAL) which requires strict POSIX advisory locking. Your `trebek.db` volume **must** be mounted to a local disk (ext4, NTFS, APFS). Mapping it to a network share (NFS, SMB, CIFS) will result in database corruption.

---

## ⚙️ Configuration

Trebek uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration, automatically loading values from environment variables or a `.env` file in the project root.

Create a `.env` file:

```env
# ─── Core Paths ───
db_path=trebek.db
output_dir=gpu_outputs
input_dir=input_videos

# ─── API Keys ───
# Get your key at https://aistudio.google.com/apikey
GEMINI_API_KEY=your_api_key_here

# ─── Logging ───
log_level=INFO

# ─── GPU Constraints ───
gpu_vram_target_gb=16
whisper_batch_size=8
whisper_compute_type=float16
```

### Configuration Validation

The `Settings` class enforces runtime constraints via Pydantic field validators:

| Setting | Constraint | Default |
|---------|-----------|---------|
| `gpu_vram_target_gb` | Between 4 and 24 (inclusive) | `16` |
| `whisper_compute_type` | `float16` or `float32` | `float16` |
| `whisper_batch_size` | Must be greater than 0 | `8` |

Invalid configurations will raise a `ValidationError` at startup, preventing the daemon from running with unsafe GPU parameters.

---

## 🚀 Usage

Trebek is designed to run as a **continuous daemon**. Once started, it will **recursively** scan the configured `input_dir` (and all subdirectories) for video files and orchestrate the full extraction pipeline automatically. Trebek supports 12 video formats natively: MP4, TS, MKV, AVI, MOV, WebM, MPG, MPEG, FLV, WMV, M2TS, and VOB.

### Start the Pipeline

```bash
# Point at a media library with nested folders
trebek --input-dir /path/to/TV/Jeopardy

# Docker mode (recommended)
trebek --docker --input-dir /path/to/TV/Jeopardy

# Process current queue then exit
trebek --once

# Preview discovered files without processing
trebek --dry-run

# View database analytics dashboard
trebek --stats
```

### Process Episodes

1. Point `--input-dir` at any directory (nested season folders work automatically).
2. The ingestion worker recursively discovers all video files and registers them as `PENDING`.
3. Each episode flows through the pipeline stages automatically.
4. Monitor progress via the Rich console output, or run `trebek --stats` to view aggregate metrics.

### Graceful Shutdown

Send `SIGINT` (Ctrl+C) or `SIGTERM` to the process. The daemon will:
1. Stop accepting new work.
2. Cancel all running async tasks.
3. Wait for the GPU subprocess to complete its current task.
4. Flush and close the database connection.
5. Render a final session summary with telemetry.

### Querying Results

After processing, query the SQLite database directly:

```sql
-- Find the fastest buzzers
SELECT c.name, ba.true_buzzer_latency_ms
FROM buzz_attempts ba
JOIN contestants c
  ON ba.contestant_id = c.contestant_id
WHERE ba.is_correct = 1
ORDER BY ba.true_buzzer_latency_ms ASC
LIMIT 5;

-- Identify irrational Daily Double wagers
SELECT c.name,
       w.actual_wager,
       w.game_theory_optimal_wager,
       w.wager_irrationality_delta
FROM wagers w
JOIN contestants c
  ON w.contestant_id = c.contestant_id
WHERE ABS(w.wager_irrationality_delta) > 500
ORDER BY ABS(w.wager_irrationality_delta) DESC;

-- Wordplay-heavy categories
SELECT category,
       AVG(semantic_lateral_distance) AS avg_dist
FROM clues
GROUP BY category
ORDER BY avg_dist DESC
LIMIT 10;
```

---

## 🧪 Development

### Toolchain

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **pytest** | Test runner (`pytest-asyncio` for async) | `pyproject.toml` |
| **ruff** | Linter + formatter | Line length 120, target py311 |
| **mypy** | Static type checker | Strict mode enabled |
| **pre-commit** | Git hook enforcement | `.pre-commit-config.yaml` |

### Commands

```bash
# Run the full quality gate
make all

# Individual commands
make test       # pytest with coverage
make lint       # ruff check
make typecheck  # mypy
make format     # ruff auto-format

# Or directly:
pytest tests/ -v
ruff check .
mypy trebek/
```

### Test Coverage

The test suite validates critical system contracts:

| Test Module | Coverage Area |
|-------------|---------------|
| `test_state_machine.py` | Score calculation, board control, chronological adjustments |
| `test_core_database.py` | Actor-pattern write execution, atomic polling |
| `test_schema_integrity.py` | Foreign key, CHECK, and NOT NULL constraints |
| `test_config_validation.py` | GPU VRAM bounds, compute type, batch size validation |
| `test_schemas.py` | Pydantic model constraints, podium positions, wager types |
| `test_gpu_orchestrator.py` | Subprocess lifecycle, `.json.gz` output, mock binaries |
| `test_llm_pipeline.py` | Speaker anchoring Pass 1 with mocked Gemini client |
| `test_job_telemetry.py` | Telemetry schema, validation rules, upsert logic |

---

## 📁 Project Structure

```
trebek/
├── trebek/
│   ├── cli.py              # CLI parser + Docker orchestration
│   ├── main.py             # Pipeline entrypoint wrapper
│   ├── config.py           # Pydantic Settings + validators
│   ├── schemas.py          # Pydantic v2 data contracts
│   ├── schema.sql          # SQLite DDL (9 tables)
│   ├── state_machine.py    # Deterministic game state replay
│   ├── physics_engine.py   # Buzzer latency + semantic distance
│   ├── database/           # Database writer and operations (Actor-pattern)
│   ├── gpu/                # ProcessPoolExecutor + VRAM mgmt workers
│   ├── llm/                # Multi-pass Gemini extraction pipeline
│   ├── pipeline/           # Async pipeline orchestrator and workers
│   └── ui/                 # Rich CLI dashboard and rendering components
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── mock_bin/            # Mock ffmpeg/whisperx binaries
│   ├── test_state_machine.py
│   ├── test_core_database.py
│   ├── test_schema_integrity.py
│   ├── test_config_validation.py
│   ├── test_schemas.py
│   ├── test_gpu_orchestrator.py
│   ├── test_llm_pipeline.py
│   └── test_job_telemetry.py
├── docs/                    # Design documents and plans
├── Makefile                 # Developer shortcuts
├── pyproject.toml           # Build system + tool config
├── .pre-commit-config.yaml  # Git hook enforcement
├── .env.example             # Template configuration
├── .gitignore
└── README.md
```

---

## 🔒 Safety Invariants

These are **non-negotiable** constraints that must be preserved across all contributions:

1. **GPU Warm Worker Architecture.** PyTorch/WhisperX operations execute inside a `ProcessPoolExecutor`. Workers maintain model weights in VRAM for speed but must perform rigorous explicit memory management (`del`, `gc.collect()`, `torch.cuda.empty_cache()`) per task. The pool must automatically catch `MemoryError` (CUDA OOM) and restart itself cleanly to guarantee stability over multi-day inference runs.

2. **Database Write Serialization.** All SQLite write operations **must** be routed through the `DatabaseWriter` actor queue. Direct `conn.execute()` calls from workers will cause `database is locked` errors under concurrent load. Multi-statement commits should utilize atomic transactions via `execute_transaction` to avoid I/O bottlenecks.

3. **Event Loop Protection.** Heavy CPU-bound operations (specifically `Episode.model_validate_json`) **must** be offloaded to a background thread via `asyncio.to_thread()`. Stage coordination **must** use `asyncio.Event` triggers rather than tight `asyncio.sleep` polling loops.

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

### Hardware Isolation & Active Management
VRAM fragmentation is inevitable in long-running PyTorch processes. While previous iterations relied on ephemeral subprocesses, the current "Warm Worker" paradigm achieves higher throughput by managing memory explicitly and gracefully recovering from OOM states by restarting the pool when necessary.

### What Trebek Is NOT
- **Not a real-time application.** This is a batch-processing daemon pipeline, not an interactive or real-time streaming service.
- **Not an API server.** It operates via filesystem polling and SQLite state management, not over HTTP endpoints.
- **Not a keyword matcher.** The dataset relies on vectorized embeddings (`sqlite-vec`) for semantic evaluation of clues, isolating wordplay from direct factual recall.

---

<div align="center">
  <sub>Built for ML researchers who believe the best datasets are the ones you extract yourself.</sub>
</div>
