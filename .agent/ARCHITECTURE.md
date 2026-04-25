## 0. Project Topology
`[backend, ml-ai]`

## 1. Tech Stack
- **Language**: Python 3.9+
- **Core Frameworks**: `pydantic>=2.0.0`, `pydantic-settings>=2.0.0`
- **Logging**: `structlog>=23.1.0`
- **Build System**: `setuptools` (`build-backend = "setuptools.build_meta"`)
- **Testing/Linting**: `pytest>=7.0.0`, `pytest-asyncio`, `ruff`, `mypy>=1.4.0`

## 2. System Architecture & Data Flow
Trebek is a highly resilient, fault-tolerant data extraction pipeline connecting local GPU compute, Cloud LLMs, and a deterministic State Machine. The core philosophy is **Database-Backed Queueing (True Resumability)**.

**Pipeline Stages:**
1. **Stage 1 (Ingestion):** File registered in `pipeline_state` with status `PENDING`.
2. **Stage 2 & 3 (GPU Extraction):** Dispatched to `ProcessPoolExecutor`. FFmpeg extracts audio in-memory, WhisperX/Pyannote perform transcription/diarization. Result is written to a local `.json.gz` file. Filepath is returned via IPC. Database status updated to `TRANSCRIPT_READY`.
3. **Stage 4 (Commercial Filtering):** Gemini 1.5 Flash outputs bounding boxes to slice commercials while preserving word-level timestamps. Status: `CLEANED`.
4. **Stage 5 (Structured Extraction):**
   - **Pass 1:** Gemini 1.5 Flash generates a rigid `{SPEAKER_XX: "Name"}` mapping.
   - **Pass 2:** Gemini 1.5 Pro performs massive extraction of clues, buzzes, wagers into strict JSON. Includes a self-healing Pydantic loop.
5. **Stage 6 (Multimodal Augmentation):** If `requires_visual_context` is true, Gemini Pro Vision analyzes frames to reconstruct clues and extract the podium illumination timestamp for True Buzzer Latency. Status: `SAVING`.
6. **Stage 7 (State Machine):** Deterministic Python `TrebekStateMachine` replays atomic events chronologically to calculate running scores perfectly, enforce score adjustments, and execute Nash Equilibrium wager math.
7. **Stage 8 & 9 (Relational & Semantics):** `DatabaseWriter` executes relational `INSERT`s into `jeopardy_data`. Vectors for Clues/Responses are embedded into `sqlite-vec` virtual tables for RAG semantic searches. Status: `COMPLETED`.

## 3. Concurrency Model & Hardware Management
- **Asyncio Orchestrator**: Manages I/O bound state polling.
- **ProcessPoolExecutor (GPU Isolation)**: Uses `mp_context="spawn"`, `max_workers=1`, and `max_tasks_per_child=1`. Guaranteeing worker death forcefully defragments 100% of VRAM, making the system immune to PyTorch memory fragmentation.
- **Actor Pattern for SQLite**: A single `DatabaseWriter` asyncio task owns an internal `asyncio.Queue` for writes, serializing concurrent requests to prevent `database is locked` errors.
- **Thread Offloading**: Heavy Pydantic JSON validation is executed via `asyncio.to_thread()` to protect the main event loop GIL.
- **IPC Optimization**: Only lightweight strings (e.g. `video_filepath`) are passed across process boundaries. Massive JSON arrays are read/written to disk to avoid pickling overhead.

## 4. Database & State (SQLite)
- **Engine**: SQLite 3.35+
- **Pragmas**: `journal_mode=WAL; busy_timeout=5000; auto_vacuum=INCREMENTAL;`
- **Background Task**: An asyncio task runs `PRAGMA incremental_vacuum;` every 5 minutes to prevent bloat.
- **Core Schema Tables**: 
  - `pipeline_state`: The core queue mechanism.
  - `episodes`, `contestants`, `episode_performances`: High-level anchors (including Coryat Score and Forrest Bounce Index).
  - `clues`: The board matrix and linguistic/temporal markers. Foreign keys link `clues` to `episodes`.
  - `buzz_attempts`: The behavioral physics (brain freeze durations, acoustic confidence, lockout inferences). Foreign keys link to `clues` and `contestants`.
  - `wagers`: Game-theory math (irrationality deltas). Foreign keys link to `clues` and `contestants`.
  - `score_adjustments`: Chronological corrections. Foreign keys link to `episodes` and `contestants`.
  - `job_telemetry`: Granular operational physics. Captures millisecond-precision stage latencies, GPU resource signatures, Gemini token usage (with caching metrics), and self-healing retry counts. Foreign keys link to `pipeline_state`.
  *(No cascading deletes are currently defined; cleanup is manual.)*

## 5. ML/AI Integration Ledger
| Provider | Model | Application | Capabilities |
|----------|-------|-------------|--------------|
| Google | `gemini-1.5-flash` | Speaker Anchoring & Commercial Filtering | High-speed structured mapping. Leverages Context Caching logic optimizations for repeated large transcript processing. |
| Google | `gemini-1.5-pro` | Stage 5 Structured Extraction | Massive Pydantic parsing with strict instructions. Leverages Context Caching to minimize token costs. |
| Google | `Gemini Pro Vision` | Stage 6 Multimodal Augmentation | Reconstruct visual clues and detect exact podium lockout illumination frames. |
| Local GPU | `WhisperX / Pyannote` | Stage 3 & 4 Audio Processing & Filtering | Large-v3 float16 transcription, forced alignment, diarization, and hardware-accelerated commercial pre-filtering. |
| Local GPU | `Ollama / Llama-3-8B` | Stage 5 Local Fallback | Offline structured extraction, leveraging idle 16GB VRAM after Stage 3 completion. |
| Local/API | Text Embedding | Stage 9 Semantic Distance | Calculation of Cosine Distance for `semantic_lateral_distance`. |

## 6. Environment Variables (`.env.example`)
```env
# Path to the SQLite database
db_path=trebek.db

# Directory to store intermediate pipeline outputs (.json.gz)
output_dir=gpu_outputs

# GCP / Gemini API Key
gemini_api_key=

# Logging level
log_level=INFO
```

## 7. Local Development Commands
- **Install for Dev**: `pip install -e .[dev]`
- **Run Tests**: `pytest`
- **Run Linter**: `ruff check .`
- **Run Typechecker**: `mypy src/`

## 8. Safety Invariants & Constraints
- **MUST** isolate GPU operations inside a `ProcessPoolExecutor` with `max_tasks_per_child=1`. 
- **MUST** pass an `asyncio.Future` in the `DatabaseWriter` payload and `await` it using `asyncio.wait_for()` to prevent false resumability and silent deadlocks.
- **MUST** offload `Episode.model_validate_json` to a background thread to prevent `watchdog` heartbeat timeouts.
- **MUST** perform score adjustments logically aligned to the `selection_order`.
- **MUST** explicitly implement Gemini Context Caching optimization for recurring high-token payloads (e.g., transcripts) to minimize API cost and latency, as monitored by `job_telemetry`.
- **MUST** surface operational telemetry via a terminal-native Rich/Textual CLI dashboard, strictly avoiding web-based HTTP frontends.