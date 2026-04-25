<div align="center">
  <h1>Trebek 🎙️</h1>
  <p><b>A highly resilient, fault-tolerant data extraction pipeline daemon for transcribing and extracting structured game events from Jeopardy episodes.</b></p>
</div>

---

Trebek is an advanced orchestration system that bridges local GPU compute (WhisperX, Pyannote), Cloud LLMs (Google Gemini 1.5 Pro/Flash, Vision), and a deterministic Python state machine. Its core purpose is to extract highly accurate, chronological, and structurally validated data from raw Jeopardy video episodes into a normalized relational format designed for RAG semantic searches and game-theoretic analysis.

## ✨ Core Features

*   **Database-Backed Queueing (True Resumability)**: Uses a persistent SQLite `pipeline_state` mechanism to manage jobs across 9 stages of execution. Interruptions are cleanly caught, and the daemon will seamlessly resume where it left off.
*   **VRAM Fragmentation Immunity**: Local GPU operations (PyTorch/WhisperX) are sandboxed in a `ProcessPoolExecutor` with `max_tasks_per_child=1`. Workers forcefully die after every episode, defragmenting 100% of the VRAM.
*   **Multi-Pass LLM Architecture**: 
    *   **Pass 1 (Flash)**: Fast speaker anchoring to prevent hallucinations.
    *   **Pass 2 (Pro)**: Massive structured extraction with a built-in Pydantic self-healing retry loop.
    *   **Pass 3 (Vision)**: Multimodal augmentation for exact contestant podium lockout logic and visual clue reconstruction.
*   **Deterministic State Machine**: Replays the extracted atomic game events chronologically to calculate perfect running scores, track Forrest Bounces, and analyze Nash Equilibrium wagers.
*   **Physics Engine (True Buzzer Latency)**: Cross-references visual podium illumination timestamps with WhisperX's acoustic bounds to calculate true contestant reaction speeds.

## 🏗️ System Architecture & Data Flow

Trebek processes video through a rigorous 9-stage data lifecycle:

1.  **Ingestion:** Incoming `.mp4` files are registered in the state queue as `PENDING`.
2.  **GPU Extraction:** FFmpeg executes in-memory audio extraction. WhisperX and Pyannote execute float16 transcription and forced alignment diarization.
3.  **Commercial Filtering:** Hardware-accelerated removal of advertisements while preserving exact word-level timings.
4.  **Structured Extraction (LLM):** Gemini models execute highly constrained schema mapping and data pulling.
5.  **Multimodal Augmentation:** Vision models reconstruct non-text clues (e.g., image clues).
6.  **State Verification:** The `TrebekStateMachine` validates the sequence logic (daily doubles, score adjustments).
7.  **Relational Database Commit:** Outputs are persisted securely into the `sqlite3` database via an Actor-pattern `DatabaseWriter` (preventing lock contention).
8.  **Vector Embeddings:** Clues and responses are embedded via `sqlite-vec` virtual tables for semantic search operations.

## 🛠️ Installation

**Prerequisites:**
- Python 3.9+
- A local installation of `ffmpeg` (required for audio extraction)
- An NVIDIA GPU (Minimum 16GB VRAM recommended for Large-v3 processing)

**Setup:**

1. Clone the repository:
   ```bash
   git clone https://github.com/arvarik/trebek.git
   cd trebek
   ```

2. Install the package (with development tools):
   ```bash
   pip install -e .[dev]
   ```

3. Configure your environment variables. Create a `.env` file in the root directory:
   ```env
   # Path to the SQLite database
   db_path=trebek.db

   # Directory to store intermediate pipeline outputs (.json.gz)
   output_dir=gpu_outputs

   # GCP / Gemini API Key
   GEMINI_API_KEY=your_api_key_here

   # Logging level
   log_level=INFO
   ```

## 🚀 Usage

Trebek is designed to run as a continuous daemon. Once started, it will watch your target input directory for new episodes and orchestrate the full pipeline.

```bash
# Start the pipeline orchestrator
python src/main.py
```

## 🧪 Development

This project emphasizes strict type checking, robust linting, and high test coverage.

```bash
# Run the test suite
pytest

# Run the linter
ruff check .

# Run the typechecker
mypy src/
```

### Safety Invariants
If contributing, ensure you adhere to the project's invariants:
*   Always offload heavy Pydantic `Episode.model_validate_json` calls to an `asyncio.to_thread()` background thread to prevent `watchdog` heartbeat timeouts on the main event loop.
*   Database writes **must** use the `DatabaseWriter` class interface to serialize writes and prevent concurrent SQLite `database is locked` exceptions.
*   Do not pass large JSON structures across IPC (Process boundaries). Write them to disk (`.json.gz`) and pass the filepath string instead.
