# Architecture & Design

Deep technical documentation for the Trebek extraction pipeline.

---

## System Architecture

```
┌──────────────────────────────────────────────────┐
│          TrebekPipelineOrchestrator               │
│              (asyncio event loop)                 │
├──────────┬──────────┬───────────┬────────────────┤
│ Ingest   │ GPU      │ LLM      │ State Machine  │
│ Worker   │ Worker   │ Worker   │ Worker         │
│          │          │          │                │
│ polls    │ FFmpeg + │ Flash-   │ Score verify   │
│ input/   │ WhisperX │ Lite +   │ Board ctrl     │
│ dir      │ Pyannote │ Pro      │ Wager math     │
└────┬─────┴────┬─────┴────┬─────┴───────┬────────┘
     │          │          │             │
     ▼          ▼          ▼             ▼
┌──────────────────────────────────────────────────┐
│           DatabaseWriter (Actor)                  │
│        asyncio.Queue → sqlite3.Connection         │
│    journal_mode=WAL | foreign_keys=ON             │
│    busy_timeout=5000 | auto_vacuum=INCREMENTAL    │
└──────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│              SQLite Database                      │
│    pipeline_state │ episodes │ contestants        │
│    clues │ buzz_attempts │ wagers                 │
│    score_adjustments │ episode_performances       │
│    job_telemetry                                  │
└──────────────────────────────────────────────────┘
```

### Concurrency Model

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **I/O Orchestration** | `asyncio` Event-Driven | `asyncio.Event` objects replace polling latency |
| **GPU Isolation** | `ProcessPoolExecutor` (`spawn`) | Warm Worker with active GC and OOM-recovery restarts |
| **Write Serialization** | Actor pattern (`Queue` + `Future`) | Prevents SQLite `database is locked` errors |
| **CPU Offloading** | `asyncio.to_thread()` | Offloads Pydantic JSON validation off the event loop |
| **IPC Optimization** | Filepath strings over `.json.gz` | Avoids pickling large JSON across process boundaries |

---

## Pipeline Stages

Each episode flows through a rigorous sequence of stages, with `pipeline_state` as the persistent, crash-safe queue:

| Stage | Name | Engine | Description |
|-------|------|--------|-------------|
| 1 | **Ingestion** | Filesystem polling | New video files registered as `PENDING` |
| 2–3 | **GPU Extraction** | FFmpeg + WhisperX | Audio extraction, transcription, diarization |
| 4 | **Commercial Filtering** | Gemini Flash-Lite | Ad removal preserving word-level timings |
| 5 | **Structured Extraction** | Flash-Lite + Pro | Speaker anchoring → full event extraction |
| 6 | **Multimodal Augmentation** | Gemini Pro | Visual clue + podium illumination detection |
| 7 | **State Verification** | `TrebekStateMachine` | Deterministic score/adjustment validation |
| 8–9 | **Relational & Semantic Commit** | `DatabaseWriter` | Normalized INSERT + vector embeddings |

Each stage is **idempotent** — re-running only processes unfinished episodes. If any stage fails, the episode is marked `FAILED` and the daemon continues with others.

---

## Multi-Pass LLM Architecture

### Pass 1: Speaker Anchoring (Gemini 3.1 Flash-Lite)

Fast extraction targeting the host interview segment. Generates a rigid `{SPEAKER_XX: "Name"}` mapping from vocal timbres. Uses a format-agnostic normalizer handling:
- Standard JSON dicts
- Single-quoted Python dicts
- List of dicts
- Nested structures
- Prose with regex fallback

This prevents name hallucinations in later passes.

### Pass 2: Map-Reduce Structured Extraction (Gemini 3.1 Pro)

The core extraction pipeline:

1. **Meta extraction** — contestants, Final J!, score adjustments (full transcript)
2. **Semantic chunking** — splits by round boundaries (not arbitrary line counts)
3. **Concurrent clue extraction** — semaphore-bounded (3 concurrent) per chunk
4. **Timestamp reconstruction** — Line IDs resolved against parsed WhisperX JSON
5. **Composite-key deduplication** — time bucket + round + category keys
6. **Speaker normalization** — fuzzy matching against contestant roster
7. **Integrity validation** — deterministic domain rules

Includes a **Pydantic self-healing retry loop**: if LLM output fails schema validation, the `ValidationError` is injected back into the prompt for automatic correction (up to 2 retries).

**Cost optimizations:**
- Prompt compression: timestamps stripped, speaker IDs abbreviated
- Inline chunk extraction: avoids 2.5x cost penalty of context caching

### Pass 3: Multimodal Augmentation (Gemini 3.1 Pro Vision)

Vision-based temporal sniping:
- Visual clue reconstruction (images, maps, video clues)
- Exact podium lockout illumination frame detection
- Board state verification

---

## Deterministic State Machine

The `TrebekStateMachine` replays extracted atomic game events chronologically:

- **Running scores** — never trusts LLMs to do arithmetic
- **Coryat scores** — clue face value only, no DD wagers, no FJ
- **True Daily Double resolution** — resolved at runtime: `max(current_score, max_board_value)`
- **Board control tracking** — shifts only on correct responses
- **Forrest Bounce detection** — quantitative board control analysis
- **Chronological score adjustments** — applied at exactly the correct `selection_order` index
- **Speaker validation** — skips unknown speakers with capped logging

### Score Adjustment Rigidity

Score adjustments are anchored to a specific `effective_after_clue_selection_order`. The state machine applies them at *exactly* that index — not before, not after. This ensures judge reversals affect running scores precisely when they occurred in the broadcast.

---

## Physics Engine

Cross-references visual and acoustic data to compute true contestant metrics:

| Metric | Source | Calculation |
|--------|--------|-------------|
| **True buzzer latency** | Podium illumination (Vision) + acoustic buzz (WhisperX) | `buzz_timestamp_ms - host_finish_timestamp_ms` |
| **Acoustic confidence** | WhisperX word-level logprobs | Raw logprob aggregation per response |
| **Disfluency count** | WhisperX acoustic data | Deterministic um/uh detection (not LLM guesses) |
| **Semantic lateral distance** | Text embeddings | Cosine distance between clue and response vectors |
| **Brain freeze duration** | Response timestamps | `response_start - (buzz + 250ms + lockout_penalty)` |

---

## Actor-Pattern Database Writer

All SQLite writes are routed through a single `DatabaseWriter` actor:

```
   Worker 1 ─┐
   Worker 2 ──┤──▶ asyncio.Queue ──▶ sqlite3.Connection
   Worker 3 ──┤
   Worker N ─┘
              │
              └── Each enqueue returns asyncio.Future
                  protected by asyncio.wait_for()
```

**Key properties:**
- Serializes concurrent write requests (no `database is locked`)
- Supports atomic transactions via `execute_transaction` (batched multi-query commits)
- `asyncio.Future` per operation prevents silent deadlocks
- WAL mode with `busy_timeout=5000` for read concurrency

---

## Warm Worker GPU Architecture

PyTorch/WhisperX operations run inside a `ProcessPoolExecutor` (`spawn`):

- **Model weights stay resident in VRAM** — no cold-start latency between episodes
- **Explicit memory management** — `del tensor`, `gc.collect()`, `torch.cuda.empty_cache()` per task
- **Automatic OOM recovery** — if CUDA OOM occurs, the pool is restarted cleanly
- **Graceful shutdown** — `SIGTERM` via `psutil`, falling back to `SIGKILL` for zombies
- **IPC hygiene** — data passed as `.json.gz` filepaths, never pickled across process boundaries

---

## Data Model

### Relational Schema (9 Tables)

```
pipeline_state
├── episode_id (PK)
├── status           PENDING → TRANSCRIBING → TRANSCRIPT_READY →
│                    CLEANED → SAVING → VECTORIZING → COMPLETED
├── transcript_path
├── retry_count
├── last_error
├── created_at / updated_at

episodes
├── episode_id (PK)
├── air_date / host_name / is_tournament

contestants
├── contestant_id (PK)
├── name / occupational_category / is_returning_champion

episode_performances
├── episode_id (FK) + contestant_id (FK)  [composite PK]
├── podium_position (1/2/3)
├── coryat_score / final_score / forrest_bounce_index

clues
├── clue_id (PK)
├── episode_id (FK) / round / category
├── board_row / board_col / selection_order
├── clue_text / correct_response
├── is_daily_double / daily_double_wager
├── host_start_timestamp_ms / host_finish_timestamp_ms
├── clue_syllable_count / requires_visual_context
├── clue_embedding / response_embedding (BLOB)
├── semantic_lateral_distance

buzz_attempts
├── attempt_id (PK)
├── clue_id (FK) / contestant_id (FK)
├── attempt_order / buzz_timestamp_ms
├── podium_light_timestamp_ms / true_buzzer_latency_ms
├── is_lockout_inferred / response_given / is_correct
├── brain_freeze_duration_ms
├── true_acoustic_confidence_score / disfluency_count

wagers
├── wager_id (PK)
├── clue_id (FK) / contestant_id (FK)
├── running_score_at_time / actual_wager
├── game_theory_optimal_wager / wager_irrationality_delta

score_adjustments
├── adjustment_id (PK)
├── episode_id (FK) / contestant_id (FK)
├── points_adjusted / reason
├── effective_after_clue_selection_order

job_telemetry
├── episode_id (FK)
├── peak_vram_mb / avg_gpu_utilization_pct
├── stage_*_ms (6 stages)
├── gemini_total_input/output/cached_tokens
├── gemini_total_cost_usd / gemini_api_latency_ms
├── pydantic_retry_count
```

### Pydantic Data Contracts

| Model | Description |
|-------|-------------|
| `Episode` | Top-level container: contestants, clues, Final J!, score adjustments |
| `Clue` | Board position, temporal bounds, Daily Double metadata, buzz attempts |
| `BuzzAttempt` | Per-buzz reaction data: timestamps, lockout inference, response text |
| `Contestant` | Name, podium position, occupation category, champion status |
| `FinalJep` | Category, clue text, per-contestant wagers and responses |
| `ScoreAdjustment` | Chronologically anchored point corrections with reasons |
| `JobTelemetry` | Hardware signatures, token usage, latency, cost tracking |

---

## ML/AI Integration

| Provider | Model | Stage | Application |
|----------|-------|-------|-------------|
| **Local GPU** | WhisperX / Pyannote | 2–3 | Large-v3 float16 transcription, diarization |
| **Google** | Gemini 3.1 Flash-Lite | 4–5 | Speaker anchoring, commercial filtering |
| **Google** | Gemini 3.1 Pro | 5 | Structured extraction + Pydantic self-healing |
| **Google** | Gemini 3.1 Pro | 6 | Visual clue + podium illumination detection |
| **Local/API** | Text Embeddings | 9 | Cosine distance for semantic lateral distance |

---

## Integrity Validation

Post-extraction validation encodes hard domain rules:

| Check | Rule |
|-------|------|
| **Round clue counts** | J!: max 30, DJ: max 30; warn if < 15 |
| **Daily Double limits** | Max 3 total (1 in J!, 2 in DJ) |
| **DD structural** | Exactly 1 attempt per DD; wager > 0 and ≤ 50,000 |
| **Board bounds** | row ∈ [1,5], col ∈ [1,6] |
| **Position uniqueness** | No duplicate board_row within same category+round |
| **Contestant FK** | All buzz speakers and FJ wagerers must match roster |
| **Timestamp ordering** | No overlapping clue reads within a round |
| **Data completeness** | Non-empty clue_text and correct_response |
| **Category sanity** | Max 6 unique categories per round |
| **Overall quality** | < 45 clues = FAIL; ≤ 3 warnings = DEGRADED |

---

## Safety Invariants

These are **non-negotiable** constraints preserved across all contributions:

1. **GPU Warm Worker Architecture.** PyTorch/WhisperX in `ProcessPoolExecutor`. Explicit memory management per task. Auto-restart on CUDA OOM.

2. **Database Write Serialization.** All writes through `DatabaseWriter` actor queue. No direct `conn.execute()`. Multi-statement commits via `execute_transaction`.

3. **Event Loop Protection.** Heavy CPU work (`Episode.model_validate_json`) offloaded to `asyncio.to_thread()`. Stage coordination via `asyncio.Event`, not polling.

4. **IPC Boundary Hygiene.** Large JSON → compressed `.json.gz` on disk → filepath string across process boundaries. Never pickle.

5. **LLM Fact Extraction Only.** LLMs extract facts. `TrebekStateMachine` executes all arithmetic. No running score math in prompts.

6. **Chronological Score Adjustments.** Applied at exactly the correct `selection_order` index.

7. **Persistent Queue Only.** `pipeline_state` table is the inter-stage queue. Never `asyncio.Queue` for stage handoffs.

---

## Design Philosophy

### Database-Driven State over Memory
True resumability requires database-backed queueing. The pipeline can be killed at any point and resumes cleanly. Zero data loss during multi-day inference runs.

### Deterministic Math over LLM Approximations
LLMs hallucinate when doing arithmetic. They extract pure facts; the state machine handles scores, True Daily Doubles, Coryat calculations, and game-theory optimal wagers deterministically.

### Hardware Isolation & Active Management
VRAM fragmentation is inevitable in long-running PyTorch processes. The Warm Worker paradigm manages memory explicitly and recovers gracefully from OOM states.

### What Trebek Is NOT
- **Not a real-time application.** Batch-processing daemon pipeline.
- **Not an API server.** Filesystem polling + SQLite state management.
- **Not a keyword matcher.** Vectorized embeddings for semantic evaluation.

---

<div align="center">
  <sub>📖 <a href="README.md">README</a> · <a href="SETUP.md">Setup Guide</a></sub>
</div>
