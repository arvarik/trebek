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
| 4 | **Commercial Filtering** | Gemini 3.1 Flash-Lite | Ad removal preserving word-level timings |
| 5 | **Manifest-Verify-Fill Extraction** | Flash-Lite + Pro | Speaker anchoring → targeted structured extraction |
| 5.5 | **Verification & Correction** | Gemini 3.1 Flash-Lite | Post-extraction ASR correction & response verification |
| 6 | **Multimodal Augmentation** | Gemini 3.1 Pro | Visual clue + podium illumination detection |
| 7 | **State Verification** | `TrebekStateMachine` | Deterministic score/adjustment validation + quality gate |
| 8 | **Relational Commit** | `DatabaseWriter` | Normalized INSERT into 9 analytical tables |
| 9 | **Semantic Vectorization** | *Not yet implemented* | Text embeddings + cosine distance (see `docs/embeddings_feature.md`) |

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

### Pass 2: Manifest-Verify-Fill Extraction (Gemini 3.1 Pro + Flash-Lite)

The core extraction pipeline:

1. **Meta extraction** — contestants, categories, Final J!, score adjustments (full transcript)
2. **Board Manifest Construction** — splits transcript by round (J! vs Double J!) and builds grid bounds.
3. **Category Discovery Fallback** — uses Flash to find categories missed during truncations.
4. **Concurrent Round Extraction** — independent, category-aware extraction per round.
5. **Gap Detection & Targeted Fill** — compares extracted clues to the manifest and surgically re-extracts missing rows.
6. **Verify & Correct** — uses Flash to cross-validate clue texts and correct ASR errors.
7. **Deterministic Board Inference** — maps exact dollar values (e.g., "$800") to board rows, overriding LLM guesses.
8. **Normalization** — fuzzy speaker reconciliation and strict "What is/Who is" response formatting.

Includes a **Pydantic self-healing retry loop**: if LLM output fails schema validation, the `ValidationError` is injected back into the prompt for automatic correction (up to 2 retries).

**Cost optimizations:**
- Prompt compression: timestamps stripped, speaker IDs abbreviated
- Targeted Gap Fills: only missing categories are re-queried, avoiding full chunk retries.

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
│                    CLEANED → SAVING → MULTIMODAL_PROCESSING →
│                    MULTIMODAL_DONE → VECTORIZING → COMPLETED
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
├── is_verified / original_response        ← Stage 5.5 verification metadata
├── is_daily_double / is_triple_stumper
├── daily_double_wager / wagerer_name
├── host_start_timestamp_ms / host_finish_timestamp_ms
├── clue_syllable_count / host_speech_rate_wpm
├── requires_visual_context / selector_had_board_control
├── clue_embedding / response_embedding (BLOB)  ← Not yet populated (Stage 9)
├── semantic_lateral_distance                    ← Not yet populated (Stage 9)

buzz_attempts
├── attempt_id (PK)
├── clue_id (FK) / contestant_id (FK)
├── attempt_order / buzz_timestamp_ms
├── podium_light_timestamp_ms / true_buzzer_latency_ms
├── is_lockout_inferred / response_given / is_correct
├── brain_freeze_duration_ms
├── true_acoustic_confidence_score / disfluency_count
├── phonetic_similarity_score

wagers
├── wager_id (PK)
├── clue_id (FK) / contestant_id (FK)
├── running_score_at_time / opponent_1_score / opponent_2_score
├── actual_wager
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

schema_version
├── version (PK) / name / applied_at
```

### Pydantic Data Contracts

| Model | Module | Description |
|-------|--------|-------------|
| `Episode` | `schemas` | Top-level container: contestants, clues, Final J!, score adjustments |
| `Clue` | `schemas` | Board position, temporal bounds, verification status, Daily Double metadata, buzz attempts |
| `BuzzAttempt` | `schemas` | Per-buzz reaction data: timestamps, lockout inference, response text |
| `Contestant` | `schemas` | Name, podium position, occupation category, champion status |
| `FinalJep` | `schemas` | Category, clue text, per-contestant wagers and responses |
| `ScoreAdjustment` | `schemas` | Chronologically anchored point corrections with reasons |
| `JobTelemetry` | `schemas` | Hardware signatures, token usage, latency, cost tracking |
| `ClueExtraction` | `llm/schemas` | Intermediate extraction model with `is_verified` / `original_response` for Stage 5.5 |
| `BuzzAttemptExtraction` | `llm/schemas` | Extraction-time buzz attempt with line ID references |

---

## ML/AI Integration

| Provider | Model | Stage | Application |
|----------|-------|-------|-------------|
| **Local GPU** | WhisperX / Pyannote | 2–3 | Large-v3 float16 transcription, diarization |
| **Google** | Gemini 3.1 Flash-Lite | 4–5 | Speaker anchoring, commercial filtering |
| **Google** | Gemini 3.1 Pro | 5 | Structured extraction + Pydantic self-healing |
| **Google** | Gemini 3.1 Flash-Lite | 5.5 | Post-extraction verification & ASR correction |
| **Google** | Gemini 3.1 Pro | 6 | Visual clue + podium illumination detection |
| **Google/Local** | Text Embeddings | 9 | Cosine distance for semantic lateral distance *(not yet implemented)* |

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
| **Pre-Commit FK Firewall**| Drops buzzes, wagers, or adjustments tied to unknown speakers |
| **Format Normalization** | All correct responses forced into proper J! question format |
| **Timestamp ordering** | No overlapping clue reads within a round |
| **Final J! Completeness** | Validates wager count against eliminated contestants |
| **Category sanity** | Max 6 unique categories per round |
| **Overall quality** | < 45 clues = FAIL; severity-weighted warnings > 3 = DEGRADED |

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
