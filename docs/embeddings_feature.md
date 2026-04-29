# Embedding Feature — Design & Implementation Guide

> **Status**: Designed, scaffolded, not yet wired into the pipeline.
> **Priority**: P2 — implement after core extraction pipeline is stable.
> **Estimated effort**: ~2–3 hours (Option A).

---

## Original Intent

The embedding feature was designed as **Stage 9** of the pipeline — the final step before an episode reaches `COMPLETED` status. The original architecture docs (`docs/archive/pre-gemstack/04_pipeline_stages.md`, `05_ml_feature_engineering.md`) describe it as:

> **Stage 9: Semantic Board Search (RAG via sqlite-vec)**
> Generates two separate embeddings — `clue_embedding` (Category + Clue) and `response_embedding` (Correct Response) — and computes the `semantic_lateral_distance` (cosine distance between them).

### Semantic Lateral Distance (SLD)

The key analytical metric this feature enables is the **Semantic Difficulty Index**:

- **Low cosine distance** = direct factual recall (clue and response are semantically similar)
  - e.g., "This French city has a famous iron tower" → "What is Paris?"
- **High cosine distance** = heavy wordplay/lateral thinking (clue and response are semantically distant)
  - e.g., "It can be a chess piece or a crow's sound" → "What is a rook?"

**ML applications:**
1. Mathematically quantify how "lateral" a J! clue is
2. Predict contestant miss probability based on clue semantic structure (not just category)
3. Build a RAG search engine over all J! clues for semantic similarity queries
4. Distinguish categories that rely on wordplay vs. direct knowledge

---

## What's Already Built

| Component | File | Status |
|---|---|---|
| `cosine_distance()` | `trebek/analysis/embeddings.py` | ✅ Production-ready, tested |
| `process_semantic_lateral_distance()` | `trebek/analysis/embeddings.py` | ✅ Production-ready, tested |
| Unit tests (10 cases) | `tests/analysis/test_embeddings.py` | ✅ Passing |
| Schema columns | `trebek/schema.sql` (lines 66–68) | ✅ In DB, all NULL |
| Pipeline status `VECTORIZING` | `trebek/status.py` | ✅ Exists (currently misused) |
| Telemetry field `stage_vectorization_ms` | `trebek/schema.sql` (line 127) | ✅ Exists (currently misused) |
| Design documentation | `DESIGN.md`, `README.md` | ✅ References embeddings |

### Schema Columns (Already in `clues` Table)

```sql
clue_embedding BLOB,           -- Packed float vector (struct.pack)
response_embedding BLOB,       -- Packed float vector (struct.pack)
semantic_lateral_distance REAL  -- Cosine distance between the two
```

### Current Misuse of `VECTORIZING`

The `VECTORIZING` status is currently repurposed by `state_machine_worker` as a transient state during the "verify + commit" step. The `stage_vectorization_ms` telemetry field records state machine commit time, not actual vectorization. When implementing embeddings, this needs to be restored to its original purpose.

---

## Why It Wasn't Implemented

The embedding stage was deprioritized during the "Gemstack" migration (pre-Gemstack docs are in `docs/archive/pre-gemstack/`). Pipeline development focused on:

1. GPU transcription stability and warm worker architecture
2. Multi-pass LLM extraction accuracy (manifest-verify-fill)
3. Stage 3.5 verification & correction
4. Multimodal augmentation (visual clues, podium illumination)
5. State machine integrity and Coryat score calculation

The scaffolding was laid across 6 different files but the actual embedding generation and pipeline wiring were never connected.

---

## Implementation Options

### Option A: Gemini Text Embedding API (Recommended)

**Provider**: `text-embedding-004` via the existing `google-genai` SDK
**Why recommended**: Zero new dependencies, same API key, same rate-limit handling.

```python
# The google-genai SDK already in use supports embeddings:
response = client.models.embed_content(
    model='text-embedding-004',
    contents='Category: Songwriters. Clue: This French city...',
    config=types.EmbedContentConfig(
        task_type='SEMANTIC_SIMILARITY',
        output_dimensionality=256  # Reduced for storage efficiency
    )
)
embedding = response.embeddings[0].values  # List[float], 256 dims
```

**Cost**: Free tier includes 1,500 RPM / 1M tokens per day. At ~60 clues × 2 embeddings × ~20 tokens = ~2,400 tokens/episode. A 1,000-episode batch is well within free tier.

**Input formatting** (per the original design):
- `clue_embedding`: `"{category}. {clue_text}"` — combines category context with clue text
- `response_embedding`: `"{correct_response}"` — the correct response in isolation

### Option B: Local Sentence Transformers

**Provider**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
**Why**: Offline, no API calls, faster per-item.
**Drawback**: New dependency (though PyTorch is already in the Docker image for WhisperX). Needs GPU memory arbitration — must run after WhisperX unloads or on CPU.
**Extra effort**: ~1 hour for model lifecycle management.

### Option C: sqlite-vec RAG Extension (Stretch Goal)

Adds KNN search over the embedding vectors using SQLite's `sqlite-vec` extension:

```sql
CREATE VIRTUAL TABLE clue_vectors USING vec0(
    clue_id TEXT PRIMARY KEY,
    embedding FLOAT[256]
);
-- KNN search:
SELECT * FROM clue_vectors WHERE embedding MATCH ? ORDER BY distance LIMIT 10;
```

This is an **independent feature** that builds on top of embedding data. Effort: ~2 hours after embeddings exist.

---

## Implementation Checklist

> For whichever option is chosen, these are the concrete tasks:

### 1. Add `embed_content()` to `GeminiClient` (~15 lines)

Add an embedding method to `trebek/llm/client.py` wrapping `client.models.embed_content()` with the same retry/logging pattern as `generate_content()`.

### 2. Add config settings to `trebek/config.py`

```python
# New constants
EMBEDDING_MODEL = "text-embedding-004"

# New Settings fields
embedding_dimensionality: int = Field(default=256, description="Output dimensionality for text embeddings")
```

### 3. Create `trebek/pipeline/workers/embedding.py`

New pipeline worker following the exact pattern of `multimodal_worker.py`:
- Poll for episodes at the target status
- Load episode data from the JSON file
- For each clue: generate `clue_embedding` and `response_embedding`
- Compute `semantic_lateral_distance` using existing `process_semantic_lateral_distance()`
- UPDATE the clues table with the three new values
- Advance to `COMPLETED`

### 4. Wire into `trebek/pipeline/orchestrator.py`

- Add `embedding_work_ready = asyncio.Event()`
- Add `embedding_worker` to `start_workers()` under a new `"vectorize"` stage
- Update `ACTIVE_STAGES` in `trebek/pipeline/stages.py`

### 5. Fix status flow in `trebek/pipeline/workers/state_machine.py`

Restore `VECTORIZING` to its original purpose:
- State machine commits → status becomes ready for embedding
- Embedding worker polls for that status → generates vectors → `COMPLETED`

### 6. Update `trebek/database/operations.py`

Map `clue_embedding`, `response_embedding`, and `semantic_lateral_distance` in the clue INSERT payload. Use `struct.pack()` for BLOB serialization:

```python
import struct

def pack_embedding(vec: list[float]) -> bytes:
    return struct.pack(f'{len(vec)}f', *vec)

def unpack_embedding(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f'{dim}f', blob))
```

### 7. Add `--stage vectorize` CLI support

Follow existing pattern in `trebek/cli.py` and `trebek/pipeline/stages.py` for stage-targeted runs. This enables:

```bash
# Backfill embeddings for all completed episodes
trebek run --stage vectorize --once
```

### 8. Add telemetry

Track `stage_vectorization_ms` with actual embedding generation time (currently tracks state machine commit time — needs renaming or a new field).

---

## Architecture: Inline vs. Separate Stage

**Recommended: Separate Stage** (original design)

```
MULTIMODAL_DONE → State Machine + Commit → [new status] → Embedding Worker → COMPLETED
```

- Episodes reach a "data complete" state even if embeddings fail
- Embeddings can be backfilled independently via `--stage vectorize --once`
- Embedding API outages don't block the extraction pipeline
- Matches the original Stage 9 design intent

**Not recommended: Inline** (embed during state machine commit)
- Simpler, but couples vectorization to the critical path
- If Gemini embedding API is down, episodes get stuck at `VECTORIZING`

---

## Backfill Strategy

Once implemented, all existing episodes can be backfilled:

```bash
# Process all completed episodes that don't have embeddings yet
trebek run --stage vectorize --once
```

The worker should check for NULL `clue_embedding` and only process unembedded clues, making it safe to run repeatedly.

---

## References

- Original pipeline stages: `docs/archive/pre-gemstack/04_pipeline_stages.md` (Stage 9)
- ML feature engineering: `docs/archive/pre-gemstack/05_ml_feature_engineering.md` (Section 3)
- Current design doc: `DESIGN.md` (Stage 8–9, Physics Engine, ML/AI table)
- Existing math utilities: `trebek/analysis/embeddings.py`
- Existing tests: `tests/analysis/test_embeddings.py`
- Gemini embedding API: https://ai.google.dev/gemini-api/docs/embeddings
