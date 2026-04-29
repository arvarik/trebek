<div align="center">

```
 ╺┳╸┏━┓┏━╸┏┓ ┏━╸┃┏╸
  ┃ ┣┳┛┣╸ ┣┻┓┣╸ ┣┻┓
  ╹ ╹┗╸┗━╸┗━┛┗━╸╹ ╹
```

**The definitive multimodal AI pipeline for extracting structured game data from J! episodes.**

*From casual trivia lovers to ML engineers — one dataset to rule them all.*

<a href="https://github.com/arvarik/trebek/actions/workflows/ci.yml">
  <img alt="CI" src="https://github.com/arvarik/trebek/actions/workflows/ci.yml/badge.svg" />
</a>
<a href="https://pypi.org/project/trebek/">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/trebek" />
</a>
<a href="https://pypi.org/project/trebek/">
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/trebek" />
</a>
<img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" />
<img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" />
<img alt="Mypy" src="https://img.shields.io/badge/types-Mypy-blue.svg" />
<img alt="License" src="https://img.shields.io/badge/license-AGPL_3.0-green" />

</div>

---

## What is Trebek?

Trebek is an advanced, fault-tolerant pipeline that processes **raw J! video recordings** — not scraped web pages — and produces a **surgically clean, event-sourced relational dataset** of every game event that occurred on screen. It bridges local GPU compute (WhisperX, Pyannote), cloud LLMs (Google Gemini 3.1), and a deterministic Python state machine into a single, continuously running daemon.

The resulting dataset doesn't just capture questions and answers. It captures the full *cognitive fingerprint* of each game:

- ⚡ **Millisecond-precision buzzer latencies** — cross-referenced from visual podium illumination and acoustic buzz detection
- 🗣️ **Speech disfluency tracking** — ums, uhs, and stutters from WhisperX logprobs, not LLM hallucinations
- 🎲 **Game-theory optimal wager analysis** — calculated wagers compared against actual contestant choices
- 🔍 **Post-extraction verification** — Stage 5.5 cross-validates every clue against transcript context to correct ASR errors
- 🧠 **Semantic lateral distance** — cosine distance on embeddings distinguishing wordplay from direct recall *(schema ready, see `docs/embeddings_feature.md`)*
- 🏗️ **Board control & Forrest Bounce detection** — strategic selection pattern analysis
- 📊 **Coryat scores** — calculated deterministically per contestant per episode

---

## Trebek vs. J-Archive

Existing J! datasets are **static text scrapes** — frozen lists of clues and responses with no temporal, behavioral, or strategic context. Trebek extracts from the *raw video*, producing an entirely different class of dataset.

| Dimension | J-Archive / Scraped Data | Trebek |
|-----------|--------------------------|--------|
| **Source** | Web scraping | Raw video processing |
| **Buzzer timing** | ❌ Not available | ✅ True ms-precision latency |
| **Speech patterns** | ❌ Not available | ✅ Disfluency counts, acoustic confidence |
| **Wager analysis** | Partial (raw numbers only) | ✅ Game-theory optimal + irrationality delta |
| **Board control** | ❌ Not available | ✅ Full selection order + Forrest Bounce index |
| **Score adjustments** | Sometimes noted | ✅ Chronologically anchored to exact clue index |
| **Visual clues** | Text description | ✅ Multimodal extraction from video frames |
| **Semantic analysis** | ❌ Not available | ✅ Embedding schema ready (see `docs/embeddings_feature.md`) |
| **Data format** | Flat HTML / CSV | ✅ Normalized relational DB (9 tables) |
| **Freshness** | Depends on scraper maintenance | ✅ Process your own recordings on demand |
| **Coryat scores** | Manual fan calculation | ✅ Deterministic, per-contestant |

---

## Who Is This For?

<table>
<tr>
<td width="33%" align="center">

### 🎯 Trivia Enthusiasts

Explore your favorite episodes with deep analytics. Query buzzer speeds, track contestant strategies, and discover board control patterns across seasons.

</td>
<td width="33%" align="center">

### 📊 Data Scientists

A richly normalized relational dataset designed for analytical queries. 9 tables, foreign keys, embeddings, and temporal data — ready for your notebooks.

</td>
<td width="33%" align="center">

### 🤖 ML Engineers

Train predictive models on human decision-making under televised pressure. Buzzer latency, wager irrationality, disfluency signals — features you can't get anywhere else.

</td>
</tr>
</table>

---

## ✨ Feature Highlights

### 🔄 True Crash Immunity
Database-backed queueing via SQLite `pipeline_state`. Kill the daemon at any point — `SIGINT`, `SIGTERM`, crash, power failure — and it resumes exactly where it left off. Zero data loss. Zero re-processing.

### 🧠 Multi-Pass LLM Architecture
- **Pass 1** (Flash-Lite): Speaker anchoring from host interview audio
- **Pass 2** (Pro): Manifest-Verify-Fill structured extraction with category gap detection
- **Stage 5.5** (Flash-Lite): Post-extraction verification — cross-validates every clue against transcript context, corrects ASR errors, normalizes response formatting, and tracks `is_verified` / `original_response` metadata
- **Pass 3** (Pro): Multimodal visual clue reconstruction + podium illumination detection

### ⚙️ Deterministic State Machine
Pure Python `TrebekStateMachine` replays game events chronologically. LLMs extract facts; the state machine does all arithmetic. Running scores, True Daily Double resolution, Coryat scores, and game-theory optimal wagers — all calculated deterministically.

### 🎯 Deterministic Inference
Instead of relying on LLM hallucinations for grid positions, Trebek parses exact dollar values ("for $800") and deterministically maps them to the correct board row based on the round format. Response formats are strictly normalized into J! question form.

### 🔥 Warm Worker GPU Architecture
PyTorch/WhisperX model weights stay resident in VRAM. No cold starts. Automatic OOM recovery with pool restarts. Explicit memory management for multi-day inference runs.

### 🎯 Physics Engine
Cross-references visual podium illumination (Gemini Vision) with WhisperX acoustic boundaries to compute true contestant reaction speeds. Also calculates acoustic confidence scores, brain freeze durations, and semantic lateral distance.

### 🗄️ Actor-Pattern Database
All SQLite writes serialized through a single `DatabaseWriter` actor (`asyncio.Queue` + `Future`). No `database is locked` exceptions. Atomic transactions for high-throughput batched commits.

---

## 🚀 Quick Start

The fastest way to get Trebek running is using the official Docker image via Hybrid Mode. The lightweight CLI runs on your host, while the heavy GPU workloads (PyTorch, WhisperX) are safely delegated to the `ghcr.io` container.

```bash
# 1. Install lightweight CLI
pip install trebek

# 2. Configure (requires a free Gemini API key)
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Run with Docker GPU delegation
trebek run --input-dir /path/to/your/videos --docker
```

> **📖 Full installation guide**: See **[SETUP.md](SETUP.md)** for `docker-compose` deployments, native installations (no Docker), HuggingFace token configuration, and detailed CLI usage.

> **🏗️ Architecture deep-dive**: See **[DESIGN.md](DESIGN.md)** for the complete system architecture, data model, pipeline stages, and safety invariants.

---

## 📊 Stats Dashboard

Run `trebek stats` for a live analytics dashboard showing pipeline health, cost tracking, stage timing, and recent episode status:

```
┌─ Pipeline Health ─────────────────────────────────────────┐
│  ✅ COMPLETED  42    ⏳ PENDING  3    ❌ FAILED  1       │
│  ████████████████████████████████████░░░░  91.3%          │
├─ Cost & Performance ──────────────────────────────────────┤
│  Tokens: 12.4M in / 2.1M out    Cost: $4.82 USD          │
│  Peak VRAM: 14.2 GB    Avg GPU: 87%                      │
├─ Stage Timing (avg) ──────────────────────────────────────┤
│  transcribe: 4m 12s    extract: 2m 38s    verify: 0.4s   │
└───────────────────────────────────────────────────────────┘
```

---

## 🧪 Development

```bash
make all          # Full quality gate (test + lint + typecheck)
make test         # pytest with coverage
make lint         # ruff check
make typecheck    # mypy strict mode
```

| Tool | Purpose |
|------|---------|
| **pytest** | Test runner (`pytest-asyncio` for async) |
| **ruff** | Linter + formatter (line-length 120) |
| **mypy** | Static type checker (strict mode) |
| **pre-commit** | Git hook enforcement |

---

## 📁 Project Structure

```
trebek/
├── trebek/
│   ├── cli.py              # CLI parser + Docker orchestration
│   ├── config.py           # Pydantic Settings + model constants + pricing
│   ├── schemas.py          # Pydantic v2 data contracts (Episode, Clue, etc.)
│   ├── schema.sql          # SQLite DDL (9 tables + schema_version)
│   ├── state_machine.py    # Deterministic game state replay
│   ├── status.py           # Pipeline status enum (StrEnum)
│   ├── database/           # Actor-pattern writer + relational commit ops
│   ├── gpu/                # Warm Worker pool + VRAM management
│   ├── llm/                # Multi-pass Gemini extraction (anchoring, extraction, verify, multimodal)
│   ├── pipeline/           # Async orchestrator + stage workers (ingestion, gpu, llm, state_machine)
│   ├── analysis/           # Post-extraction analytics (buzzer physics, embeddings math)
│   └── ui/                 # Rich console dashboard + rendering
├── tests/                  # Comprehensive test suite (512+ tests)
├── scripts/                # Local testing + validation utilities
├── docs/                   # Design docs, embedding feature plan, archived architecture
├── Dockerfile              # GPU-enabled container (CUDA + WhisperX + Pyannote)
├── docker-compose.yml      # One-command deployment
├── Makefile                # Developer shortcuts (test, lint, typecheck)
└── pyproject.toml          # Build system + tool config
```

---

## 📄 License

[AGPL-3.0](LICENSE)

---

<div align="center">
  <sub>Built for anyone who believes the best datasets are the ones you extract yourself.</sub>
</div>
