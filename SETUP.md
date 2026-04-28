# Setup Guide

Complete installation and usage instructions for Trebek.

---

## 🐳 Docker Quick Start (Recommended)

The fastest way to get Trebek running — handles all GPU dependencies automatically.

### Prerequisites

| Requirement | How to Get It |
|-------------|---------------|
| **Docker** | [Install Docker](https://docs.docker.com/get-docker/) |
| **NVIDIA Container Toolkit** | [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| **NVIDIA GPU** | 16GB VRAM recommended (RTX 4060 Ti / 5060 Ti) |
| **Gemini API Key** | [Get one free](https://aistudio.google.com/apikey) |

### 5-Step Setup

```bash
# 1. Download
curl -O https://raw.githubusercontent.com/arvarik/trebek/main/docker-compose.yml
curl -o .env https://raw.githubusercontent.com/arvarik/trebek/main/.env.example

# 2. Configure
cp .env.example .env
# Edit .env → set GEMINI_API_KEY and HF_TOKEN

# 3. Add videos
mkdir -p input_videos
# Copy/symlink your J! episode files here (nested folders OK)

# 4. Launch (auto-pulls GHCR image)
docker compose up -d

# 5. Monitor
docker compose logs -f
```

> ⚠️ **SQLite WAL & Network Drives**: `trebek.db` **must** be on a local disk (ext4/NTFS/APFS). Network mounts (NFS/SMB) will corrupt the database.

---

## 💻 Native Installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python** | 3.11+ |
| **FFmpeg** | Audio extraction |
| **NVIDIA GPU + CUDA** | WhisperX acceleration |
| **SQLite** | 3.35+ (`RETURNING` clause) |
| **Gemini API Key** | [Free](https://aistudio.google.com/apikey) |
| **HuggingFace Token** | [Free](https://huggingface.co/settings/tokens) — see below |

```bash
# Install trebek
pip install trebek

# Install GPU dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install whisperx pyannote.audio

# Configure
cp .env.example .env   # Edit with your API keys
```

### Speaker Diarization Setup

WhisperX uses [pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization (free HuggingFace token required):

1. **Create account** at [huggingface.co/join](https://huggingface.co/join)
2. **Accept model licenses** (click "Agree and access"):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. **Generate token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. **Add to `.env`**: `HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

> ⚠️ Without `HF_TOKEN`, diarization is skipped. Extraction accuracy drops ~50%.

### Docker Hybrid Mode

Install Trebek natively for the CLI but delegate GPU work to the official GHCR Docker image:

```bash
pip install trebek         # Lightweight (no PyTorch)
trebek run --docker        # Pulls from GHCR and delegates GPU work
```

---

## ⚙️ Configuration

Trebek uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — loads from `.env` or environment variables.

```env
# ─── Core Paths ───
db_path=trebek.db
output_dir=gpu_outputs
input_dir=input_videos

# ─── API Keys ───
GEMINI_API_KEY=              # Required
HF_TOKEN=                    # Required for diarization

# ─── Logging ───
log_level=INFO

# ─── GPU Constraints ───
gpu_vram_target_gb=16        # 4–24 GB
whisper_batch_size=8         # Tune for your VRAM
whisper_compute_type=float16 # float16 or float32
```

| Setting | Constraint | Default |
|---------|-----------|---------|
| `gpu_vram_target_gb` | 4–24 | `16` |
| `whisper_compute_type` | `float16` / `float32` | `float16` |
| `whisper_batch_size` | > 0 | `8` |

---

## 🚀 Usage Guide

### CLI Commands

| Command | Description |
|---------|-------------|
| `trebek run` | Start pipeline (daemon mode by default) |
| `trebek run --once` | Process queue then exit |
| `trebek run --docker` | Delegate GPU work to Docker |
| `trebek scan` | Preview discovered files + pipeline status |
| `trebek stats` | Live analytics dashboard |
| `trebek retry` | Reset FAILED episodes → PENDING |
| `trebek version` | Print version |

### Stage-Isolated Processing

```bash
trebek run --stage transcribe --once    # GPU transcription only
trebek run --stage extract --model flash # Cheap LLM extraction
trebek scan --stage transcribe           # Files needing GPU work
```

### Model Selection

| Alias | Model | Cost (per M tokens) |
|-------|-------|---------------------|
| `pro` | gemini-3.1-pro-preview | $2.00 in / $12.00 out |
| `flash3` | gemini-3-flash-preview | $0.50 in / $3.00 out |
| `flash` | gemini-3.1-flash-lite-preview | $0.25 in / $1.50 out |

### Supported Video Formats

MP4, TS, MKV, AVI, MOV, WebM, MPG, MPEG, FLV, WMV, M2TS, VOB

---

## 📊 Stats Dashboard

`trebek stats` renders a Rich-powered analytics dashboard:

| Section | Metrics |
|---------|---------|
| **Pipeline Health** | Per-status counts, completion %, progress bar |
| **Performance** | Total tokens, API cost (USD), peak VRAM |
| **Stage Timing** | Avg/min/max per stage |
| **Recent Episodes** | Last 10 with status, retries, errors |
| **Cost Breakdown** | Per-model token usage + cost split |

### Status Legend

| Status | Meaning |
|--------|---------|
| 🟢 COMPLETED | All stages passed, committed to DB |
| ⏳ PENDING | Queued for processing |
| 🎤 TRANSCRIBING | GPU worker active |
| 📝 TRANSCRIPT_READY | Awaiting LLM extraction |
| 🧹 CLEANED | LLM extraction underway |
| 🔬 MULTIMODAL | Visual augmentation in progress |
| 🔴 FAILED | Check `trebek stats` for details |

---

## 🔍 Querying Results

```bash
sqlite3 trebek.db
```

```sql
-- Fastest buzzers
SELECT c.name, ba.true_buzzer_latency_ms
FROM buzz_attempts ba
JOIN contestants c ON ba.contestant_id = c.contestant_id
WHERE ba.is_correct = 1
ORDER BY ba.true_buzzer_latency_ms ASC LIMIT 5;

-- Irrational Daily Double wagers
SELECT c.name, w.actual_wager, w.game_theory_optimal_wager, w.wager_irrationality_delta
FROM wagers w
JOIN contestants c ON w.contestant_id = c.contestant_id
WHERE ABS(w.wager_irrationality_delta) > 500
ORDER BY ABS(w.wager_irrationality_delta) DESC;

-- Wordplay-heavy categories
SELECT category, AVG(semantic_lateral_distance) AS avg_dist
FROM clues GROUP BY category ORDER BY avg_dist DESC LIMIT 10;

-- Top Coryat scores
SELECT c.name, ep.coryat_score, ep.final_score
FROM episode_performances ep
JOIN contestants c ON ep.contestant_id = c.contestant_id
ORDER BY ep.coryat_score DESC LIMIT 10;
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| **"database is locked"** | Ensure only one `trebek` instance per DB. Verify local filesystem (not NFS/SMB). |
| **CUDA OOM** | Lower `whisper_batch_size` to `4`. Lower `gpu_vram_target_gb` to `12`. Pipeline auto-recovers. |
| **Gemini rate limits** | Use `--model flash`. Pipeline has built-in retry with backoff. |
| **HuggingFace 403** | Accept licenses at [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0). |
| **Low clue count (<45)** | Use `--model pro`. Ensure `HF_TOKEN` is set. Try `trebek retry && trebek run --once`. |

---

<div align="center">
  <sub>📖 <a href="README.md">README</a> · <a href="DESIGN.md">Architecture</a></sub>
</div>
