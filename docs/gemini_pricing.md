# Gemini API Pricing Reference

> **Last Updated:** 2026-04-26
> **Source:** [Google AI Pricing](https://ai.google.dev/gemini-api/docs/pricing#standard)

## Models Used by Trebek

### Gemini 3.1 Pro Preview (`gemini-3.1-pro-preview`) — alias: `pro`

Used for: Pass 2 Clue Extraction (default), Pass 3 Multimodal Sniping, Vision Analysis

| Tier | Input (per 1M tokens) | Output (per 1M tokens) | Notes |
|------|----------------------|------------------------|-------|
| **Standard** | $2.00 (≤200k), $4.00 (>200k) | $12.00 (≤200k), $18.00 (>200k) | Output includes thinking tokens |
| **Context Caching** | $0.20 (≤200k), $0.40 (>200k) | — | + $4.50/M tokens/hour storage |
| **Batch** | TBD | TBD | Not currently used |

---

### Gemini 3 Flash Preview (`gemini-3-flash-preview`) — alias: `flash3`

Used for: Optional Pass 2 Clue Extraction (via `--model flash3`)

| Tier | Input (per 1M tokens) | Output (per 1M tokens) | Notes |
|------|----------------------|------------------------|-------|
| **Standard** | $0.50 (text/image/video), $1.00 (audio) | $3.00 | Output includes thinking tokens |
| **Context Caching** | $0.05 (text/image/video), $0.10 (audio) | — | + $1.00/M tokens/hour storage |
| **Batch** | TBD | TBD | Not currently used |

---

### Gemini 3.1 Flash-Lite Preview (`gemini-3.1-flash-lite-preview`) — alias: `flash`

Used for: Pass 1 Speaker Anchoring, Flash Repair (JSON fix fallback)

| Tier | Input (per 1M tokens) | Output (per 1M tokens) | Notes |
|------|----------------------|------------------------|-------|
| **Standard** | $0.25 (text/image/video), $0.50 (audio) | $1.50 | Most cost-efficient model |
| **Context Caching** | $0.025 (text/image/video), $0.05 (audio) | — | + $1.00/M tokens/hour storage |
| **Batch** | TBD | TBD | Not currently used |

---

## Model Selection Guide

| Alias | Model | Quality | Speed | Cost | Best For |
|-------|-------|---------|-------|------|----------|
| `pro` | gemini-3.1-pro-preview | ★★★★★ | ★★☆☆☆ | $$$$$ | Production extraction, max accuracy |
| `flash3` | gemini-3-flash-preview | ★★★★☆ | ★★★★☆ | $$$ | Balanced quality/cost |
| `flash` | gemini-3.1-flash-lite-preview | ★★★☆☆ | ★★★★★ | $ | Speaker anchoring, repairs, bulk runs |

## Pipeline Cost Estimates (Per Episode)

Typical Jeopardy episode (~22 min, ~300 transcript segments):

### Using `--model pro` (default)

| Stage | Model | Est. Input Tokens | Est. Output Tokens | Est. Cost |
|-------|-------|-------------------|--------------------|-----------| 
| Pass 1 — Speaker Anchoring | flash-lite | ~2,500 | ~100 | ~$0.0008 |
| Pass 2 — Metadata Extraction | pro | ~3,000 | ~800 | ~$0.016 |
| Pass 2 — Clue Chunks (×2-3) | pro | ~8,000 | ~6,000 | ~$0.088 |
| Pass 3 — Multimodal (if needed) | pro | ~1,000 | ~200 | ~$0.004 |
| **Total** | | | | **~$0.11** |

### Using `--model flash3`

| Stage | Model | Est. Input Tokens | Est. Output Tokens | Est. Cost |
|-------|-------|-------------------|--------------------|-----------| 
| Pass 1 — Speaker Anchoring | flash-lite | ~2,500 | ~100 | ~$0.0008 |
| Pass 2 — Metadata + Clues | flash3 | ~11,000 | ~6,800 | ~$0.026 |
| Pass 3 — Multimodal (if needed) | flash3 | ~1,000 | ~200 | ~$0.001 |
| **Total** | | | | **~$0.03** |

### Using `--model flash`

| Stage | Model | Est. Input Tokens | Est. Output Tokens | Est. Cost |
|-------|-------|-------------------|--------------------|-----------| 
| Pass 1 — Speaker Anchoring | flash-lite | ~2,500 | ~100 | ~$0.0008 |
| Pass 2 — Metadata + Clues | flash-lite | ~11,000 | ~6,800 | ~$0.013 |
| Pass 3 — Multimodal (if needed) | flash-lite | ~1,000 | ~200 | ~$0.001 |
| **Total** | | | | **~$0.01** |

## At Scale

| Scale | Episodes | Pro Cost | Flash3 Cost | Flash Cost |
|-------|----------|----------|-------------|------------|
| Single season | ~200 | ~$22 | ~$6 | ~$3 |
| Full J-Archive | ~9,000 | ~$990 | ~$270 | ~$135 |

## Notes

- All prices are **Paid Tier** rates in USD
- Trebek uses **Standard** mode only (no Batch API)
- Context caching is used for Pass 2 when transcript is large enough (>32k tokens)
- Flash-Lite is always used for Pass 1 speaker anchoring (audio input pricing applies: $0.50/M)
- Flash Repair (JSON fix fallback) always uses Flash-Lite regardless of `--model` setting
- Grounding with Google Search: 5,000 free prompts/month shared across Gemini 3, then $14/1,000 queries (not used by Trebek)
