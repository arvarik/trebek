# Exploration: Job Processing Telemetry & Metrics Surfacing

### Context
Trebek currently operates as a resilient, database-driven daemon. While it successfully processes multi-day inference runs, we lack granular visibility into the pipeline's operational physics. We don't know where the time is being spent, how efficiently we are utilizing our GPU vs Cloud LLM resources, or the exact cost and token footprint of our Gemini API calls. We need a telemetry table and a mechanism to surface these insights beautifully to our users.

### Ideas Considered

#### 1. Gemini AI Telemetry & Self-Healing Metrics
- **The pain**: We currently treat Gemini API calls as a black box regarding costs and efficiency. If a job takes longer, we don't know if it's due to rate limiting, high token counts, or the Pydantic self-healing retry loop fighting hallucinations.
- **The opportunity**: A dedicated telemetry table capturing token usage (input/output/cached), total API latency, and a critical "retry count" metric. We want to know exactly how often the LLM failed schema validation and required correction.
- **Rough size**: Medium
- **Priority**: Must-have

#### 2. Pipeline Stage Execution Latency (Micro-timing)
- **The pain**: When a batch of 100 episodes runs overnight, users have no idea what the bottleneck is. Is WhisperX transcription holding us back, or is Gemini Flash commercial filtering the slow step?
- **The opportunity**: Tracking millisecond-precision execution times for every distinct pipeline stage (Ingestion, GPU Extraction, Commercial Filtering, Structured Extraction, Multimodal, Vectorization). 
- **Rough size**: Small
- **Priority**: Must-have

#### 3. Hardware Resource Signatures
- **The pain**: We sandbox GPU operations to prevent VRAM fragmentation, but we don't actually know if we are redlining the GPU or leaving memory on the table.
- **The opportunity**: Capturing peak VRAM usage and average GPU utilization during the `GPU Extraction` phase for each specific episode.
- **Rough size**: Medium
- **Priority**: Should-have

#### 4. Surfacing the Data: Dynamic CLI Dashboard vs. Frontend Website
- **The pain**: Having incredible data locked in SQLite requires writing manual queries to understand pipeline health. It's opaque and lacks a "wow" factor.
- **The opportunity**: ML Engineers and Data Scientists live in their terminals. Instead of forcing them out of their workflow into a browser, we bring a rich, dynamic, hardware-accelerated TUI (Text User Interface) directly to their terminal. Think of `htop` but specifically for the Trebek ML pipeline. It visualizes Gemini token costs, WhisperX latencies, and self-healing retry metrics using beautiful ASCII sparklines and rich tables.
- **Rough size**: Large
- **Priority**: Must-have

### Recommendation
**Top Pick: Implement Pipeline Stage Latency + Gemini Telemetry surfaced via a Rich CLI Dashboard.**

*Why a CLI over a Frontend Website?*
In our `PHILOSOPHY.md`, we explicitly state under "What This Is NOT": Trebek is **NOT an API server** and it is a heavy-compute daemon pipeline. Introducing a frontend website violates this core principle by requiring us to spin up an HTTP server, manage web sockets, and deal with browser routing. Our target personas are ML Engineers and Data Scientists—people who live in the terminal. 

A gorgeous, dynamic CLI command (e.g., using Python's `Rich` or `Textual` libraries) provides that premium, indispensable experience without breaking our architectural invariant of being a standalone daemon. It respects their workflow—a quick `trebek stats` in the terminal is vastly superior to tabbing to a browser.

### Open Questions
- Do we want to store these stats in a new `job_telemetry` table, or should we expand the existing `pipeline_state` table to include these metrics?
- For Gemini Token Tracking, do we want to implement caching logic optimization based on these metrics, or purely observe for now?
