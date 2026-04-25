# Step-by-Step Task Plan: Job Telemetry & Caching

Based on the architectural exploration in `2026-04-24-product-visionary-job-stats.md` and the contracts in `schema.sql`, we need to implement telemetry tracking for hardware, pipeline latency, and Gemini API usage.

## Task List

### 1. Define `JobTelemetry` Pydantic Schema
- **File:** `src/schemas.py`
- **Action:** Create a new `JobTelemetry` Pydantic model.
- **Fields:**
  - `episode_id` (str)
  - `peak_vram_mb` (Optional[float])
  - `avg_gpu_utilization_pct` (Optional[float])
  - `stage_ingestion_ms` (Optional[float])
  - `stage_gpu_extraction_ms` (Optional[float])
  - `stage_commercial_filtering_ms` (Optional[float])
  - `stage_structured_extraction_ms` (Optional[float])
  - `stage_multimodal_ms` (Optional[float])
  - `stage_vectorization_ms` (Optional[float])
  - `gemini_total_input_tokens` (Optional[int])
  - `gemini_total_output_tokens` (Optional[int])
  - `gemini_total_cached_tokens` (Optional[int])
  - `gemini_total_cost_usd` (Optional[float])
  - `gemini_api_latency_ms` (Optional[float])
  - `pydantic_retry_count` (Optional[int])
- **Validation:** Ensure float values are positive or zero.

### 2. Implement Database Writer Method
- **File:** `src/core_database.py`
- **Action:** Add an `insert_job_telemetry(self, telemetry: JobTelemetry)` method to `DatabaseWriter`.
- **Query:** 
  ```sql
  INSERT INTO job_telemetry (
      episode_id, peak_vram_mb, avg_gpu_utilization_pct, ...
  ) VALUES (?, ?, ?, ...)
  ```

### 3. Pipeline Integration (LLM Metrics)
- **File:** `src/llm_pipeline.py`
- **Action:** Update the Google GenAI SDK calls to capture `usage_metadata` (token counts) and time the requests.
- **Action:** Update the Pydantic retry loops to increment a counter and log it to telemetry.
- **Action:** Add caching logic optimizations: calculate costs and potentially dynamically switch models or caching configurations based on token footprint.

### 4. Pipeline Integration (Latency & Hardware Metrics)
- **File:** `src/gpu_orchestrator.py` & `src/main.py`
- **Action:** Add timing logic (e.g., `time.perf_counter()`) around pipeline stages.
- **Action:** (Stretch) Integrate `pynvml` to sample GPU usage during the `stage_gpu_extraction_ms` phase.

### 5. CLI Rich Dashboard
- **File:** `src/cli.py` or new dashboard module
- **Action:** Implement a TUI using `Rich` to display the `job_telemetry` metrics, replacing manual query requirements.
