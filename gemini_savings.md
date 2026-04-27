# Gemini API Cost Efficiency Review

This document outlines a thorough review of the Gemini API usage within the `trebek` LLM pipeline, focusing on identifying cost-saving opportunities and evaluating the effectiveness of currently implemented optimizations.

## 1. Context Caching Anti-Pattern in Map-Reduce (Pass 2)

**Current Implementation:** 
In `trebek/llm/pass2_extraction.py` (lines 103-118), the pipeline utilizes Gemini Context Caching for the full Jeopardy transcript. This cached transcript is then queried for the initial Episode Meta extraction and subsequently for *every* individual chunk during the semantic chunking phase (lines 162-187).

**The Inefficiency:** 
Context Caching is highly effective when a large, static context is queried multiple times with distinct prompts, provided the total volume of inline prompts would exceed the cache creation + read costs. However, in this map-reduce pattern:
- Cache creation is billed at 1x the regular token rate.
- Each chunk query against the cache is billed at the cached read rate (typically ~0.25x) for the *entire transcript length*, not just the chunk length.
- **Estimate:** A typical Jeopardy transcript is approximately 15,000 tokens. If the semantic chunker splits it into 6 chunks:
  - **With Caching:** 15,000 (creation) + 6 * (0.25 * 15,000) = 15,000 + 22,500 = **37,500 equivalent tokens** billed.
  - **Without Caching (Inline):** We pass each chunk individually. 6 chunks * ~2,500 tokens/chunk = **15,000 equivalent tokens** billed.
  - The caching approach is mathematically ~2.5x *more expensive* than simply passing the chunks inline.
- Furthermore, removing the full transcript from the context window during chunk extraction reduces the risk of the model hallucinating clues from other parts of the episode (a common issue in long-context extraction).

**Step-by-Step Fix:**
1. In `trebek/llm/pass2_extraction.py`, remove the `client.create_cache` block.
2. Remove the `caching_active` boolean branches.
3. Update `extract_chunk` to unconditionally construct the prompt using `chunk_text` (the current "No cache" branch).
4. Remove the `try...finally` block that cleans up the cache at the end of the script.

## 2. LLM-Based Schema Repair on Retries

**Current Implementation:**
The pipeline employs a robust retry mechanism in `trebek/llm/utils.py` (`_extract_part` function). If the output fails Pydantic `schema_cls.model_validate_json` validation, it simply continues the loop and re-runs the entire extraction prompt.

**The Inefficiency:**
When a chunk fails schema validation or JSON parsing (often due to a trailing comma, missed quote, or slight schema deviation), the entire chunk text (thousands of tokens) is re-sent to the model in the next attempt. This multiplies the input token cost for failures, especially when using the expensive `gemini-3.1-pro-preview` model.
- **Estimate:** Retrying a 3,000 token chunk with `Pro` costs ~0.00375 USD per attempt. A repair prompt using `Flash` with just the broken JSON string (~500 tokens) costs ~0.0000375 USD—almost exactly 100x cheaper.

**Step-by-Step Fix:**
1. In `trebek/llm/utils.py`, modify `_extract_part`. Wrap `schema_cls.model_validate_json` in a `try...except` block catching `ValueError` or Pydantic validation errors.
2. If an error is caught, instead of immediately retrying the full `generate_content` call, invoke a new helper function: `_repair_json(broken_json_str, error_message)`.
3. `_repair_json` should call `gemini-3-flash-preview` (regardless of the main model being used) with a simple prompt: "Fix the following JSON to adhere to strict JSON standards and resolve this error: {error}. Output ONLY valid JSON."
4. Attempt validation on the repaired JSON. If it passes, return the object. Only if the repair *also* fails should the system burn tokens on a full retry of the chunk.

## 3. Multimodal API Call Batching (Pass 3)

**Current Implementation:**
In `trebek/llm/pass3_multimodal.py`, "Temporal Sniping" is used to extract a precise 3-second video clip for each clue requiring visual context using `ffmpeg`. Each clip is uploaded and evaluated independently via `client.upload_file` and `client.generate_content`.

**The Inefficiency:**
While extracting 3-second clips is vastly cheaper than uploading the entire 30-minute episode, evaluating clues independently means overlapping or adjacent clues (common in rapid-fire Double Jeopardy categories) result in multiple API calls, overlapping video processing, and redundant upload overhead.
- **Estimate:** 5 sequential visual clues equal 5 separate FFmpeg spawns, 5 file uploads, and 5 network requests. Coalescing them into a single 15-second clip requires 1 FFmpeg spawn, 1 upload, and 1 request. While token count (video tokens/sec) remains similar, the wall-clock time and API request volume drop by 80%.

**Step-by-Step Fix:**
1. In `trebek/llm/pass3_multimodal.py`, pre-process `episode.clues` before executing `process_clue_multimodal`.
2. Group clues that have `host_finish_timestamp_ms` within a small window (e.g., 5-10 seconds) of each other.
3. For each group, calculate a bounding `start_time` and `duration` covering all clues in the group.
4. Run FFmpeg to extract this single, slightly longer clip.
5. Modify the system prompt to ask the model to return a JSON array of timestamps corresponding to the multiple buzz events it observes.
6. Map the returned array of timestamps back to the clues in the group based on their temporal order.

## 4. CLI Model Flag Verification

**Current Implementation:**
The pipeline defaults to `gemini-3.1-pro-preview` (Pro) for maximum accuracy. A `--model flash` flag is provided in the CLI (`trebek/cli.py`) to allow users to opt into the cheaper `gemini-3-flash-preview` model for testing or cost-sensitive runs.

**Findings & The Inefficiency:**
- The `--model` flag is correctly parsed in `cli.py` and successfully propagates through `orchestrator.py` into `execute_pass_2_data_extraction()`.
- **Bug Identified:** Pass 3 currently ignores this flag. In `trebek/llm/pass3_multimodal.py:76`, the `generate_content` call is hardcoded to `model="gemini-3.1-pro-preview"`. 
- **Estimate:** Users running `trebek run --model flash` expect cost savings across the board. If Pass 3 executes 10 visual checks, it secretly uses Pro, incurring ~16x higher input costs than Flash for those video tokens.

**Step-by-Step Fix:**
1. In `trebek/llm/pass3_multimodal.py`, update the signature of `execute_pass_3_multimodal_augmentation` to accept a `model: str = MODEL_PRO` parameter.
2. On line 76, change `model="gemini-3.1-pro-preview"` to `model=model`.
3. In `trebek/pipeline/workers/llm.py` (around line 125, inside the Pass 3 branch), update the call to `execute_pass_3_multimodal_augmentation` to pass `model=orchestrator.llm_model`.
4. Ensure `MODEL_PRO` is imported from `trebek.config` in `pass3_multimodal.py`.