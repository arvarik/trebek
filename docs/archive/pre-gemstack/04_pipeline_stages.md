# Pipeline Stages

## 1. Stage 1: Ingestion & State Tracking
* **Trigger:** The orchestrator discovers a new video file.
* **Action:** The file's hash, path, and filename are pushed via the `DatabaseWriter` task into the `pipeline_state` table with status `PENDING`.
* **Flow:** The `Extractor` daemon polls for `PENDING` files.

## 2. Stage 2: Dispatch to GPU Worker
* **Action:** The main orchestrator dispatches only the lightweight `video_filepath` string to the GPU worker. The heavy I/O of extracting audio via `ffmpeg` into a `numpy.float32` array is deferred to the subprocess to completely avoid Python IPC pickling overhead.
* **State Update:** Database updated to `TRANSCRIBING`.

## 3. Stage 3: Local GPU Transcription (Hardware Locked)
* **Action:** The `video_filepath` is passed to a `ProcessPoolExecutor(max_workers=1, mp_context='spawn')`.
* **Execution:**
  1. The isolated worker process executes `ffmpeg` to extract the audio into its own memory space.
  2. WhisperX (`large-v3`, `float16`) generates the base transcript and timestamps.
  3. Word alignment is performed.
  4. `pyannote.audio` diarization generates speaker labels.
* **Finalization:** The `ProcessPool` writes the payload to a local compressed file (`.json.gz`), and returns *only the filepath string* back to the orchestrator via IPC. Because the subprocess dies, 100% of the VRAM (10-11GB) is instantly released to the OS.
* **State Update:** The `DatabaseWriter` persists the `.json.gz` reference to the database and updates the state to `TRANSCRIPT_READY`. 
* *Crucially: The GPU is now free to immediately begin processing the next file, regardless of upstream LLM speeds.*

## 4. Stage 4: Commercial Filtering (Metadata Preserving Pass)
* **Trigger:** The `LLMWorker` polls the database for `TRANSCRIPT_READY`.
* **Action:** Pulls the raw JSON transcript from SQLite and feeds it to **Gemini 3.1 Flash** as a decision engine.
* **Prompting:** Instructs Gemini Flash to output ONLY a JSON array of commercial bounding boxes (e.g., `{"commercial_breaks": [{"start": 405.2, "end": 620.1}]}`).
* **Execution:** A Python script programmatically slices those time-blocks out of the original WhisperX JSON. This perfectly preserves WhisperX's precise word-level timestamps and rigid `SPEAKER_XX` diarization tags.
* **State Update:** The sliced JSON transcript replaces the raw transcript in the database. State updated to `CLEANED`.

## 5. Stage 5: Structured Extraction (LLM Pass 2)
* **Trigger:** The `LLMWorker` polls the database for `CLEANED`.
* **Action (Pass 1 - Speaker Anchoring):** A fast Gemini Flash pass isolates the host's interview segment to generate a rigid `{SPEAKER_XX: "Name"}` mapping dictionary.
* **Action (Pass 2 - Extraction):** Feeds the Clean Transcript to **Gemini 3.1 Pro** using the locked speaker mapping injected into the System Prompt.
* **Self-Healing LLM Loop:** Uses a `try/except ValidationError` loop for Pydantic validation. If validation fails, feeds the specific error string back to Gemini with a strict prompt to correct the JSON schema (max 2 retries).
* **Concurrency:** The heavy Pydantic validation of the return payload is executed via `asyncio.to_thread()`.
* **Rate Limits:** If Gemini throws a 429 Error, the `LLMWorker` `asyncio.sleep()`s using exponential backoff. Because the GPU uses the database as a queue, the system does not deadlock.
* **State Update:** If the extracted payload flags any clue with `requires_visual_context=True`, the database state is updated to `VISUAL_EXTRACTION_PENDING`. Otherwise, it is updated to `SAVING`.

## 6. Stage 6: Multimodal Visual Clue Augmentation (LLM Pass 3)
* **Trigger:** A dedicated `MultimodalWorker` polls the database for `VISUAL_EXTRACTION_PENDING`.
* **Action:** For every clue where `requires_visual_context=True`, the orchestrator uses `ffmpeg` to extract a single video frame at the `host_finish_timestamp`.
* **Execution:** The image frame, along with the incomplete clue text and category, is passed into **Gemini 3.1 Pro** (Multimodal). Gemini dynamically reads the visual trivia and accurately reconstructs the full `clue_text`. Additionally, the model analyzes the video frames immediately following the `host_finish_timestamp` to detect the exact frame the "podium indicator lights" illuminate for True Visual Buzzer Latency calculations. 
* **State Update:** Database updated to `SAVING`.

## 7. Stage 7: Deterministic Game-State Verification (State Machine)
* **Trigger:** The orchestrator polls for `SAVING`.
* **Action:** The LLM output only contains atomic events. A strict Python State Machine replays every event chronologically to calculate running scores perfectly (Score = Score +/- Clue Value), and strictly sequences `score_adjustments` at their respective chronological anchors. If a 'True Daily Double' string literal is encountered, the state machine intercepts it and dynamically applies the contestant's exact running score. If a Daily Double or Final Jeopardy wager is encountered, the exact score state is passed to a Minimax/Nash Equilibrium algorithm to calculate the `game_theory_optimal_wager` and the resulting `wager_irrationality_delta`.
* **State Update:** State remains `SAVING`, transitioning seamlessly into relational insertion.

## 8. Stage 8: Finalization & Relational Storage
* **Trigger:** Completed State Machine verification.
* **Action:** The validated and verified object is deconstructed. The `DatabaseWriter` executes relational `INSERT`s into the `jeopardy_data` tables (e.g., `Episodes`, `Contestants`, `Clues`, `Attempts`). True Buzzer Latency is calculated using `buzz_timestamp - podium_light_timestamp_ms`.
* **State Update:** Status is marked `VECTORIZING`.
* **Cleanup:** The intermediate `TEXT` transcript blob is set to `NULL`. Because SQLite is initialized with `PRAGMA auto_vacuum = INCREMENTAL;` (alongside a background vacuuming task), it safely reclaims dead space over time without stalling concurrent pipeline operations.

## 9. Stage 9: Semantic Board Search (RAG via sqlite-vec)
* **Trigger:** A dedicated `VectorWorker` polls for `VECTORIZING`.
* **Action:** Reads the newly inserted Clues and Contestant data.
* **Execution:** Generates two separate embeddings: `clue_embedding` (Category + Clue) and `response_embedding` (Correct Response) via a local embedding model (or Google Text Embedding API) and computes the `semantic_lateral_distance`.
* **Storage:** Inserts the vector embeddings into `sqlite-vec` virtual tables, creating an instant Retrieval-Augmented Generation (RAG) search engine for semantic clue queries.
* **State Update:** State updated to `COMPLETED`. The source video file is optionally moved to an archive directory.