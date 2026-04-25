# Hardware & Concurrency Management

## 1. Defeating VRAM Fragmentation
A critical failure vector in long-running 24/7 AI pipelines is GPU memory fragmentation. Relying on PyTorch's custom caching memory allocator inside a long-running thread is fatal; `torch.cuda.empty_cache()` does not defragment VRAM.

### The `ProcessPoolExecutor` Solution
To guarantee our 16GB VRAM limit is never breached:
* Stage 3 (WhisperX & Pyannote) operates inside a `concurrent.futures.ProcessPoolExecutor` explicitly capped at `max_workers=1`.
* The pool must be initialized using the `spawn` multiprocessing context (`mp.get_context('spawn')`).
* **Hardware Lock:** We explicitly use `maxtasksperchild=1` (Python 3.11+) to guarantee worker death and VRAM reclamation. By executing the transcription pipeline inside a dedicated, ephemeral OS subprocess, we guarantee that when the transcript is returned and the subprocess dies, the OS flawlessly and forcefully reclaims 100% of the VRAM.
* **Zombie Process Safeguards:** An `atexit` or `signal` handler in the main orchestrator sends `SIGKILL` to subprocesses if the main event loop crashes, making the pipeline immune to PyTorch memory fragmentation over thousands of episodes.

## 2. Protecting the GIL (Global Interpreter Lock)
`asyncio` provides extreme concurrency for I/O bounds, but it immediately stalls if given heavy CPU tasks.

### A. FFmpeg Byte Conversion (Eliminating IPC Serialization Bottleneck)
Extracting 20+ minutes of audio into a ~85MB `numpy.float32` array in the main asyncio process creates a massive IPC (Inter-Process Communication) penalty because Python's `spawn` context must pickle the array to pass it to the GPU worker.
* **Solution:** Move the I/O entirely into the subprocess. The main async orchestrator passes only the `video_filepath` string to the `ProcessPoolExecutor`. The isolated worker process runs `ffmpeg` to extract the audio into its own memory space, runs WhisperX, writes the massive JSON payload to a local compressed file (`.json.gz`), and returns *only the filepath string* back to the orchestrator via IPC. The worker then dies. This completely eliminates the IPC pickling bottleneck for massive JSON dictionaries.

### B. Massive Pydantic Validation
Parsing and validating the massive Gemini JSON payload (containing 60+ clues, multiple contestants, scores, and timestamps) via `model_validate_json` will stall the asyncio event loop, causing dropped `watchdog` events and heartbeat timeouts.
* **Solution:** The Pydantic validation phase is explicitly offloaded from the main event loop.
```python
# Offloads CPU-bound JSON schema validation
structured_data = await asyncio.to_thread(EpisodeSchema.model_validate_json, gemini_response)
```

## 3. Resolving Thread-to-Async Deadlocks
A massive flaw in traditional hybrid architectures is connecting a synchronous thread (like our GPU process) directly to an `asyncio.Queue`. If the async queue hits its `maxsize` (e.g., due to an API rate limit), the thread blocks indefinitely when attempting to `.put()` its payload.

**Resolution:** As detailed in the Architecture spec, we completely decouple the synchronization via the SQLite database. The GPU process writes its result to a local `.json.gz` file, returns the filepath reference via IPC, and cleanly exits. The Orchestrator then writes this reference to the database. It never waits on the downstream LLM processing speeds, thus completely preventing deadlocks and keeping the GPU utilization at 100%.