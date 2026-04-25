## 1. Core Purpose
Trebek is a high-fidelity multimodal pipeline designed to extract the cognitive, temporal, and spatiotemporal "Running State" of Jeopardy! episodes into an Event-Sourced Relational Model. It turns unstructured game show video into deterministic arrays of reaction times, wager irrationality, and semantic board strategies for predictive ML modeling.

## 2. Target Persona
ML Engineers, Data Scientists, and researchers requiring a massive, surgically clean dataset of game theory and human behavioral physics under televised pressure.

## 3. Core Beliefs
- **Database-Driven State Machine over Memory**: True resumability and crash immunity are paramount. Zero data loss during multi-day inference runs requires database-backed queueing, not fragile in-memory `asyncio` queues.
- **Deterministic Math over LLM Approximations**: LLMs are hallucination-prone when doing arithmetic. They must extract pure facts; deterministic Python State Machines must execute the score tracking and Game Theory optimal wager calculations.
- **Hardware Isolation is Safety**: VRAM fragmentation is inevitable in long-running processes. Forceful memory reclamation via ephemeral subprocesses (`max_tasks_per_child=1`) guarantees stability.

## 4. What This Is NOT
- **NOT a real-time application**: This is a batch-processing, heavy-compute daemon pipeline, not an interactive or real-time streaming service.
- **NOT an API server**: It operates via filesystem polling (`watchdog`) and SQLite state management, not over HTTP endpoints.
- **NOT a keyword matcher**: The dataset relies heavily on vectorized embeddings (`sqlite-vec`) for semantic evaluation of clues, isolating "wordplay" from direct factual recall.