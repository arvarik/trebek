# Step 3 Build Tasks: Schema & Config Enforcement

This document defines the implementable task list based on the Architect's contracts from Step 1. These tasks map directly to the failing test suite trap laid in Step 2.

## 1. Config Validation (`src/config.py`)
- [x] Add a `field_validator` for `gpu_vram_target_gb` in `Settings` to ensure it is `>= 4` and `<= 24`.
- [x] Add a `field_validator` for `whisper_compute_type` to ensure it must be either `"float16"` or `"float32"`.
- [x] Add a `field_validator` for `whisper_batch_size` to ensure it is strictly positive (`> 0`).

## 2. Schema Integrity (`src/core_database.py`)
- [x] Ensure SQLite is instantiated with `PRAGMA foreign_keys = ON;` in `DatabaseWriter`.
- [x] Ensure `executemany` or similar repository methods are available for inserting complex records into the new normalized tables (`clues`, `buzz_attempts`, `wagers`, etc.).
- [x] Ensure database constraints (such as `podium_position IN (1, 2, 3)` and `round IN (...)`) are gracefully mapped to Python errors or handled appropriately in the repository layer.
