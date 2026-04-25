## 1. Code Structuring Patterns
- **Typing**: Use strict type hints (`mypy strict = true`).
- **Data Contracts**: Use Pydantic v2 `BaseModel` and `Field` for all complex inputs, schemas, and LLM structured outputs.
- **Logging**: Rely entirely on `structlog` for structured, JSON-ready, timestamped logging (`structlog.get_logger()`).

## 2. Formatting Conventions
- **Linter**: `ruff` (also enforces import ordering)
- **Line Length**: 120 characters
- **Target Version**: Python 3.9
- **Naming**: 
  - Python files: `snake_case.py`
  - Variables and Functions: `snake_case`
  - Classes: `PascalCase`
- **Docstrings**: Use descriptive docstrings for all pipeline stages and database interaction logic.

## 3. UI/Styling Tokens
*N/A — This is a backend daemon pipeline, no frontend interface exists.*

## 4. Anti-Patterns (FORBIDDEN)
- **NEVER** use `asyncio.Queue` passing between the GPU stages and the LLM stages. The SQLite `pipeline_state` table MUST act as the persistent queue.
- **NEVER** rely on PyTorch's custom caching memory allocator inside a long-running thread (`torch.cuda.empty_cache()` does not defragment VRAM). Always use a dedicated, ephemeral OS subprocess.
- **NEVER** pass massive extracted data arrays back to the orchestrator via IPC pickling. Write massive outputs to compressed disk files and return the string path.
- **NEVER** execute concurrent writes directly against the SQLite database. ALL writes MUST be routed through the `DatabaseWriter` actor queue.
- **NEVER** instruct the LLM to perform running score math or calculate wagers. LLMs only extract facts; the `TrebekStateMachine` calculates math.
- **NEVER** block the main `asyncio` event loop with CPU-heavy parsing (like `model_validate_json`).