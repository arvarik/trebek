## 1. Local Development Setup
**Prerequisites:**
- Python 3.9+
- Local FFmpeg installation (for Stage 2/3 and Stage 6 frame extraction).
- Valid `.env` configuration for Gemini APIs.

**Commands:**
- **Install for Dev**: `pip install -e .[dev]`
- **Run Tests**: `pytest`
- **Run Linter**: `ruff check .`
- **Run Typechecker**: `mypy src/`

## 2. Execution Evidence Rules
**CRITICAL**: You must empirically verify changes before marking tasks complete.
1. Add a test case for any new feature or bug fix.
2. Run the specific test and ensure it passes.
3. Ensure the test fails if the feature is removed or broken (red/green validation).
4. Run the full test suite to ensure no regressions.

## 3. Empty Scenario Tables
### Feature Test Scenarios
| Scenario | Status | Evidence |
|----------|--------|----------|

### Regression Scenarios
| Scenario | Status | Evidence |
|----------|--------|----------|

## 4. Backend Route Coverage Matrix
*N/A — This is a daemon-based pipeline orchestrator and CLI processor, not an HTTP API.*

## 5. Frontend Component State Matrix
*N/A — No frontend interface.*

## 6. ML/AI Evaluation Thresholds
- **Pydantic LLM Self-Healing**: Pass 2 data extraction relies on strict Pydantic validation of the LLM JSON output. Implement a maximum of 2 retries passing the `ValidationError` string back to the prompt before failure.
- **Deterministic Disfluency Tracking**: Cross-reference WhisperX `.prob` bounds against Semantic outputs rather than allowing LLM to hallucinate disfluency counts.