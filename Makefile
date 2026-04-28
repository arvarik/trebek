.PHONY: test lint typecheck format all dev clean

# Run the full quality gate: lint → typecheck → test
all: lint typecheck test

# Run tests with coverage enforcement
test:
	pytest tests/ -v --tb=short --cov=trebek --cov-report=term-missing --cov-fail-under=50

# Lint with ruff
lint:
	ruff check .

# Type check with mypy
typecheck:
	mypy trebek/

# Auto-format with ruff
format:
	ruff check --fix .
	ruff format .

# Install in editable mode with dev dependencies
dev:
	pip install -e ".[dev]"
	pre-commit install

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info trebek/*.egg-info trebek/trebek.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
