# Justfile for FrameCloud project

# Default recipe to list all available recipes
default:
    @just --list

# Run all tests
test:
    uv run pytest

test-slow:
    uv run pytest --runslow

# Internal: Create report directory structure
_create_report_structure:
    @mkdir -p reports/benchmarks

# Run benchmark tests (excludes slow tests by default)
benchmark: _create_report_structure
    uv run pytest tests/test_benchmark.py -m "benchmark and not slow" --benchmark-only --benchmark-autosave --benchmark-storage=reports/benchmarks --benchmark-json=reports/benchmarks/benchmark.json --benchmark-histogram=reports/benchmarks/histogram

# Run all benchmarks including slow/large-scale tests
benchmark-all: _create_report_structure
    uv run pytest tests/test_benchmark.py -m benchmark --benchmark-only --benchmark-autosave --benchmark-storage=reports/benchmarks --benchmark-json=reports/benchmarks/benchmark.json --benchmark-histogram=reports/benchmarks/histogram

# Run benchmarks and compare with previous results
benchmark-compare: _create_report_structure
    uv run pytest tests/test_benchmark.py -m "benchmark and not slow" --benchmark-only --benchmark-autosave --benchmark-storage=reports/benchmarks --benchmark-json=reports/benchmarks/benchmark.json --benchmark-histogram=reports/benchmarks/histogram --benchmark-compare

# Run benchmarks and save baseline for future comparisons
benchmark-save: _create_report_structure
    uv run pytest tests/test_benchmark.py -m "benchmark and not slow" --benchmark-only --benchmark-autosave --benchmark-storage=reports/benchmarks --benchmark-json=reports/benchmarks/benchmark.json --benchmark-histogram=reports/benchmarks/histogram --benchmark-save=baseline

# View benchmark histogram in reports/benchmarks/histogram/
benchmark-view:
    @echo "Benchmark reports are saved in:"
    @echo "  - JSON: reports/benchmarks/benchmark.json"
    @echo "  - Histogram: reports/benchmarks/histogram/"
    @ls -lh reports/benchmarks/ 2>/dev/null || echo "No benchmark reports found. Run 'just benchmark' first."

# Run tests in parallel (requires pytest-xdist)
test-parallel:
    uv run pytest -n auto

# Run linting with ruff
lint:
    uvx ruff check --fix src tests

# Check formatting without making changes
format:
    uvx ruff format --check src tests

# Type check with ty
type-check: install
    uvx ty check

# Run all quality checks (lint, format, type-check)
check: lint format type-check
# Fix all auto-fixable issues
fix: lint format

# Clean up generated files
clean:
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf reports
    rm -rf htmlcov
    rm -rf .coverage
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Install all dependencies
install:
    uv sync

# Update dependencies
update:
    uv lock --upgrade

# Show project info
info:
    @echo "FrameCloud - Point Cloud Processing Library"
    @echo "============================================"
    @echo "Available test suites:"
    @echo "  - test: Run all tests"
    @echo "  - test-slow: Run slow tests only"
    @echo ""
    @echo "Quality checks:"
    @echo "  - lint: Run linter"
    @echo "  - format: Format code"
    @echo "  - type-check: Run type checker"
    @echo "  - check: Run all quality checks"
