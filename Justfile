# Justfile for FrameCloud project

# Default recipe to list all available recipes
default:
    @just --list

# Run all tests
test:
    uv run pytest

# Run only unit tests (excluding benchmarks)
test-unit:
    uv run pytest -m "not benchmark" -v

# Run only benchmark tests
test-benchmark:
    uv run pytest -m benchmark -v --durations=0

# Run tests for np package only
test-np:
    uv run pytest tests/test_point_cloud.py tests/test_point_cloud_io.py -v

# Run tests for pd package only
test-pd:
    uv run pytest tests/test_pd_point_cloud.py tests/test_pd_point_cloud_io.py -v

# Run tests with coverage report
test-coverage:
    uv run pytest --cov=framecloud --cov-report=term-missing --cov-report=html

# Run linting with ruff
lint:
    uvx ruff check src tests

# Fix linting issues automatically
lint-fix:
    uvx ruff check --fix src tests

# Format code with ruff
format:
    uvx ruff format src tests

# Check formatting without making changes
format-check:
    uvx ruff format --check src tests

# Type check with ty
type-check:
    uvx ty check

# Run all quality checks (lint, format-check, type-check)
check: lint format-check type-check

# Fix all auto-fixable issues
fix: lint-fix format

# Clean up generated files
clean:
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf reports
    rm -rf htmlcov
    rm -rf .coverage
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Run quick tests (small subset for rapid iteration)
test-quick:
    uv run pytest tests/test_pd_point_cloud.py::TestPointCloudInitialization -v
    uv run pytest tests/test_point_cloud.py::TestPointCloudInitialization -v

# Run tests in parallel (requires pytest-xdist)
test-parallel:
    uv run pytest -n auto

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
    @echo "  - test-unit: Run unit tests only"
    @echo "  - test-benchmark: Run benchmark tests only"
    @echo "  - test-np: Run numpy package tests"
    @echo "  - test-pd: Run pandas package tests"
    @echo ""
    @echo "Quality checks:"
    @echo "  - lint: Run linter"
    @echo "  - format: Format code"
    @echo "  - type-check: Run type checker"
    @echo "  - check: Run all quality checks"
