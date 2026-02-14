# Justfile for FrameCloud project

# Default recipe to list all available recipes
default:
    @just --list

# Run all tests
test: _create_report_structure
    uv run pytest -n auto

test-slow:
    uv run pytest -n auto -m "slow"

# Internal: Create report directory structure
_create_report_structure:
    @mkdir -p reports/benchmarks

# Run benchmark tests and generate plots
benchmark: _create_report_structure
    uv run pytest -m "benchmark and not slow" --benchmark-only --benchmark-json=reports/benchmarks/benchmark.json
    @echo "\nGenerating custom benchmark plots..."
    uv run python scripts/benchmark_plot.py

# Run large/slow benchmark tests and merge with regular results if available
benchmark-large: _create_report_structure
    uv run pytest -m "slow and benchmark" --benchmark-only --benchmark-json=reports/benchmarks/benchmark-large.json
    @if [ -f reports/benchmarks/benchmark.json ]; then \
        echo "\nMerging regular and large benchmark results..."; \
        uv run python scripts/merge_benchmark_results.py \
            reports/benchmarks/benchmark.json \
            reports/benchmarks/benchmark-large.json \
            --output reports/benchmarks/benchmark.json; \
    else \
        cp reports/benchmarks/benchmark-large.json reports/benchmarks/benchmark.json; \
    fi
    @echo "\nGenerating custom benchmark plots..."
    uv run python scripts/benchmark_plot.py

# View benchmark histogram in reports/benchmarks/histogram/
benchmark-view:
    @echo "Benchmark reports are saved in:"
    @echo "  - JSON: reports/benchmarks/benchmark.json"
    @echo "  - Dashboard: reports/benchmarks/benchmark_dashboard.html"
    @echo "  - Individual SVG plots: reports/benchmarks/plots/*.svg"
    @ls -lh reports/benchmarks/plots/*.svg reports/benchmarks/*.html 2>/dev/null || echo "No benchmark reports found. Run 'just benchmark' first."

# Run linting with ruff
lint:
    uvx ruff check --fix src tests

# Check formatting with auto fix
format:
    uvx ruff format

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


tracer file:
    uv run viztracer {{file}} -o output/viztracer.html
