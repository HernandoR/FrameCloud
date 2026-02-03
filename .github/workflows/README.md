# CI/CD Workflows

This directory contains GitHub Actions workflows for continuous integration and testing.

## Workflows Overview

### 1. Fast Tests - Ubuntu (`test-fast-ubuntu.yml`)
- **Purpose**: Run unit tests (excluding benchmarks) across multiple Python versions on Ubuntu
- **Python Versions**: 3.10, 3.11, 3.12, 3.13, 3.14t (free-threaded)
- **Triggers**: 
  - Push to `main` branch
  - Push to release branches (`release/**`, `release-*`)
  - Push to version tags (`v*`, `release*`)
  - Pull requests
  - Manual dispatch
- **Test Command**: `just test-unit`
- **Duration**: ~2-5 minutes

### 2. Slow Tests - Ubuntu (`test-slow-ubuntu.yml`)
- **Purpose**: Run benchmark tests with large datasets (10-100M points)
- **Python Version**: 3.12 (stable)
- **Triggers**:
  - Push to `main` branch
  - Push to release branches (`release/**`, `release-*`)
  - Push to version tags (`v*`, `release*`)
  - Manual dispatch
- **Test Command**: `just test-benchmark`
- **Duration**: ~10-30 minutes (depends on dataset size)

### 3. Fast Tests - Cross Platform (`test-fast-cross-platform.yml`)
- **Purpose**: Run unit tests on macOS and Windows
- **Python Version**: 3.12 (stable)
- **Platforms**: macOS-latest, Windows-latest
- **Triggers**:
  - Push to `main` branch
  - Push to release branches (`release/**`, `release-*`)
  - Push to version tags (`v*`, `release*`)
  - Pull requests
  - Manual dispatch
- **Test Command**: `just test-unit`
- **Duration**: ~3-7 minutes per platform

### 4. Slow Tests - Cross Platform (`test-slow-cross-platform.yml`)
- **Purpose**: Run benchmark tests on macOS and Windows
- **Python Version**: 3.12 (stable)
- **Platforms**: macOS-latest, Windows-latest
- **Triggers**:
  - Push to `main` branch
  - Push to release branches (`release/**`, `release-*`)
  - Push to version tags (`v*`, `release*`)
  - Manual dispatch
- **Test Command**: `just test-benchmark`
- **Duration**: ~15-45 minutes per platform

### 5. Quality Checks (`quality-checks.yml`)
- **Purpose**: Lint, format check, and type check the codebase
- **Python Version**: 3.12
- **Triggers**:
  - Push to `main` branch
  - Push to release branches (`release/**`, `release-*`)
  - Push to version tags (`v*`, `release*`)
  - Pull requests
  - Manual dispatch
- **Commands**: `just check` (runs lint, format-check, type-check)
- **Duration**: ~1-2 minutes

## Justfile Recipes Used

The workflows use the following recipes from `Justfile`:

- `test-unit`: Run unit tests excluding benchmarks (`pytest -m "not benchmark"`)
- `test-benchmark`: Run only benchmark tests (`pytest -m benchmark`)
- `check`: Run all quality checks (lint + format-check + type-check)

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.benchmark`: Marks tests that run with large datasets and take longer
- Tests without markers: Fast unit tests

## Running Tests Locally

```bash
# Fast tests (unit tests)
just test-unit

# Slow tests (benchmarks)
just test-benchmark

# All tests
just test

# Quality checks
just check
```

## Path Triggers

All workflows are configured to run only when relevant files change:
- Source code: `src/**`
- Tests: `tests/**`
- Dependencies: `pyproject.toml`, `uv.lock`
- Build config: `Justfile`, workflow files themselves

This helps conserve CI resources and speeds up feedback loops.

## Manual Triggering

All workflows support manual triggering via `workflow_dispatch` from the GitHub Actions tab.

## Setup Steps Reference

The workflows follow the setup pattern from `copilot-setup-steps.yml`:
1. Checkout code with actions/checkout@v4
2. Install uv with astral-sh/setup-uv@v6
3. Install Python version with `uv python install`
4. Install just with `uv tool install rust-just`
5. Install dependencies with `uv sync --dev`
6. Run tests/checks using just recipes
