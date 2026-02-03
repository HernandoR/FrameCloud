# Git Workflow & Commit Guidelines

## Pre-Commit Checks

Before committing code, ensure the following checks pass:

- **Format**: `uvx ruff format` (auto-formats code)
- **Lint**: `uvx ruff check --fix` (fixes linting issues)
- **Type Check**: `uvx ty check` (verifies type hints)
- **Tests**: `uv run pytest` (ensures tests pass)

Alternatively, use the automated check:

```bash
uvx pre-commit run --all-files
```

## Branch Naming

- **Feature**: `feat/<feature-name>` (e.g., `feat/add-point-cloud-loader`)
- **Bug Fix**: `fix/<issue-description>` (e.g., `fix/memory-leak-in-io`)
- **Documentation**: `docs/<topic>` (e.g., `docs/api-reference`)
- **Refactor**: `refactor/<area>` (e.g., `refactor/np-module`)

## Commit Message Format

Follow the commit message convention defined in `.copilot-commit-message-instructions.md`. In brief:

- **Format**: `<type>(<scope>): <subject>` with optional body and footer
- **Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- **Language**: Write commit messages in Chinese
- **Length**: No line should exceed 72 characters

## Pull Requests

- Reference related issues in the PR description
- Ensure all checks pass before requesting review
- Keep PRs focused on a single concern
- Provide clear description of changes and motivation
