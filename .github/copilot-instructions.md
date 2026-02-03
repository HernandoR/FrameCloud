# Copilot Instructions for Python Development

This project uses a modern, high-performance Python toolchain. Please adhere to the following guidelines and tool usages when generating code or commands.

## Toolchain & Workflow

### Dependency & Project Management: `uv`

- **Primary Tool**: Use `uv` for all project management tasks.
- **Overview**: `uv` is an extremely fast Python package and project manager written in Rust. It effectively replaces `pip`, `poetry`, and `pip-tools`.
- **Commands**:
  - Add dependencies: `uv add <package>`
  - Add dev dependencies: `uv add --dev <package>`
  - Run scripts: `uv run <script.py>` or `uv run -m <module>`
  - Sync environment: `uv sync` (creates/updates `.venv`)
  - Lock dependencies: `uv lock`
- **Configuration**: Managed in `pyproject.toml`.

### Linting & Formatting: `ruff`

- **Linter & Formatter**: Use `ruff` for all linting and formatting.
- **Overview**: `ruff` is an extremely fast Python linter and code formatter written in Rust. It replaces tools like Black, Flake8, and isort.
- **Commands**:
  - Check: `uvx ruff check` (fix with `--fix`)
  - Format: `uvx ruff format`
- **Style**: Code should be compliant with `ruff`'s default rules and any overrides in `pyproject.toml` or `ruff.toml`.

### Type Checking: `ty`

- **Type Checker**: Use `ty` for static type checking.
- **Overview**: `ty` is a fast Python type checker and language server found in the Astral ecosystem.
- **Commands**:
  - Check: `uvx ty check`
- **Typing**: Ensure all code is fully typed. Use modern Python type hinting features.

### Task Management: `just`

- **Task Runner**: Uses `just` to run project-specific tasks.
- **Overview**: `just` is a handy command runner that saves and runs project-specific commands (recipes) stored in a `Justfile`.
- **Usage**: Check the `Justfile` (if present) for available recipes.
- **Recommendation**: If users ask to run common tasks (build, test, lint), check if a `just` recipe exists.
- **installation**: `just` can be installed via package managers like `brew`, `apt`, or using uv:
  ```bash
  uv tool install rust-just
  ```

### Git Hooks: `pre-commit`

- **Automation**: `pre-commit` ensures code quality before committing.
- **Overview**: Framework for managing and maintaining multi-language pre-commit hooks.
- **Commands**:
  - Run manually: `uvx pre-commit run --all-files`
- **installation**:
  ```bash
  uv tool install pre-commit
  pre-commit install
  ```

### Build System: `hatchling`

- **Backend**: The project uses `hatchling` as the build backend.
- **Overview**: `hatchling` is a modern, extensible build backend for Python projects.
- **Configuration**: Build settings are defined in the `[build-system]` section of `pyproject.toml`.

### Testing: `pytest`

- **Framework**: Use `pytest` for testing.
- **Overview**: The pytest framework makes it easy to write small, readable tests, and can scale to support complex functional testing.
- **Commands**:
  - Run tests: `uv run pytest`
- **Practices**: Write unit tests for new functionality in the `tests/` directory.

## Commit Messages

- **Strict Requirement**: You **MUST** reference and follow the commit message conventions defined in `.copilot-commit-message-instructions.md` located in the project root.
- All generated commit messages should adhere to the format and rules specified in that file.

## AI-Generated Files

- **Location**: All AI-generated files (documentation, analysis, code summaries, etc.) should be placed in the `agc/` directory.
- **Naming Convention**: Use the format `YYMMDD_idx_<description>.<ext>`, where:
  - `YYMMDD` is the date in year-month-day format (e.g., `20260202`)
  - `idx` is a sequential index starting from `01` (e.g., `01`, `02`, `03`)
  - `<description>` is a brief descriptive name
  - `<ext>` is the file extension (e.g., `.md`, `.txt`)
- **Example**: `agc/20260202_01_implementation_summary.md`, `agc/20260202_03_protocol_architecture.md`

## Additional Guidelines

For detailed guidance on specific topics, refer to these supplementary instruction files:

- **Code Style & Best Practices**: See [python-code-style-instructions.md](python-code-style-instructions.md) for docstrings, path handling, logging, and type hinting standards.
- **Testing**: See [copilot-testing-instructions.md](copilot-testing-instructions.md) for test organization, running tests, and best practices.
- **Git Workflow & Commits**: See [copilot-git-workflow-instructions.md](copilot-git-workflow-instructions.md) for pre-commit checks, branch naming, and commit message guidelines.
- **Project Patterns & Architecture**: See [copilot-project-patterns-instructions.md](copilot-project-patterns-instructions.md) for module structure, protocols, and I/O utilities.
- **Error Handling & Logging**: See [copilot-error-handling-instructions.md](copilot-error-handling-instructions.md) for exception handling and logging practices.

## General Development Guidelines

- **Modern Python**: Use modern Python features (3.10+).
- **Type Safety**: Prioritize type safety and clarity.
- **Performance**: Leverage the speed of the provided Rust-based tools (`uv`, `ruff`, `ty`).
