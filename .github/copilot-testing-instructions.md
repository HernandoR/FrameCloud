# Testing Guidelines

## Test Organization

- **Location**: All tests are in the `tests/` directory.
- **File Naming**: Test files should be named `test_<module_name>.py` to match the modules they test.
- **Fixtures**: Define shared fixtures in `conftest.py` for reuse across test files.

## Test Writing

- **Framework**: Use `pytest` for all testing.
- **Coverage**: Aim for meaningful test coverage; run `uv run pytest` to verify.
- **Test Naming**: Use descriptive names starting with `test_` that clearly indicate what is being tested.
- **Assertions**: Use clear, specific assertions. Avoid generic assertions.

## Running Tests

- **Basic**: `uv run pytest` (run all tests)
- **Specific**: `uv run pytest tests/test_<module>.py` (run specific file)
- **With Coverage**: `uv run pytest --cov` (generate coverage reports)
- **Watch Mode**: Use `uv run pytest --watch` for test-driven development

## Test Best Practices

- Write tests alongside implementation when possible
- Use fixtures for common setup/teardown
- Mock external dependencies (APIs, file I/O) when appropriate
- Test both success and failure cases
- Avoid hardcoding test data; use parameterized tests for multiple scenarios
