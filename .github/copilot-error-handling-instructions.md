# Error Handling & Logging

## Using Project Exceptions

- **Location**: Custom exceptions are defined in `src/framecloud/exceptions.py`
- **Usage**: Import and use project-specific exceptions instead of built-in exceptions
- **Example**:

  ```python
  from framecloud.exceptions import InvalidPointCloudError

  if not is_valid(data):
      raise InvalidPointCloudError("Invalid point cloud format")
  ```

## Logging

- **Logger**: Always use `loguru.logger` for all logging
- **Avoid**: Do not use `print()` or the standard `logging` module
- **Levels**:
  - `logger.debug()` - Development/troubleshooting information
  - `logger.info()` - General informational messages
  - `logger.warning()` - Warning conditions
  - `logger.error()` - Error conditions with full traceback
  - `logger.critical()` - Critical errors

- **Example**:

  ```python
  from loguru import logger

  try:
      result = process_point_cloud(data)
      logger.info(f"Processed {len(data)} points")
  except Exception as e:
      logger.error(f"Failed to process: {e}", exc_info=True)
      raise
  ```

## Exception Handling Best Practices

- Catch specific exceptions, not bare `except:`
- Log exceptions with context before re-raising
- Use `exc_info=True` in logger calls to capture stack traces
- Provide meaningful error messages with context
- Clean up resources in `finally` blocks or use context managers

## Validation

- Validate inputs at function entry points
- Use type hints to document expected types
- Raise appropriate project exceptions for invalid data
- Include helpful error messages for debugging
