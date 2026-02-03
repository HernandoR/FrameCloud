# Python Code Style & Best Practices

Please adhere to the following code style and best practices when generating Python code for this project.

## Documentation (Docstrings)

- **Style**: Follow **Google Style** for all docstrings.
- **Coverage**: Ensure all public modules, classes, and functions have docstrings.
- **Example**:
  ```python
  def fetch_data(url: str) -> dict:
      """Fetches data from the specified URL.

      Args:
          url (str): The URL to fetch data from.

      Returns:
          dict: The JSON response parsed into a dictionary.
      """
      ...
  ```

## Path Handling

- **Library**: Use `pathlib.Path` for all file system interactions.
- **Avoid**: Do not use `os.path` functions (like `os.path.join`) or string manipulation for paths.
- **Project Root**: Leverage `rootutils` (already in dependencies) for robust project root detection where applicable.

## Logging

- **Library**: Use **`loguru`** for logging. Do not use the standard `logging` module.
- **Usage**:
  ```python
  from loguru import logger

  logger.info("Processing started")
  logger.error("An error occurred", exc_info=True)
  ```
- **Print**: Avoid `print()` statements in production code; use `logger.debug()` for development output.

## Type Hinting

- **Strictness**: All code must be fully typed.
- **Modern Syntax**: Use Python 3.10+ features like `X | Y` for unions instead of `Union[X, Y]`, and `list[int]` instead of `List[int]`.
