# Project-Specific Patterns & Architecture

## Module Structure

### NumPy Module (`src/framecloud/np/`)

- Contains NumPy-based point cloud implementations
- Provides efficient array-based operations for large-scale data
- Use vectorized NumPy operations instead of loops
- Leverage broadcasting for performance

### Pandas Module (`src/framecloud/pd/`)

- Contains Pandas-based point cloud implementations
- Useful for tabular data and labeled operations
- Use DataFrame operations for filtering and transformation

### Protocols (`src/framecloud/protocols.py`)

- Defines protocol/interface specifications for the point cloud API
- Use protocols for type hints and polymorphism
- Implement protocols in submodules (`np/`, `pd/`, etc.)
- Ensure all point cloud classes satisfy required protocols

## Point Cloud API

- **Base Operations**: Point clouds support indexing, filtering, transformation
- **I/O Operations**: Use utilities from `_io_utils.py` for file operations
- **Exceptions**: Use custom exceptions from `exceptions.py` for error handling

## I/O Utilities (`src/framecloud/_io_utils.py`)

- Centralized module for all I/O operations
- Use `pathlib.Path` for all file handling (as per style guide)
- Support multiple formats (PLY, PCD, etc.)
- Include proper error handling and logging

## Exceptions (`src/framecloud/exceptions.py`)

- Define project-specific exceptions here
- Use for validation and error handling throughout the codebase
- Never use generic `Exception` or `ValueError` when project-specific exceptions are more appropriate

## Code Organization

- Keep related functionality in the same module
- Use `__init__.py` to expose public API
- Mark internal/private functions with leading underscore (`_function_name`)
- Import typing from `typing` module using modern syntax (Python 3.10+)
