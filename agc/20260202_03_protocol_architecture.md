# Consolidated I/O Architecture

This document explains the consolidated I/O architecture in FrameCloud, which embeds all I/O operations directly into the PointCloud class with unified protocols.

## Overview

The I/O operations are consolidated into the PointCloud core classes with:

1. **All I/O methods embedded in PointCloud** - No separate implementation files
2. **Unified protocols** - Each protocol combines read and write operations
3. **Clear section organization** - Comment headers separate different I/O formats
4. **Better static type checking** - Everything in one file for better analysis

## Architecture

### Unified Protocol Definitions (`src/framecloud/protocols.py`)

Protocols define interface contracts with read and write operations combined:

```python
from typing import Protocol, Self, runtime_checkable

@runtime_checkable
class LasIOProtocol(Protocol):
    """Protocol for LAS/LAZ file I/O operations (read and write)."""
    
    @classmethod
    def from_las(cls, file_path: Path | str) -> Self:
        ...
    
    def to_las(self, file_path: Path | str) -> None:
        ...
```

Available protocols (each combines read + write):
- `LasIOProtocol` - LAS/LAZ file operations
- `ParquetIOProtocol` - Parquet file operations
- `BinaryIOProtocol` - Binary buffer/file operations
- `NumpyIOProtocol` - NumPy file format operations

### Consolidated Core Classes

All I/O methods are embedded directly in the PointCloud classes:

#### NumPy-based implementation (`src/framecloud/np/core.py`):
```python
class PointCloud(BaseModel):
    # ... core attributes and methods ...
    
    # ========================================================================
    # LAS/LAZ File I/O Operations
    # ========================================================================
    
    @classmethod
    def from_las(cls, file_path: Path | str):
        ...
    
    def to_las(self, file_path: Path | str):
        ...
    
    # ========================================================================
    # Parquet File I/O Operations  
    # ========================================================================
    
    @classmethod
    def from_parquet(cls, file_path: Path | str, position_cols: list[str] = None):
        ...
```

#### Pandas-based implementation (`src/framecloud/pd/core.py`):
Similar structure with all I/O methods embedded directly in the PdPointCloud class.

### Shared Utilities

**Common exceptions** (`src/framecloud/exceptions.py`):
- `AttributeExistsError`
- `ArrayShapeError`

**Utility functions** (`src/framecloud/_io_utils.py`):
- `validate_xyz_in_attribute_names()`
- `validate_buffer_size()`
- `default_attribute_names()`
- `extract_xyz_arrays()`
- `extract_attributes_dict()`

## Benefits

### 1. Better Maintainability
All I/O operations are in the PointCloud class, making the codebase easier to navigate:

```python
# Direct use of PointCloud methods
from framecloud.np.core import PointCloud

pc = PointCloud(points=points)
pc.to_las("output.las")
loaded = PointCloud.from_las("input.las")
```

### 2. Improved Static Type Checking
Type checkers can properly analyze everything since it's in one file:

```python
from framecloud.protocols import LasIOProtocol
from framecloud.np.core import PointCloud

# Type checker verifies PointCloud implements LasIOProtocol
def process_las_file(handler: LasIOProtocol, path: str):
    return handler.from_las(path)

pc_class = PointCloud
assert isinstance(pc_class, LasIOProtocol)  # Runtime check
```

### 3. Unified Protocols
Each protocol combines read and write operations, reflecting that if something can read a format, it should be able to write it too.

### 4. Code Organization
Clear section markers make it easy to find specific I/O operations:

```python
# Navigate to specific section in core.py
# ========================================================================
# LAS/LAZ File I/O Operations
# ========================================================================
```

### 5. Reduced Duplication
Common logic is extracted into shared utilities and exception classes.

## Usage Examples

| Rust Traits | Python Protocols |
|-------------|------------------|
| `trait Reader { fn read(...) }` | `class ReaderProtocol(Protocol): def read(...)` |
| `impl Reader for MyType` | `class MyType` (structural typing) |
| Compile-time checking | Static type checker (mypy, pyright) |
| Explicit trait bounds | Type hints with Protocol |

## Examples

See `examples/protocol_architecture_demo.py` for a comprehensive demonstration.

### Basic Usage

```python
from framecloud.np.core import PointCloud
from framecloud.np.las_io import LasIO

# Create point cloud
points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
pc = PointCloud(points=points)

# Use specific implementation
LasIO.to_las(pc, "output.las")
loaded = LasIO.from_las("output.las")
```

### Protocol Compliance

```python
from framecloud.protocols import LasReaderProtocol
from framecloud.np.las_io import LasIO

# Verify implementation satisfies protocol
assert isinstance(LasIO(), LasReaderProtocol)
```

### Cross-Implementation Compatibility

```python
from framecloud.np.parquet_io import ParquetIO as NpParquetIO
from framecloud.pd.parquet_io import ParquetIO as PdParquetIO

# Save with NumPy implementation
NpParquetIO.to_parquet(np_pc, "data.parquet")

# Load with Pandas implementation
pd_pc = PdParquetIO.from_parquet("data.parquet")
```

## Testing

Run Protocol architecture tests:

```bash
uv run pytest tests/test_protocol_architecture.py -v
```

Run all I/O tests:

```bash
uv run pytest tests/test_point_cloud_io.py tests/test_pd_point_cloud_io.py -v
```

## Migration Guide

No migration needed! The refactoring maintains 100% backward compatibility:

```python
# Both approaches work identically:

# Approach 1: Direct implementation (new)
from framecloud.np.las_io import LasIO
LasIO.to_las(pc, "output.las")

# Approach 2: Unified interface (existing)
from framecloud.np.pointcloud_io import PointCloudIO
PointCloudIO.to_las(pc, "output.las")
```

## Further Reading

- [PEP 544 – Protocols: Structural subtyping](https://peps.python.org/pep-0544/)
- [Rust Traits Documentation](https://doc.rust-lang.org/book/ch10-02-traits.html)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
