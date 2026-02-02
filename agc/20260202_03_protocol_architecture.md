# Protocol-Based Architecture (Rust Trait Pattern)

This document explains the Protocol-based architecture used for I/O operations in FrameCloud, which follows a Rust Trait-like pattern using Python's `typing.Protocol`.

## Overview

Instead of using factory classes for I/O operations, we've implemented a Protocol-based architecture that:

1. **Separates interfaces from implementations** - Protocols define what methods must exist, implementations provide the actual code
2. **Enables type-safe composition** - Static type checkers can verify that implementations satisfy protocols
3. **Supports cross-file implementation splitting** - Each I/O format has its own isolated module
4. **Provides better maintainability** - Clear separation of concerns makes code easier to understand and modify

## Architecture

### Protocol Definitions (`src/framecloud/protocols.py`)

Protocols define the interface contracts (similar to Rust Traits):

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LasReaderProtocol(Protocol):
    """Protocol for reading LAS/LAZ files."""
    def from_las(self, file_path: Path | str):
        ...

@runtime_checkable
class LasWriterProtocol(Protocol):
    """Protocol for writing LAS/LAZ files."""
    def to_las(self, point_cloud, file_path: Path | str) -> None:
        ...
```

Available protocols:
- `LasReaderProtocol` / `LasWriterProtocol` - LAS/LAZ file operations
- `ParquetReaderProtocol` / `ParquetWriterProtocol` - Parquet file operations
- `BinaryReaderProtocol` / `BinaryWriterProtocol` - Binary buffer/file operations
- `NumpyReaderProtocol` / `NumpyWriterProtocol` - NumPy file format operations

### Implementation Modules

Each I/O format has its own implementation module:

#### NumPy-based implementations:
- `src/framecloud/np/las_io.py` - LAS/LAZ file I/O
- `src/framecloud/np/parquet_io.py` - Parquet file I/O
- `src/framecloud/np/binary_io.py` - Binary buffer/file I/O
- `src/framecloud/np/numpy_io.py` - NumPy file format I/O

#### Pandas-based implementations:
- `src/framecloud/pd/las_io.py` - LAS/LAZ file I/O
- `src/framecloud/pd/parquet_io.py` - Parquet file I/O
- `src/framecloud/pd/binary_io.py` - Binary buffer/file I/O
- `src/framecloud/pd/numpy_io.py` - NumPy file format I/O

### Composition Pattern

The main `PointCloudIO` class composes all implementations through multiple inheritance:

```python
class PointCloudIO(LasIO, ParquetIO, BinaryIO, NumpyIO):
    """Unified interface composing all I/O implementations."""
    
    @staticmethod
    def from_las(file_path: Path | str) -> PointCloud:
        return LasIO.from_las(file_path)
    
    @staticmethod
    def to_las(point_cloud: PointCloud, file_path: Path | str):
        return LasIO.to_las(point_cloud, file_path)
    
    # ... similar delegation for other formats
```

## Benefits

### 1. Separation of Concerns
Each I/O format is isolated in its own module, making the code easier to understand and maintain:

```python
# Use specific implementation directly
from framecloud.np.las_io import LasIO
LasIO.to_las(point_cloud, "output.las")

# Or use the unified interface
from framecloud.np.pointcloud_io import PointCloudIO
PointCloudIO.to_las(point_cloud, "output.las")
```

### 2. Type Safety
Static type checkers can verify protocol compliance:

```python
from framecloud.protocols import LasReaderProtocol
from framecloud.np.las_io import LasIO

# Type checker verifies LasIO implements LasReaderProtocol
def process_las_file(reader: LasReaderProtocol, path: str):
    return reader.from_las(path)

las_io = LasIO()
assert isinstance(las_io, LasReaderProtocol)  # Runtime check
```

### 3. Extensibility
Adding new I/O formats is straightforward:

1. Define new protocols in `protocols.py`
2. Create implementation module
3. Add to `PointCloudIO` composition

### 4. Testability
Each implementation can be tested independently:

```python
# Test only LAS I/O
from framecloud.np.las_io import LasIO

def test_las_roundtrip():
    pc = create_test_pointcloud()
    LasIO.to_las(pc, "test.las")
    loaded = LasIO.from_las("test.las")
    assert_equal(pc, loaded)
```

### 5. Backward Compatibility
All existing code continues to work unchanged:

```python
# Old code still works
from framecloud.np.pointcloud_io import PointCloudIO

PointCloudIO.to_las(pc, "output.las")
PointCloudIO.to_parquet(pc, "output.parquet")
```

## Comparison with Rust Traits

This pattern is inspired by Rust's trait system:

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
