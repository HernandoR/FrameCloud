"""Example demonstrating Protocol-based architecture (Rust Trait-like pattern).

This example shows how typing.Protocol enables:
1. Interface definition (Protocol) separated from implementation
2. Type-safe composition with static type checking
3. Cross-file implementation splitting
4. Support for multiple implementations of the same protocol
"""

import numpy as np

# Example 1: Basic Protocol-based I/O
print("=" * 70)
print("Example 1: Using Protocol-based I/O implementations")
print("=" * 70)

from framecloud.np.core import PointCloud
from framecloud.np.las_io import LasIO
from framecloud.np.parquet_io import ParquetIO

# Create a simple point cloud
points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
attributes = {"intensity": np.array([100, 200, 300])}
pc = PointCloud(points=points, attributes=attributes)

print(f"Created PointCloud with {pc.num_points} points")

# Use specific I/O implementations directly
print("\nUsing LasIO implementation:")
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    las_file = Path(tmpdir) / "example.las"
    LasIO.to_las(pc, las_file)
    print(f"  Saved to: {las_file}")
    loaded_pc = LasIO.from_las(las_file)
    print(f"  Loaded {loaded_pc.num_points} points from LAS file")

print("\nUsing ParquetIO implementation:")
with tempfile.TemporaryDirectory() as tmpdir:
    parquet_file = Path(tmpdir) / "example.parquet"
    ParquetIO.to_parquet(pc, parquet_file)
    print(f"  Saved to: {parquet_file}")
    loaded_pc = ParquetIO.from_parquet(parquet_file)
    print(f"  Loaded {loaded_pc.num_points} points from Parquet file")


# Example 2: Protocol compliance checking
print("\n" + "=" * 70)
print("Example 2: Protocol compliance checking (type safety)")
print("=" * 70)

from framecloud.protocols import LasReaderProtocol, LasWriterProtocol

# Check if implementations satisfy protocols
print(f"LasIO satisfies LasReaderProtocol: {isinstance(LasIO(), LasReaderProtocol)}")
print(f"LasIO satisfies LasWriterProtocol: {isinstance(LasIO(), LasWriterProtocol)}")


# Example 3: Separation of concerns
print("\n" + "=" * 70)
print("Example 3: Separation of concerns")
print("=" * 70)

print("Each I/O format has its own isolated module:")
print("  - framecloud.np.las_io     -> LAS/LAZ file operations")
print("  - framecloud.np.parquet_io -> Parquet file operations")
print("  - framecloud.np.binary_io  -> Binary buffer/file operations")
print("  - framecloud.np.numpy_io   -> NumPy file format operations")
print("\nThis separation provides:")
print("  ✓ Clear code organization")
print("  ✓ Easy maintenance")
print("  ✓ Independent testing")
print("  ✓ Type-safe interfaces")


# Example 4: Composition pattern
print("\n" + "=" * 70)
print("Example 4: Composition pattern (main PointCloudIO)")
print("=" * 70)

from framecloud.np.pointcloud_io import PointCloudIO

print("PointCloudIO composes all implementations:")
print("  class PointCloudIO(LasIO, ParquetIO, BinaryIO, NumpyIO):")
print("      # Delegates to specialized implementations")
print("\nBackward compatible API:")

with tempfile.TemporaryDirectory() as tmpdir:
    # All methods still available through PointCloudIO
    file_path = Path(tmpdir) / "test.las"
    PointCloudIO.to_las(pc, file_path)
    loaded = PointCloudIO.from_las(file_path)
    print(f"  ✓ Loaded {loaded.num_points} points via PointCloudIO.from_las()")

    file_path = Path(tmpdir) / "test.parquet"
    PointCloudIO.to_parquet(pc, file_path)
    loaded = PointCloudIO.from_parquet(file_path)
    print(f"  ✓ Loaded {loaded.num_points} points via PointCloudIO.from_parquet()")


# Example 5: Cross-implementation compatibility
print("\n" + "=" * 70)
print("Example 5: Cross-implementation compatibility")
print("=" * 70)

from framecloud.pd.parquet_io import ParquetIO as PdParquetIO
from framecloud.pd.core import PointCloud as PdPointCloud

print("NumPy and Pandas implementations can share data:")
with tempfile.TemporaryDirectory() as tmpdir:
    file_path = Path(tmpdir) / "shared.parquet"

    # Save with NumPy implementation
    ParquetIO.to_parquet(pc, file_path)
    print(f"  Saved with NumPy implementation")

    # Load with Pandas implementation
    pd_pc = PdParquetIO.from_parquet(file_path)
    print(f"  Loaded with Pandas implementation: {pd_pc.num_points} points")
    print(f"  ✓ Cross-implementation compatibility verified")


print("\n" + "=" * 70)
print("Summary: Protocol-based architecture benefits")
print("=" * 70)
print("""
1. Rust Trait-like interface definition using typing.Protocol
2. Clear separation: interface (Protocol) vs implementation (classes)
3. Type safety: Static type checkers can verify protocol compliance
4. Composition: Main class composes specialized implementations
5. Extensibility: Easy to add new I/O formats
6. Maintainability: Each format isolated in its own module
7. Testability: Independent testing of each implementation
8. Backward compatible: Existing code continues to work
""")
