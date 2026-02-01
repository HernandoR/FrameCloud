"""Protocol definitions for PointCloud I/O operations.

This module defines Protocol interfaces (similar to Rust Traits) for various
I/O operations on PointCloud objects. Protocols provide type-safe interfaces
that separate interface definitions from implementations.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LasReaderProtocol(Protocol):
    """Protocol for reading LAS/LAZ files."""

    def from_las(self, file_path: Path | str) -> "PointCloud":
        """Load a PointCloud from a LAS/LAZ file.

        Args:
            file_path: Path to the LAS/LAZ file.

        Returns:
            PointCloud object.
        """
        ...


@runtime_checkable
class LasWriterProtocol(Protocol):
    """Protocol for writing LAS/LAZ files."""

    def to_las(self, point_cloud, file_path: Path | str) -> None:
        """Save a PointCloud to a LAS file.

        Args:
            point_cloud: The PointCloud object to save.
            file_path: Path to the output LAS file.
        """
        ...


@runtime_checkable
class ParquetReaderProtocol(Protocol):
    """Protocol for reading Parquet files."""

    def from_parquet(self, file_path: Path | str, position_cols: list[str] | None = None) -> "PointCloud":
        """Load a PointCloud from a Parquet file.

        Args:
            file_path: Path to the Parquet file.
            position_cols: List of column names for point positions.

        Returns:
            PointCloud object.
        """
        ...


@runtime_checkable
class ParquetWriterProtocol(Protocol):
    """Protocol for writing Parquet files."""

    def to_parquet(
        self, point_cloud, file_path: Path | str, position_cols: list[str] | None = None
    ) -> None:
        """Save a PointCloud to a Parquet file.

        Args:
            point_cloud: The PointCloud object to save.
            file_path: Path to the output Parquet file.
            position_cols: List of column names for point positions.
        """
        ...


@runtime_checkable
class BinaryReaderProtocol(Protocol):
    """Protocol for reading binary buffers and files."""

    def from_binary_buffer(
        self,
        bytes_buffer: bytes,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> "PointCloud":
        """Load a PointCloud from a binary buffer.

        Args:
            bytes_buffer: Bytes buffer containing the binary data.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.

        Returns:
            PointCloud object.
        """
        ...

    def from_binary_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> "PointCloud":
        """Load a PointCloud from a binary file.

        Args:
            file_path: Path to the binary file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.

        Returns:
            PointCloud object.
        """
        ...


@runtime_checkable
class BinaryWriterProtocol(Protocol):
    """Protocol for writing binary buffers and files."""

    def to_binary_buffer(
        self,
        point_cloud,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> bytes:
        """Save a PointCloud to a binary buffer.

        Args:
            point_cloud: The PointCloud object to save.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.

        Returns:
            Bytes buffer containing the binary data.
        """
        ...

    def to_binary_file(
        self,
        point_cloud,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> None:
        """Save a PointCloud to a binary file.

        Args:
            point_cloud: The PointCloud object to save.
            file_path: Path to the output binary file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.
        """
        ...


@runtime_checkable
class NumpyReaderProtocol(Protocol):
    """Protocol for reading NumPy file formats (.npy, .npz)."""

    def from_numpy_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> "PointCloud":
        """Load a PointCloud from a NumPy .npy file.

        Args:
            file_path: Path to the NumPy .npy file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.

        Returns:
            PointCloud object.
        """
        ...

    def from_npz_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> "PointCloud":
        """Load a PointCloud from a NumPy .npz file.

        Args:
            file_path: Path to the NumPy .npz file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.

        Returns:
            PointCloud object.
        """
        ...


@runtime_checkable
class NumpyWriterProtocol(Protocol):
    """Protocol for writing NumPy file formats (.npy, .npz)."""

    def to_numpy_file(
        self,
        point_cloud,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> None:
        """Save a PointCloud to a NumPy .npy file.

        Args:
            point_cloud: The PointCloud object to save.
            file_path: Path to the output NumPy .npy file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.
        """
        ...

    def to_npz_file(
        self,
        point_cloud,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> None:
        """Save a PointCloud to a NumPy .npz file.

        Args:
            point_cloud: The PointCloud object to save.
            file_path: Path to the output NumPy .npz file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.
        """
        ...
