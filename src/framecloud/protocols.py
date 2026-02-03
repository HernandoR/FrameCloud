"""Protocol definitions for PointCloud I/O operations.

This module defines Protocol interfaces (similar to Rust Traits) for various
I/O operations on PointCloud objects. Protocols provide type-safe interfaces
that separate interface definitions from implementations.

Note: Protocols are unified - if something can read a format, it can also write it.
"""

from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np


@runtime_checkable
class LasIOProtocol(Protocol):
    """Protocol for LAS/LAZ file I/O operations (read and write)."""

    @classmethod
    def from_las(cls, file_path: Path | str) -> Self:
        """Load a PointCloud from a LAS/LAZ file.

        Args:
            file_path: Path to the LAS/LAZ file.

        Returns:
            PointCloud object.
        """
        ...

    def to_las(self, file_path: Path | str) -> None:
        """Save this PointCloud to a LAS file.

        Args:
            file_path: Path to the output LAS file.
        """
        ...


@runtime_checkable
class ParquetIOProtocol(Protocol):
    """Protocol for Parquet file I/O operations (read and write)."""

    @classmethod
    def from_parquet(
        cls,
        file_path: Path | str,
        position_cols: list[str] | None = None,
    ) -> Self:
        """Load a PointCloud from a Parquet file.

        Args:
            file_path: Path to the Parquet file.
            position_cols: List of column names for point positions.

        Returns:
            PointCloud object.
        """
        ...

    def to_parquet(
        self, file_path: Path | str, position_cols: list[str] | None = None
    ) -> None:
        """Save this PointCloud to a Parquet file.

        Args:
            file_path: Path to the output Parquet file.
            position_cols: List of column names for point positions.
        """
        ...


@runtime_checkable
class BinaryIOProtocol(Protocol):
    """Protocol for binary buffer/file I/O operations (read and write)."""

    @classmethod
    def from_binary_buffer(
        cls,
        bytes_buffer: bytes,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> Self:
        """Load a PointCloud from a binary buffer.

        Args:
            bytes_buffer: Bytes buffer containing the binary data.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.

        Returns:
            PointCloud object.
        """
        ...

    def to_binary_buffer(
        self,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> bytes:
        """Save this PointCloud to a binary buffer.

        Args:
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.

        Returns:
            Bytes buffer containing the binary data.
        """
        ...

    @classmethod
    def from_binary_file(
        cls,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> Self:
        """Load a PointCloud from a binary file.

        Args:
            file_path: Path to the binary file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.

        Returns:
            PointCloud object.
        """
        ...

    def to_binary_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> None:
        """Save this PointCloud to a binary file.

        Args:
            file_path: Path to the output binary file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.
        """
        ...


@runtime_checkable
class NumpyIOProtocol(Protocol):
    """Protocol for NumPy file format I/O operations (read and write for .npy and .npz)."""

    @classmethod
    def from_numpy_file(
        cls,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> Self:
        """Load a PointCloud from a NumPy .npy file.

        Args:
            file_path: Path to the NumPy .npy file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.

        Returns:
            PointCloud object.
        """
        ...

    def to_numpy_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> None:
        """Save this PointCloud to a NumPy .npy file.

        Args:
            file_path: Path to the output NumPy .npy file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.
        """
        ...

    @classmethod
    def from_npz_file(
        cls,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> Self:
        """Load a PointCloud from a NumPy .npz file.

        Args:
            file_path: Path to the NumPy .npz file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.

        Returns:
            PointCloud object.
        """
        ...

    def to_npz_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> None:
        """Save this PointCloud to a NumPy .npz file.

        Args:
            file_path: Path to the output NumPy .npz file.
            attribute_names: List of attribute names in order.
            dtype: NumPy data type for the array.
        """
        ...
