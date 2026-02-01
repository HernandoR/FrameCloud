"""Tests for Protocol-based architecture implementation.

This module demonstrates how the Protocol-based architecture (similar to Rust Traits)
provides type-safe interfaces and clean separation of concerns.
"""

import tempfile
from pathlib import Path
from typing import runtime_checkable

import numpy as np
import pytest

from framecloud.np.binary_io import BinaryIO as NpBinaryIO
from framecloud.np.core import PointCloud as NpPointCloud
from framecloud.np.las_io import LasIO as NpLasIO
from framecloud.np.numpy_io import NumpyIO as NpNumpyIO
from framecloud.np.parquet_io import ParquetIO as NpParquetIO
from framecloud.pd.binary_io import BinaryIO as PdBinaryIO
from framecloud.pd.core import PointCloud as PdPointCloud
from framecloud.pd.las_io import LasIO as PdLasIO
from framecloud.pd.numpy_io import NumpyIO as PdNumpyIO
from framecloud.pd.parquet_io import ParquetIO as PdParquetIO
from framecloud.protocols import (
    BinaryReaderProtocol,
    BinaryWriterProtocol,
    LasReaderProtocol,
    LasWriterProtocol,
    NumpyReaderProtocol,
    NumpyWriterProtocol,
    ParquetReaderProtocol,
    ParquetWriterProtocol,
)


class TestProtocolCompliance:
    """Test that implementations satisfy Protocol interfaces."""

    def test_las_io_implements_protocols(self):
        """Test that LasIO classes implement LAS protocols."""
        # NumPy-based LAS IO
        assert isinstance(NpLasIO(), LasReaderProtocol)
        assert isinstance(NpLasIO(), LasWriterProtocol)

        # Pandas-based LAS IO
        assert isinstance(PdLasIO(), LasReaderProtocol)
        assert isinstance(PdLasIO(), LasWriterProtocol)

    def test_parquet_io_implements_protocols(self):
        """Test that ParquetIO classes implement Parquet protocols."""
        # NumPy-based Parquet IO
        assert isinstance(NpParquetIO(), ParquetReaderProtocol)
        assert isinstance(NpParquetIO(), ParquetWriterProtocol)

        # Pandas-based Parquet IO
        assert isinstance(PdParquetIO(), ParquetReaderProtocol)
        assert isinstance(PdParquetIO(), ParquetWriterProtocol)

    def test_binary_io_implements_protocols(self):
        """Test that BinaryIO classes implement Binary protocols."""
        # NumPy-based Binary IO
        assert isinstance(NpBinaryIO(), BinaryReaderProtocol)
        assert isinstance(NpBinaryIO(), BinaryWriterProtocol)

        # Pandas-based Binary IO
        assert isinstance(PdBinaryIO(), BinaryReaderProtocol)
        assert isinstance(PdBinaryIO(), BinaryWriterProtocol)

    def test_numpy_io_implements_protocols(self):
        """Test that NumpyIO classes implement NumPy protocols."""
        # NumPy-based NumPy IO
        assert isinstance(NpNumpyIO(), NumpyReaderProtocol)
        assert isinstance(NpNumpyIO(), NumpyWriterProtocol)

        # Pandas-based NumPy IO
        assert isinstance(PdNumpyIO(), NumpyReaderProtocol)
        assert isinstance(PdNumpyIO(), NumpyWriterProtocol)


class TestSeparationOfConcerns:
    """Test that the Protocol-based architecture provides clear separation."""

    def test_las_implementation_is_isolated(self):
        """Test that LAS I/O implementation is isolated in its own module."""
        # Create a simple point cloud
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pc = NpPointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.las"

            # Use LasIO directly for LAS operations
            NpLasIO.to_las(pc, file_path)
            loaded_pc = NpLasIO.from_las(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points, decimal=2)

    def test_parquet_implementation_is_isolated(self):
        """Test that Parquet I/O implementation is isolated in its own module."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pc = NpPointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"

            # Use ParquetIO directly for Parquet operations
            NpParquetIO.to_parquet(pc, file_path)
            loaded_pc = NpParquetIO.from_parquet(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_binary_implementation_is_isolated(self):
        """Test that Binary I/O implementation is isolated in its own module."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pc = NpPointCloud(points=points)

        # Use BinaryIO directly for binary operations
        buffer = NpBinaryIO.to_binary_buffer(pc)
        loaded_pc = NpBinaryIO.from_binary_buffer(buffer)

        assert loaded_pc.num_points == pc.num_points
        np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_numpy_implementation_is_isolated(self):
        """Test that NumPy I/O implementation is isolated in its own module."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pc = NpPointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npy"

            # Use NumpyIO directly for NumPy operations
            NpNumpyIO.to_numpy_file(pc, file_path)
            loaded_pc = NpNumpyIO.from_numpy_file(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)


class TestComposition:
    """Test that composition works correctly with Protocol-based architecture."""

    def test_can_mix_implementations(self):
        """Test that different implementations can be mixed and matched."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pc_np = NpPointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with NumPy implementation
            file_path = Path(tmpdir) / "test.parquet"
            NpParquetIO.to_parquet(pc_np, file_path)

            # Can load with either implementation
            loaded_np = NpParquetIO.from_parquet(file_path)
            loaded_pd = PdParquetIO.from_parquet(file_path)

            assert loaded_np.num_points == pc_np.num_points
            assert loaded_pd.num_points == pc_np.num_points


class TestBackwardCompatibility:
    """Test that the refactored code maintains backward compatibility."""

    def test_pointcloud_io_still_works(self):
        """Test that PointCloudIO still provides all methods."""
        from framecloud.np.pointcloud_io import PointCloudIO

        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pc = NpPointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test all methods are still available
            las_path = Path(tmpdir) / "test.las"
            PointCloudIO.to_las(pc, las_path)
            loaded = PointCloudIO.from_las(las_path)
            assert loaded.num_points == pc.num_points

            parquet_path = Path(tmpdir) / "test.parquet"
            PointCloudIO.to_parquet(pc, parquet_path)
            loaded = PointCloudIO.from_parquet(parquet_path)
            assert loaded.num_points == pc.num_points

            npy_path = Path(tmpdir) / "test.npy"
            PointCloudIO.to_numpy_file(pc, npy_path)
            loaded = PointCloudIO.from_numpy_file(npy_path)
            assert loaded.num_points == pc.num_points

            npz_path = Path(tmpdir) / "test.npz"
            PointCloudIO.to_npz_file(pc, npz_path)
            loaded = PointCloudIO.from_npz_file(npz_path)
            assert loaded.num_points == pc.num_points

            bin_path = Path(tmpdir) / "test.bin"
            PointCloudIO.to_binary_file(pc, bin_path)
            loaded = PointCloudIO.from_binary_file(bin_path)
            assert loaded.num_points == pc.num_points
