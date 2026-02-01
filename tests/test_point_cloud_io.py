"""Tests for the PointCloudIO class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from framecloud.np.core import PointCloud


class TestPointCloudIOLAS:
    """Test LAS/LAZ file I/O operations."""

    def test_las_roundtrip(self):
        """Test saving and loading LAS file."""
        points = np.array([[0.0, 0.0, 0.0], [1.5, 2.5, 3.5], [10.0, 20.0, 30.0]])
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.las"
            pc.to_las(file_path)
            loaded_pc = PointCloud.from_las(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points, decimal=2)

    def test_las_with_attributes(self):
        """Test LAS with additional attributes."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        intensities = np.array([10, 20], dtype=np.uint16)
        returns = np.array([1, 2], dtype=np.uint8)

        pc = PointCloud(
            points=points, attributes={"intensity": intensities, "return_num": returns}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.las"
            pc.to_las(file_path)
            loaded_pc = PointCloud.from_las(file_path)

            assert loaded_pc.num_points == pc.num_points
            assert "intensity" in loaded_pc.attribute_names


class TestPointCloudIOParquet:
    """Test Parquet file I/O operations."""

    def test_parquet_roundtrip(self):
        """Test saving and loading Parquet file."""
        points = np.array([[0.0, 0.0, 0.0], [1.5, 2.5, 3.5], [10.0, 20.0, 30.0]])
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"
            pc.to_parquet(file_path)
            loaded_pc = PointCloud.from_parquet(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_parquet_with_attributes(self):
        """Test Parquet with attributes."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        intensities = np.array([1.0, 2.0, 3.0])

        pc = PointCloud(
            points=points, attributes={"colors": colors, "intensities": intensities}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"
            pc.to_parquet(file_path)
            loaded_pc = PointCloud.from_parquet(file_path)

            assert loaded_pc.num_points == pc.num_points
            assert "colors" in loaded_pc.attribute_names
            assert "intensities" in loaded_pc.attribute_names

    def test_parquet_custom_position_cols(self):
        """Test Parquet with custom position column names."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"
            position_cols = ["px", "py", "pz"]
            pc.to_parquet(file_path, position_cols=position_cols)
            loaded_pc = PointCloud.from_parquet(
                file_path, position_cols=position_cols
            )

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)


class TestPointCloudIOBinary:
    """Test binary buffer and file I/O operations."""

    def test_binary_buffer_roundtrip(self):
        """Test binary buffer serialization."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=points)

        buffer = pc.to_binary_buffer()
        loaded_pc = PointCloud.from_binary_buffer(buffer)

        assert loaded_pc.num_points == pc.num_points
        np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_binary_buffer_with_attributes(self):
        """Test binary buffer with attributes."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        intensities = np.array([1.0, 2.0], dtype=np.float32)
        pc = PointCloud(points=points, attributes={"intensity": intensities})

        attribute_names = ["X", "Y", "Z", "intensity"]
        buffer = pc.to_binary_buffer(attribute_names=attribute_names)
        loaded_pc = PointCloud.from_binary_buffer(
            buffer, attribute_names=attribute_names
        )

        assert loaded_pc.num_points == pc.num_points
        assert "intensity" in loaded_pc.attribute_names

    def test_binary_file_roundtrip(self):
        """Test binary file I/O."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.bin"
            pc.to_binary_file(file_path)
            loaded_pc = PointCloud.from_binary_file(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_binary_invalid_buffer(self):
        """Test binary buffer with invalid attribute names."""
        buffer = b"invalid"
        with pytest.raises(ValueError, match="Attribute names must include"):
            PointCloud.from_binary_buffer(buffer, attribute_names=["A", "B", "C"])


class TestPointCloudIONumPy:
    """Test NumPy .npy file I/O operations."""

    def test_numpy_file_roundtrip(self):
        """Test NumPy .npy file I/O."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npy"
            pc.to_numpy_file(file_path)
            loaded_pc = PointCloud.from_numpy_file(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_numpy_file_with_attributes(self):
        """Test NumPy file with attributes."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        intensities = np.array([1.0, 2.0], dtype=np.float32)
        pc = PointCloud(points=points, attributes={"intensity": intensities})

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npy"
            attribute_names = ["X", "Y", "Z", "intensity"]
            pc.to_numpy_file(file_path, attribute_names=attribute_names)
            loaded_pc = PointCloud.from_numpy_file(
                file_path, attribute_names=attribute_names
            )

            assert loaded_pc.num_points == pc.num_points
            assert "intensity" in loaded_pc.attribute_names


class TestPointCloudIONPZ:
    """Test NumPy .npz file I/O operations."""

    def test_npz_file_roundtrip(self):
        """Test NumPy .npz file I/O."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npz"
            pc.to_npz_file(file_path)
            loaded_pc = PointCloud.from_npz_file(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_npz_file_with_attributes(self):
        """Test NPZ file with attributes."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        intensities = np.array([1.0, 2.0], dtype=np.float32)
        pc = PointCloud(points=points, attributes={"intensities": intensities})

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npz"
            attribute_names = ["X", "Y", "Z", "intensities"]
            pc.to_npz_file(file_path, attribute_names=attribute_names)
            loaded_pc = PointCloud.from_npz_file(
                file_path, attribute_names=attribute_names
            )

            assert loaded_pc.num_points == pc.num_points
            assert "intensities" in loaded_pc.attribute_names

    def test_npz_missing_attribute(self):
        """Test NPZ loading with missing attribute."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npz"
            pc.to_npz_file(file_path)

            # Try loading with attributes that don't exist
            with pytest.raises(ValueError, match="not found in .npz file"):
                PointCloud.from_npz_file(
                    file_path, attribute_names=["X", "Y", "Z", "nonexistent"]
                )


