"""Tests for the PointCloudIO class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from framecloud.np.core import PointCloud
from framecloud.np.pintcloud_io import PointCloudIO


class TestPointCloudIOLAS:
    """Test LAS/LAZ file I/O operations."""

    def test_las_roundtrip(self):
        """Test saving and loading LAS file."""
        points = np.array([[0.0, 0.0, 0.0], [1.5, 2.5, 3.5], [10.0, 20.0, 30.0]])
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.las"
            PointCloudIO.to_las(pc, file_path)
            loaded_pc = PointCloudIO.from_las(file_path)

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
            PointCloudIO.to_las(pc, file_path)
            loaded_pc = PointCloudIO.from_las(file_path)

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
            PointCloudIO.to_parquet(pc, file_path)
            loaded_pc = PointCloudIO.from_parquet(file_path)

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
            PointCloudIO.to_parquet(pc, file_path)
            loaded_pc = PointCloudIO.from_parquet(file_path)

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
            PointCloudIO.to_parquet(pc, file_path, position_cols=position_cols)
            loaded_pc = PointCloudIO.from_parquet(
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

        buffer = PointCloudIO.to_binary_buffer(pc)
        loaded_pc = PointCloudIO.from_binary_buffer(buffer)

        assert loaded_pc.num_points == pc.num_points
        np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_binary_buffer_with_attributes(self):
        """Test binary buffer with attributes."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        intensities = np.array([1.0, 2.0], dtype=np.float32)
        pc = PointCloud(points=points, attributes={"intensity": intensities})

        attribute_names = ["X", "Y", "Z", "intensity"]
        buffer = PointCloudIO.to_binary_buffer(pc, attribute_names=attribute_names)
        loaded_pc = PointCloudIO.from_binary_buffer(
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
            PointCloudIO.to_binary_file(pc, file_path)
            loaded_pc = PointCloudIO.from_binary_file(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_binary_invalid_buffer(self):
        """Test binary buffer with invalid attribute names."""
        buffer = b"invalid"
        with pytest.raises(ValueError, match="Attribute names must include"):
            PointCloudIO.from_binary_buffer(buffer, attribute_names=["A", "B", "C"])


class TestPointCloudIONumPy:
    """Test NumPy .npy file I/O operations."""

    def test_numpy_file_roundtrip(self):
        """Test NumPy .npy file I/O."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npy"
            PointCloudIO.to_numpy_file(pc, file_path)
            loaded_pc = PointCloudIO.from_numpy_file(file_path)

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
            PointCloudIO.to_numpy_file(pc, file_path, attribute_names=attribute_names)
            loaded_pc = PointCloudIO.from_numpy_file(
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
            PointCloudIO.to_npz_file(pc, file_path)
            loaded_pc = PointCloudIO.from_npz_file(file_path)

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
            PointCloudIO.to_npz_file(pc, file_path, attribute_names=attribute_names)
            loaded_pc = PointCloudIO.from_npz_file(
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
            PointCloudIO.to_npz_file(pc, file_path)

            # Try loading with attributes that don't exist
            with pytest.raises(ValueError, match="not found in .npz file"):
                PointCloudIO.from_npz_file(
                    file_path, attribute_names=["X", "Y", "Z", "nonexistent"]
                )


class TestPointCloudIOGeneric:
    """Test generic from_file and to_file methods."""

    def test_from_file_infers_las(self):
        """Test that from_file infers LAS format."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.las"
            PointCloudIO.to_file(pc, file_path)
            loaded_pc = PointCloudIO.from_file(file_path)

            assert loaded_pc.num_points == pc.num_points

    def test_from_file_infers_parquet(self):
        """Test that from_file infers Parquet format."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"
            PointCloudIO.to_file(pc, file_path)
            loaded_pc = PointCloudIO.from_file(file_path)

            assert loaded_pc.num_points == pc.num_points

    def test_from_file_infers_npy(self):
        """Test that from_file infers NPY format."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npy"
            PointCloudIO.to_file(pc, file_path)
            loaded_pc = PointCloudIO.from_file(file_path)

            assert loaded_pc.num_points == pc.num_points

    def test_from_file_infers_npz(self):
        """Test that from_file infers NPZ format."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npz"
            PointCloudIO.to_file(pc, file_path)
            loaded_pc = PointCloudIO.from_file(file_path)

            assert loaded_pc.num_points == pc.num_points

    def test_from_file_explicit_type(self):
        """Test from_file with explicit file type."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npz"
            PointCloudIO.to_file(pc, file_path, file_type=".npz")
            loaded_pc = PointCloudIO.from_file(file_path, file_type=".npz")

            assert loaded_pc.num_points == pc.num_points

    def test_unsupported_file_type(self):
        """Test that unsupported file type raises error."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        pc = PointCloud(points=points)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.xyz"
            with pytest.raises(ValueError, match="Unsupported file type"):
                PointCloudIO.to_file(pc, file_path)

    def test_from_file_unsupported_type(self):
        """Test that loading unsupported file type raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.xyz"
            file_path.write_text("dummy")
            with pytest.raises(ValueError, match="Unsupported file type"):
                PointCloudIO.from_file(file_path)
