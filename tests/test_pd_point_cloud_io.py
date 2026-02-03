"""Tests for the pd.PointCloudIO class."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framecloud.pd.core import PointCloud


class TestPointCloudIOLAS:
    """Test LAS/LAZ file I/O operations."""

    def test_las_roundtrip(self):
        """Test saving and loading LAS file."""
        df = pd.DataFrame(
            {"X": [0.0, 1.5, 10.0], "Y": [0.0, 2.5, 20.0], "Z": [0.0, 3.5, 30.0]}
        )
        pc = PointCloud(data=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.las"
            pc.to_las(file_path)
            loaded_pc = PointCloud.from_las(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points, decimal=2)

    def test_las_with_attributes(self):
        """Test LAS with additional attributes."""
        df = pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y": [0.0, 2.0],
                "Z": [0.0, 3.0],
                "intensity": np.array([10, 20], dtype=np.uint16),
                "return_num": np.array([1, 2], dtype=np.uint8),
            }
        )
        pc = PointCloud(data=df)

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
        df = pd.DataFrame(
            {"X": [0.0, 1.5, 10.0], "Y": [0.0, 2.5, 20.0], "Z": [0.0, 3.5, 30.0]}
        )
        pc = PointCloud(data=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"
            pc.to_parquet(file_path)
            loaded_pc = PointCloud.from_parquet(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_parquet_with_attributes(self):
        """Test Parquet with attributes."""
        df = pd.DataFrame(
            {
                "X": [0.0, 1.0, 4.0],
                "Y": [0.0, 2.0, 5.0],
                "Z": [0.0, 3.0, 6.0],
                "colors_0": [255, 0, 0],
                "colors_1": [0, 255, 0],
                "colors_2": [0, 0, 255],
                "intensities": [1.0, 2.0, 3.0],
            }
        )
        pc = PointCloud(data=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"
            pc.to_parquet(file_path)
            loaded_pc = PointCloud.from_parquet(file_path)

            assert loaded_pc.num_points == pc.num_points
            assert "colors_0" in loaded_pc.attribute_names
            assert "intensities" in loaded_pc.attribute_names

    def test_parquet_custom_position_cols(self):
        """Test Parquet with custom position column names."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 2.0], "Z": [0.0, 3.0]})
        pc = PointCloud(data=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"
            position_cols = ["px", "py", "pz"]
            pc.to_parquet(file_path, position_cols=position_cols)
            loaded_pc = PointCloud.from_parquet(file_path, position_cols=position_cols)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)


class TestPointCloudIOBinary:
    """Test binary buffer and file I/O operations."""

    def test_binary_buffer_roundtrip(self):
        """Test binary buffer serialization."""
        df = pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y": [0.0, 2.0],
                "Z": [0.0, 3.0],
            }
        )
        pc = PointCloud(data=df)

        buffer = pc.to_binary_buffer()
        loaded_pc = PointCloud.from_binary_buffer(buffer)

        assert loaded_pc.num_points == pc.num_points
        np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_binary_buffer_with_attributes(self):
        """Test binary buffer with attributes."""
        df = pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y": [0.0, 2.0],
                "Z": [0.0, 3.0],
                "intensity": [1.0, 2.0],
            }
        )
        pc = PointCloud(data=df)

        attribute_names = ["X", "Y", "Z", "intensity"]
        buffer = pc.to_binary_buffer(attribute_names=attribute_names)
        loaded_pc = PointCloud.from_binary_buffer(
            buffer, attribute_names=attribute_names
        )

        assert loaded_pc.num_points == pc.num_points
        assert "intensity" in loaded_pc.attribute_names

    def test_binary_file_roundtrip(self):
        """Test binary file I/O."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 2.0], "Z": [0.0, 3.0]})
        pc = PointCloud(data=df)

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
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 2.0], "Z": [0.0, 3.0]})
        pc = PointCloud(data=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npy"
            pc.to_numpy_file(file_path)
            loaded_pc = PointCloud.from_numpy_file(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_numpy_file_with_attributes(self):
        """Test NumPy file with attributes."""
        df = pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y": [0.0, 2.0],
                "Z": [0.0, 3.0],
                "intensity": [1.0, 2.0],
            }
        )
        pc = PointCloud(data=df)

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
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 2.0], "Z": [0.0, 3.0]})
        pc = PointCloud(data=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npz"
            pc.to_npz_file(file_path)
            loaded_pc = PointCloud.from_npz_file(file_path)

            assert loaded_pc.num_points == pc.num_points
            np.testing.assert_array_almost_equal(loaded_pc.points, pc.points)

    def test_npz_file_with_attributes(self):
        """Test NPZ file with attributes."""
        df = pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y": [0.0, 2.0],
                "Z": [0.0, 3.0],
                "intensities": [1.0, 2.0],
            }
        )
        pc = PointCloud(data=df)

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
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 2.0], "Z": [0.0, 3.0]})
        pc = PointCloud(data=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npz"
            pc.to_npz_file(file_path)

            # Try loading with attributes that don't exist
            with pytest.raises(ValueError, match="not found in .npz file"):
                PointCloud.from_npz_file(
                    file_path, attribute_names=["X", "Y", "Z", "nonexistent"]
                )


class TestPointCloudIOCrossCheck:
    """Cross-check pd.PointCloud against np.PointCloud I/O using fixtures."""

    def test_parquet_consistency_with_np(self, medium_point_cloud_np):
        """Test that parquet I/O produces same results as np implementation."""
        from framecloud.np.core import PointCloud as NpPointCloud
        from tests.conftest import np_to_pd_pointcloud

        # Convert np to pd
        pd_pc = np_to_pd_pointcloud(medium_point_cloud_np)

        with tempfile.TemporaryDirectory() as tmpdir:
            np_file_path = Path(tmpdir) / "test_np.parquet"
            pd_file_path = Path(tmpdir) / "test_pd.parquet"

            # Save with both implementations
            medium_point_cloud_np.to_parquet(np_file_path)
            pd_pc.to_parquet(pd_file_path)

            # Load with both implementations
            np_loaded = NpPointCloud.from_parquet(np_file_path)
            pd_loaded = PointCloud.from_parquet(pd_file_path)

            # Check they have the same number of points
            assert np_loaded.num_points == pd_loaded.num_points
            np.testing.assert_array_almost_equal(
                np_loaded.points, pd_loaded.points, decimal=5
            )
