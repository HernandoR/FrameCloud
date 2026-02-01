"""Benchmark tests for point cloud processing with large datasets (10-100M points)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framecloud.np.core import PointCloud as NpPointCloud
from framecloud.np.pointcloud_io import PointCloudIO as NpPointCloudIO
from framecloud.pd.core import PointCloud as PdPointCloud
from framecloud.pd.pointcloud_io import PointCloudIO as PdPointCloudIO


@pytest.fixture(params=[10_000_000, 50_000_000, 100_000_000])
def benchmark_size(request):
    """Parametrized fixture for benchmark point cloud sizes."""
    return request.param


@pytest.mark.benchmark
class TestBenchmarkPointCloudCreation:
    """Benchmark tests for creating large point clouds."""

    def test_np_create_large_pointcloud(self, benchmark_size):
        """Benchmark creating large point cloud with numpy."""
        np.random.seed(42)
        points = np.random.randn(benchmark_size, 3).astype(np.float32)
        intensities = np.random.rand(benchmark_size).astype(np.float32)

        pc = NpPointCloud(points=points, attributes={"intensities": intensities})
        assert pc.num_points == benchmark_size

    def test_pd_create_large_pointcloud(self, benchmark_size):
        """Benchmark creating large point cloud with pandas."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(benchmark_size).astype(np.float32),
                "Y": np.random.randn(benchmark_size).astype(np.float32),
                "Z": np.random.randn(benchmark_size).astype(np.float32),
                "intensities": np.random.rand(benchmark_size).astype(np.float32),
            }
        )

        pc = PdPointCloud(data=df)
        assert pc.num_points == benchmark_size


@pytest.mark.benchmark
class TestBenchmarkTransformation:
    """Benchmark tests for transforming large point clouds."""

    def test_np_transform_large_pointcloud(self, benchmark_size):
        """Benchmark transformation with numpy implementation."""
        np.random.seed(42)
        points = np.random.randn(benchmark_size, 3).astype(np.float32)
        pc = NpPointCloud(points=points)

        matrix = np.array([[2, 0, 0, 10], [0, 2, 0, 20], [0, 0, 2, 30], [0, 0, 0, 1]])

        transformed = pc.transform(matrix, inplace=False)
        assert transformed.num_points == benchmark_size

    def test_pd_transform_large_pointcloud(self, benchmark_size):
        """Benchmark transformation with pandas implementation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(benchmark_size).astype(np.float32),
                "Y": np.random.randn(benchmark_size).astype(np.float32),
                "Z": np.random.randn(benchmark_size).astype(np.float32),
            }
        )
        pc = PdPointCloud(data=df)

        matrix = np.array([[2, 0, 0, 10], [0, 2, 0, 20], [0, 0, 2, 30], [0, 0, 0, 1]])

        transformed = pc.transform(matrix, inplace=False)
        assert transformed.num_points == benchmark_size


@pytest.mark.benchmark
class TestBenchmarkSampling:
    """Benchmark tests for sampling large point clouds."""

    def test_np_sample_large_pointcloud(self, benchmark_size):
        """Benchmark sampling with numpy implementation."""
        np.random.seed(42)
        points = np.random.randn(benchmark_size, 3).astype(np.float32)
        pc = NpPointCloud(points=points)

        sampled = pc.sample(num_samples=10000, replace=False)
        assert sampled.num_points == 10000

    def test_pd_sample_large_pointcloud(self, benchmark_size):
        """Benchmark sampling with pandas implementation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(benchmark_size).astype(np.float32),
                "Y": np.random.randn(benchmark_size).astype(np.float32),
                "Z": np.random.randn(benchmark_size).astype(np.float32),
            }
        )
        pc = PdPointCloud(data=df)

        sampled = pc.sample(num_samples=10000, replace=False)
        assert sampled.num_points == 10000


@pytest.mark.benchmark
class TestBenchmarkIO:
    """Benchmark tests for I/O operations with large point clouds."""

    def test_np_parquet_io_large(self, benchmark_size):
        """Benchmark parquet I/O with numpy implementation."""
        np.random.seed(42)
        points = np.random.randn(benchmark_size, 3).astype(np.float32)
        intensities = np.random.rand(benchmark_size).astype(np.float32)
        pc = NpPointCloud(points=points, attributes={"intensities": intensities})

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "benchmark.parquet"
            NpPointCloudIO.to_parquet(pc, file_path)
            loaded = NpPointCloudIO.from_parquet(file_path)
            assert loaded.num_points == benchmark_size

    def test_pd_parquet_io_large(self, benchmark_size):
        """Benchmark parquet I/O with pandas implementation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(benchmark_size).astype(np.float32),
                "Y": np.random.randn(benchmark_size).astype(np.float32),
                "Z": np.random.randn(benchmark_size).astype(np.float32),
                "intensities": np.random.rand(benchmark_size).astype(np.float32),
            }
        )
        pc = PdPointCloud(data=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "benchmark.parquet"
            PdPointCloudIO.to_parquet(pc, file_path)
            loaded = PdPointCloudIO.from_parquet(file_path)
            assert loaded.num_points == benchmark_size


@pytest.mark.benchmark
class TestBenchmarkAttributeOperations:
    """Benchmark tests for attribute operations with large point clouds."""

    def test_np_add_attribute_large(self, benchmark_size):
        """Benchmark adding attributes with numpy implementation."""
        np.random.seed(42)
        points = np.random.randn(benchmark_size, 3).astype(np.float32)
        pc = NpPointCloud(points=points)

        new_attr = np.random.rand(benchmark_size).astype(np.float32)
        pc.add_attribute("new_attribute", new_attr)
        assert "new_attribute" in pc.attribute_names

    def test_pd_add_attribute_large(self, benchmark_size):
        """Benchmark adding attributes with pandas implementation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(benchmark_size).astype(np.float32),
                "Y": np.random.randn(benchmark_size).astype(np.float32),
                "Z": np.random.randn(benchmark_size).astype(np.float32),
            }
        )
        pc = PdPointCloud(data=df)

        new_attr = np.random.rand(benchmark_size).astype(np.float32)
        pc.add_attribute("new_attribute", new_attr)
        assert "new_attribute" in pc.attribute_names
