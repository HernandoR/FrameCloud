"""Benchmark tests for point cloud processing using pytest-benchmark.

This module contains performance benchmarks for various point cloud operations
using pytest-benchmark, which automatically generates reports and statistics.

To run benchmarks:
    uv run pytest tests/test_benchmark.py -m benchmark --benchmark-only

To view previous benchmark results:
    Check the reports/benchmarks/ directory for JSON and histogram outputs
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framecloud.np.core import PointCloud as NpPointCloud
from framecloud.pd.core import PointCloud as PdPointCloud


# Use smaller sizes by default for faster feedback
@pytest.fixture(params=[1e5, 1e6, 1e7])
def small_benchmark_size(request):
    """Parametrized fixture for smaller benchmark point cloud sizes."""
    return int(request.param)


@pytest.fixture(params=[1e8, 2e8])
def large_benchmark_size(request):
    """Parametrized fixture for large benchmark point cloud sizes."""
    return int(request.param)


@pytest.mark.benchmark(group="creation")
class TestBenchmarkPointCloudCreation:
    """Benchmark tests for creating point clouds."""

    def test_np_create_pointcloud(self, benchmark, small_benchmark_size):
        """Benchmark creating point cloud with numpy."""

        def create_np_pointcloud():
            np.random.seed(42)
            points = np.random.randn(small_benchmark_size, 3).astype(np.float32)
            intensities = np.random.rand(small_benchmark_size).astype(np.float32)
            return NpPointCloud(points=points, attributes={"intensities": intensities})

        result = benchmark(create_np_pointcloud)
        assert result.num_points == small_benchmark_size

    def test_pd_create_pointcloud(self, benchmark, small_benchmark_size):
        """Benchmark creating point cloud with pandas."""

        def create_pd_pointcloud():
            np.random.seed(42)
            df = pd.DataFrame(
                {
                    "X": np.random.randn(small_benchmark_size).astype(np.float32),
                    "Y": np.random.randn(small_benchmark_size).astype(np.float32),
                    "Z": np.random.randn(small_benchmark_size).astype(np.float32),
                    "intensities": np.random.rand(small_benchmark_size).astype(
                        np.float32
                    ),
                }
            )
            return PdPointCloud(data=df)

        result = benchmark(create_pd_pointcloud)
        assert result.num_points == small_benchmark_size


@pytest.mark.benchmark(group="transformation")
class TestBenchmarkTransformation:
    """Benchmark tests for transforming point clouds."""

    def test_np_transform_pointcloud(self, benchmark, small_benchmark_size):
        """Benchmark transformation with numpy implementation."""
        np.random.seed(42)
        points = np.random.randn(small_benchmark_size, 3).astype(np.float32)
        pc = NpPointCloud(points=points)
        matrix = np.array([[2, 0, 0, 10], [0, 2, 0, 20], [0, 0, 2, 30], [0, 0, 0, 1]])

        result = benchmark(pc.transform, matrix, inplace=False)
        assert result.num_points == small_benchmark_size

    def test_pd_transform_pointcloud(self, benchmark, small_benchmark_size):
        """Benchmark transformation with pandas implementation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(small_benchmark_size).astype(np.float32),
                "Y": np.random.randn(small_benchmark_size).astype(np.float32),
                "Z": np.random.randn(small_benchmark_size).astype(np.float32),
            }
        )
        pc = PdPointCloud(data=df)
        matrix = np.array([[2, 0, 0, 10], [0, 2, 0, 20], [0, 0, 2, 30], [0, 0, 0, 1]])

        result = benchmark(pc.transform, matrix, inplace=False)
        assert result.num_points == small_benchmark_size


@pytest.mark.benchmark(group="sampling")
class TestBenchmarkSampling:
    """Benchmark tests for sampling point clouds."""

    def test_np_sample_pointcloud(self, benchmark, small_benchmark_size):
        """Benchmark sampling with numpy implementation."""
        np.random.seed(42)
        points = np.random.randn(small_benchmark_size, 3).astype(np.float32)
        pc = NpPointCloud(points=points)

        result = benchmark(pc.sample, num_samples=10000, replace=False)
        assert result.num_points == 10000

    def test_pd_sample_pointcloud(self, benchmark, small_benchmark_size):
        """Benchmark sampling with pandas implementation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(small_benchmark_size).astype(np.float32),
                "Y": np.random.randn(small_benchmark_size).astype(np.float32),
                "Z": np.random.randn(small_benchmark_size).astype(np.float32),
            }
        )
        pc = PdPointCloud(data=df)

        result = benchmark(pc.sample, num_samples=10000, replace=False)
        assert result.num_points == 10000


@pytest.mark.benchmark(group="io")
class TestBenchmarkIO:
    """Benchmark tests for I/O operations with point clouds."""

    def test_np_parquet_write(self, benchmark, small_benchmark_size):
        """Benchmark parquet write with numpy implementation."""
        np.random.seed(42)
        points = np.random.randn(small_benchmark_size, 3).astype(np.float32)
        intensities = np.random.rand(small_benchmark_size).astype(np.float32)
        pc = NpPointCloud(points=points, attributes={"intensities": intensities})

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "benchmark.parquet"
            benchmark(pc.to_parquet, file_path)

    def test_np_parquet_read(self, benchmark, small_benchmark_size):
        """Benchmark parquet read with numpy implementation."""
        np.random.seed(42)
        points = np.random.randn(small_benchmark_size, 3).astype(np.float32)
        intensities = np.random.rand(small_benchmark_size).astype(np.float32)
        pc = NpPointCloud(points=points, attributes={"intensities": intensities})

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "benchmark.parquet"
            pc.to_parquet(file_path)
            result = benchmark(NpPointCloud.from_parquet, file_path)
            assert result.num_points == small_benchmark_size

    def test_pd_parquet_write(self, benchmark, small_benchmark_size):
        """Benchmark parquet write with pandas implementation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(small_benchmark_size).astype(np.float32),
                "Y": np.random.randn(small_benchmark_size).astype(np.float32),
                "Z": np.random.randn(small_benchmark_size).astype(np.float32),
                "intensities": np.random.rand(small_benchmark_size).astype(np.float32),
            }
        )
        pc = PdPointCloud(data=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "benchmark.parquet"
            benchmark(pc.to_parquet, file_path)

    def test_pd_parquet_read(self, benchmark, small_benchmark_size):
        """Benchmark parquet read with pandas implementation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(small_benchmark_size).astype(np.float32),
                "Y": np.random.randn(small_benchmark_size).astype(np.float32),
                "Z": np.random.randn(small_benchmark_size).astype(np.float32),
                "intensities": np.random.rand(small_benchmark_size).astype(np.float32),
            }
        )
        pc = PdPointCloud(data=df)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "benchmark.parquet"
            pc.to_parquet(file_path)
            result = benchmark(PdPointCloud.from_parquet, file_path)
            assert result.num_points == small_benchmark_size


@pytest.mark.benchmark(group="attributes")
class TestBenchmarkAttributeOperations:
    """Benchmark tests for attribute operations with point clouds."""

    def test_np_add_attribute(self, benchmark, small_benchmark_size):
        """Benchmark adding attributes with numpy implementation."""

        def add_attribute():
            np.random.seed(42)
            points = np.random.randn(small_benchmark_size, 3).astype(np.float32)
            pc = NpPointCloud(points=points)
            new_attr = np.random.rand(small_benchmark_size).astype(np.float32)
            pc.add_attribute("new_attribute", new_attr)
            return pc

        result = benchmark(add_attribute)
        assert "new_attribute" in result.attribute_names

    def test_pd_add_attribute(self, benchmark, small_benchmark_size):
        """Benchmark adding attributes with pandas implementation."""

        def add_attribute():
            np.random.seed(42)
            df = pd.DataFrame(
                {
                    "X": np.random.randn(small_benchmark_size).astype(np.float32),
                    "Y": np.random.randn(small_benchmark_size).astype(np.float32),
                    "Z": np.random.randn(small_benchmark_size).astype(np.float32),
                }
            )
            pc = PdPointCloud(data=df)
            new_attr = np.random.rand(small_benchmark_size).astype(np.float32)
            pc.add_attribute("new_attribute", new_attr)
            return pc

        result = benchmark(add_attribute)
        assert "new_attribute" in result.attribute_names


@pytest.mark.slow
@pytest.mark.benchmark(group="large-scale")
class TestBenchmarkLargeScale:
    """Benchmark tests for very large point clouds (10M+ points).

    These tests are marked as 'slow' and should be run separately with:
        uv run pytest -m "slow and benchmark" --benchmark-only
    """

    def test_np_create_large_pointcloud(self, benchmark, large_benchmark_size):
        """Benchmark creating very large point cloud with numpy."""

        def create_large_np_pointcloud():
            np.random.seed(42)
            points = np.random.randn(large_benchmark_size, 3).astype(np.float32)
            intensities = np.random.rand(large_benchmark_size).astype(np.float32)
            return NpPointCloud(points=points, attributes={"intensities": intensities})

        result = benchmark(create_large_np_pointcloud)
        assert result.num_points == large_benchmark_size

    def test_pd_create_large_pointcloud(self, benchmark, large_benchmark_size):
        """Benchmark creating very large point cloud with pandas."""

        def create_large_pd_pointcloud():
            np.random.seed(42)
            df = pd.DataFrame(
                {
                    "X": np.random.randn(large_benchmark_size).astype(np.float32),
                    "Y": np.random.randn(large_benchmark_size).astype(np.float32),
                    "Z": np.random.randn(large_benchmark_size).astype(np.float32),
                    "intensities": np.random.rand(large_benchmark_size).astype(
                        np.float32
                    ),
                }
            )
            return PdPointCloud(data=df)

        result = benchmark(create_large_pd_pointcloud)
        assert result.num_points == large_benchmark_size

    # transformation
    def test_np_transform_large_pointcloud(self, benchmark, large_benchmark_size):
        """Benchmark transforming very large point cloud with numpy."""
        np.random.seed(42)
        points = np.random.randn(large_benchmark_size, 3).astype(np.float32)
        pc = NpPointCloud(points=points)
        matrix = np.array(
            [
                [2, 0, 0, 10],
                [0, 2, 0, 20],
                [0, 0, 2, 30],
                [0, 0, 0, 1],
            ]
        )

        result = benchmark(pc.transform, matrix, inplace=False)
        assert result.num_points == large_benchmark_size

    def test_pd_transform_large_pointcloud(self, benchmark, large_benchmark_size):
        """Benchmark transforming very large point cloud with pandas."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(large_benchmark_size).astype(np.float32),
                "Y": np.random.randn(large_benchmark_size).astype(np.float32),
                "Z": np.random.randn(large_benchmark_size).astype(np.float32),
            }
        )
        pc = PdPointCloud(data=df)
        matrix = np.array(
            [
                [2, 0, 0, 10],
                [0, 2, 0, 20],
                [0, 0, 2, 30],
                [0, 0, 0, 1],
            ]
        )

        result = benchmark(pc.transform, matrix, inplace=False)
        assert result.num_points == large_benchmark_size
