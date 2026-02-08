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
import pytest

from tests.conftest import create_pointcloud


@pytest.mark.benchmark(group="pcl-creation")
class TestBenchmarkPointCloudCreation:
    """Benchmark tests for creating point clouds."""

    def test_create_pointcloud(self, benchmark, pointcloud_impl, small_benchmark_size):
        """Benchmark creating point cloud with different implementations."""

        def create_pc():
            return create_pointcloud(pointcloud_impl, small_benchmark_size)

        result = benchmark(create_pc)
        assert result.num_points == small_benchmark_size


@pytest.mark.benchmark(group="pcl-transformation")
class TestBenchmarkTransformation:
    """Benchmark tests for transforming point clouds."""

    def test_transform_pointcloud(
        self, benchmark, pointcloud_impl, small_benchmark_size
    ):
        """Benchmark transformation with different implementations."""
        pc = create_pointcloud(
            pointcloud_impl, small_benchmark_size, with_attributes=False
        )
        matrix = np.array([[2, 0, 0, 10], [0, 2, 0, 20], [0, 0, 2, 30], [0, 0, 0, 1]])

        result = benchmark(pc.transform, matrix, inplace=False)
        assert result.num_points == small_benchmark_size


@pytest.mark.benchmark(group="pcl-sampling")
class TestBenchmarkSampling:
    """Benchmark tests for sampling point clouds."""

    def test_sample_pointcloud(self, benchmark, pointcloud_impl, small_benchmark_size):
        """Benchmark sampling with different implementations."""
        pc = create_pointcloud(
            pointcloud_impl, small_benchmark_size, with_attributes=False
        )

        result = benchmark(pc.sample, num_samples=10000, replace=False)
        assert result.num_points == 10000


@pytest.mark.benchmark(group="pcl-io")
class TestBenchmarkIO:
    """Benchmark tests for I/O operations with point clouds."""

    def test_parquet_write(self, benchmark, pointcloud_impl, small_benchmark_size):
        """Benchmark parquet write with different implementations."""
        pc = create_pointcloud(pointcloud_impl, small_benchmark_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "benchmark.parquet"
            benchmark(pc.to_parquet, file_path)

    def test_parquet_read(self, benchmark, pointcloud_impl, small_benchmark_size):
        """Benchmark parquet read with different implementations."""
        from framecloud.np.core import PointCloud as NpPointCloud
        from framecloud.pd.core import PointCloud as PdPointCloud

        pc = create_pointcloud(pointcloud_impl, small_benchmark_size)
        PointCloudClass = NpPointCloud if pointcloud_impl == "np" else PdPointCloud

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "benchmark.parquet"
            pc.to_parquet(file_path)
            result = benchmark(PointCloudClass.from_parquet, file_path)
            assert result.num_points == small_benchmark_size


@pytest.mark.benchmark(group="pcl-attributes")
class TestBenchmarkAttributeOperations:
    """Benchmark tests for attribute operations with point clouds."""

    def test_add_attribute(self, benchmark, pointcloud_impl, small_benchmark_size):
        """Benchmark adding attributes with different implementations."""

        def add_attribute():
            pc = create_pointcloud(
                pointcloud_impl, small_benchmark_size, with_attributes=False
            )
            new_attr = np.random.rand(small_benchmark_size).astype(np.float32)
            pc.add_attribute("new_attribute", new_attr)
            return pc

        result = benchmark(add_attribute)
        assert "new_attribute" in result.attribute_names


@pytest.mark.slow
class TestBenchmarkLargeScale:
    """Benchmark tests for very large point clouds (10M+ points).

    These tests are marked as 'slow' and should be run separately with:
        uv run pytest -m "slow and benchmark" --benchmark-only
    """

    @pytest.mark.benchmark(group="pcl-creation")
    def test_create_large_pointcloud(
        self, benchmark, pointcloud_impl, large_benchmark_size
    ):
        """Benchmark creating very large point cloud."""

        def create_large_pc():
            return create_pointcloud(pointcloud_impl, large_benchmark_size)

        result = benchmark(create_large_pc)
        assert result.num_points == large_benchmark_size

    @pytest.mark.benchmark(group="pcl-transformation")
    def test_transform_large_pointcloud(
        self, benchmark, pointcloud_impl, large_benchmark_size
    ):
        """Benchmark transforming very large point cloud."""
        pc = create_pointcloud(
            pointcloud_impl, large_benchmark_size, with_attributes=False
        )
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
