"""Benchmark tests for point cloud processing using pytest-benchmark.

This module contains performance benchmarks for various point cloud operations
using pytest-benchmark, which automatically generates reports and statistics.

To run benchmarks:
    uv run pytest tests/test_benchmark.py -m benchmark --benchmark-only

To view previous benchmark results:
    Check the reports/benchmarks/ directory for JSON and histogram outputs
"""

import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import create_pointcloud


@lru_cache(maxsize=None)
def _cached_pointcloud(impl: str, num_points: int, with_attributes: bool):
    """Create and cache benchmark point clouds by parameter combination."""
    return create_pointcloud(impl, num_points, with_attributes=with_attributes)


@pytest.fixture
def benchmark_pointcloud_no_attributes(pointcloud_impl, small_benchmark_size):
    """Reusable point cloud fixture without attributes for benchmark tests."""
    return _cached_pointcloud(pointcloud_impl, small_benchmark_size, False)


@pytest.fixture
def benchmark_pointcloud_with_attributes(pointcloud_impl, small_benchmark_size):
    """Reusable point cloud fixture with attributes for benchmark tests."""
    return _cached_pointcloud(pointcloud_impl, small_benchmark_size, True)


@pytest.fixture
def benchmark_pointcloud_class(pointcloud_impl):
    """Get point cloud class by implementation."""
    from framecloud.np.core import PointCloud as NpPointCloud
    from framecloud.pd.core import PointCloud as PdPointCloud

    return NpPointCloud if pointcloud_impl == "np" else PdPointCloud


@pytest.fixture
def benchmark_transform_matrix():
    """Reusable transformation matrix for benchmark tests."""
    return np.array([[2, 0, 0, 10], [0, 2, 0, 20], [0, 0, 2, 30], [0, 0, 0, 1]])


@pytest.fixture
def benchmark_new_attribute_values(small_benchmark_size):
    """Reusable attribute values for add-attribute benchmark."""
    np.random.seed(42)
    return np.random.rand(small_benchmark_size).astype(np.float32)


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
        self,
        benchmark,
        pointcloud_impl,
        benchmark_pointcloud_no_attributes,
        benchmark_transform_matrix,
        small_benchmark_size,
    ):
        """Benchmark transformation with different implementations."""
        result = benchmark(
            benchmark_pointcloud_no_attributes.transform,
            benchmark_transform_matrix,
            inplace=False,
        )
        assert result.num_points == small_benchmark_size


@pytest.mark.benchmark(group="pcl-sampling")
class TestBenchmarkSampling:
    """Benchmark tests for sampling point clouds."""

    def test_sample_pointcloud(
        self,
        benchmark,
        pointcloud_impl,
        benchmark_pointcloud_no_attributes,
        small_benchmark_size,
    ):
        """Benchmark sampling with different implementations."""
        result = benchmark(
            benchmark_pointcloud_no_attributes.sample, num_samples=10000, replace=False
        )
        assert result.num_points == 10000


@pytest.mark.benchmark(group="pcl-io")
class TestBenchmarkIO:
    """Benchmark tests for I/O operations with point clouds."""

    def test_parquet_write(self, benchmark, benchmark_pointcloud_with_attributes):
        """Benchmark parquet write with different implementations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "benchmark.parquet"
            benchmark(benchmark_pointcloud_with_attributes.to_parquet, file_path)

    def test_parquet_read(
        self,
        benchmark,
        pointcloud_impl,
        benchmark_pointcloud_with_attributes,
        benchmark_pointcloud_class,
        small_benchmark_size,
    ):
        """Benchmark parquet read with different implementations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "benchmark.parquet"
            benchmark_pointcloud_with_attributes.to_parquet(file_path)
            result = benchmark(benchmark_pointcloud_class.from_parquet, file_path)
            assert result.num_points == small_benchmark_size


@pytest.mark.benchmark(group="pcl-attributes")
class TestBenchmarkAttributeOperations:
    """Benchmark tests for attribute operations with point clouds."""

    def test_add_attribute(
        self,
        benchmark,
        benchmark_pointcloud_no_attributes,
        benchmark_new_attribute_values,
    ):
        """Benchmark adding attributes with different implementations."""

        def add_attribute():
            pc = benchmark_pointcloud_no_attributes.model_copy(deep=True)
            pc.add_attribute("new_attribute", benchmark_new_attribute_values)
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
