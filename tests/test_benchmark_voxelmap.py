"""Benchmark tests for VoxelMap operations using pytest-benchmark.

This module contains performance benchmarks for VoxelMap operations
using pytest-benchmark.

To run benchmarks:
    uv run pytest tests/test_benchmark_voxelmap.py -m benchmark --benchmark-only

To view previous benchmark results:
    Check the reports/benchmarks/ directory for JSON and histogram outputs
"""

import numpy as np
import pandas as pd
import pytest

from framecloud.np.core import PointCloud as NpPointCloud
from framecloud.np.voxelmap import VoxelMap as NpVoxelMap
from framecloud.pd.core import PointCloud as PdPointCloud
from framecloud.pd.voxelmap import VoxelMap as PdVoxelMap


@pytest.fixture(
    params=[
        10_000,
        100_000,
        1_000_000,
    ]
)
def small_benchmark_size(request):
    """Parametrized fixture for smaller benchmark point cloud sizes."""
    return request.param


@pytest.fixture(
    params=[
        5_000_000,
        10_000_000,
    ]
)
def large_benchmark_size(request):
    """Parametrized fixture for large benchmark point cloud sizes."""
    return request.param


@pytest.mark.benchmark(
    group="voxelmap",
    min_time=0.1,
    max_time=5.0,
    min_rounds=1,
    disable_gc=True,
    warmup=False,
)
class TestBenchmarkVoxelMap:
    """Benchmark tests for VoxelMap operations."""

    def test_np_voxelmap_creation(self, benchmark, small_benchmark_size):
        """Benchmark creating VoxelMap with numpy implementation."""
        np.random.seed(42)
        points = np.random.randn(small_benchmark_size, 3).astype(np.float32) * 100
        pc = NpPointCloud(points=points)

        result = benchmark(NpVoxelMap.from_pointcloud, pc, voxel_size=1.0)
        assert result.num_voxels > 0

    def test_pd_voxelmap_creation(self, benchmark, small_benchmark_size):
        """Benchmark creating VoxelMap with pandas implementation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(small_benchmark_size).astype(np.float32) * 100,
                "Y": np.random.randn(small_benchmark_size).astype(np.float32) * 100,
                "Z": np.random.randn(small_benchmark_size).astype(np.float32) * 100,
            }
        )
        pc = PdPointCloud(data=df)

        result = benchmark(PdVoxelMap.from_pointcloud, pc, voxel_size=1.0)
        assert result.num_voxels > 0

    def test_np_voxelmap_export(self, benchmark, small_benchmark_size):
        """Benchmark exporting point cloud from VoxelMap with numpy."""
        np.random.seed(42)
        points = np.random.randn(small_benchmark_size, 3).astype(np.float32) * 100
        intensities = np.random.rand(small_benchmark_size).astype(np.float32)
        pc = NpPointCloud(points=points, attributes={"intensities": intensities})
        voxelmap = NpVoxelMap.from_pointcloud(pc, voxel_size=1.0)

        result = benchmark(voxelmap.export_pointcloud)
        assert result.num_points == voxelmap.num_voxels

    def test_pd_voxelmap_export(self, benchmark, small_benchmark_size):
        """Benchmark exporting point cloud from VoxelMap with pandas."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(small_benchmark_size).astype(np.float32) * 100,
                "Y": np.random.randn(small_benchmark_size).astype(np.float32) * 100,
                "Z": np.random.randn(small_benchmark_size).astype(np.float32) * 100,
                "intensities": np.random.rand(small_benchmark_size).astype(np.float32),
            }
        )
        pc = PdPointCloud(data=df)
        voxelmap = PdVoxelMap.from_pointcloud(pc, voxel_size=1.0)

        result = benchmark(voxelmap.export_pointcloud)
        assert result.num_points == voxelmap.num_voxels

    def test_np_voxelmap_export_nearest_to_center(
        self, benchmark, small_benchmark_size
    ):
        """Benchmark exporting with nearest_to_center aggregation (numpy)."""
        np.random.seed(42)
        points = np.random.randn(small_benchmark_size, 3).astype(np.float32) * 100
        pc = NpPointCloud(points=points)
        voxelmap = NpVoxelMap.from_pointcloud(pc, voxel_size=1.0)

        result = benchmark(
            voxelmap.export_pointcloud, aggregation_method="nearest_to_center"
        )
        assert result.num_points == voxelmap.num_voxels

    def test_pd_voxelmap_export_nearest_to_center(
        self, benchmark, small_benchmark_size
    ):
        """Benchmark exporting with nearest_to_center aggregation (pandas)."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(small_benchmark_size).astype(np.float32) * 100,
                "Y": np.random.randn(small_benchmark_size).astype(np.float32) * 100,
                "Z": np.random.randn(small_benchmark_size).astype(np.float32) * 100,
            }
        )
        pc = PdPointCloud(data=df)
        voxelmap = PdVoxelMap.from_pointcloud(pc, voxel_size=1.0)

        result = benchmark(
            voxelmap.export_pointcloud, aggregation_method="nearest_to_center"
        )
        assert result.num_points == voxelmap.num_voxels


@pytest.mark.slow
@pytest.mark.benchmark(
    group="voxelmap-large",
    min_time=0.1,
    max_time=5.0,
    min_rounds=1,
    disable_gc=True,
    warmup=False,
)
class TestBenchmarkVoxelMapLargeScale:
    """Benchmark tests for VoxelMap with large point clouds.

    These tests are marked as 'slow' and should be run separately with:
        uv run pytest -m "slow and benchmark" --benchmark-only
    """

    def test_np_voxelmap_large_creation(self, benchmark, large_benchmark_size):
        """Benchmark creating VoxelMap with large point cloud (numpy)."""
        np.random.seed(42)
        points = np.random.randn(large_benchmark_size, 3).astype(np.float32) * 100
        pc = NpPointCloud(points=points)

        result = benchmark(NpVoxelMap.from_pointcloud, pc, voxel_size=1.0)
        assert result.num_voxels > 0

    def test_pd_voxelmap_large_creation(self, benchmark, large_benchmark_size):
        """Benchmark creating VoxelMap with large point cloud (pandas)."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(large_benchmark_size).astype(np.float32) * 100,
                "Y": np.random.randn(large_benchmark_size).astype(np.float32) * 100,
                "Z": np.random.randn(large_benchmark_size).astype(np.float32) * 100,
            }
        )
        pc = PdPointCloud(data=df)

        result = benchmark(PdVoxelMap.from_pointcloud, pc, voxel_size=1.0)
        assert result.num_voxels > 0

    def test_np_voxelmap_large_export(self, benchmark, large_benchmark_size):
        """Benchmark exporting from large VoxelMap (numpy)."""
        np.random.seed(42)
        points = np.random.randn(large_benchmark_size, 3).astype(np.float32) * 100
        pc = NpPointCloud(points=points)
        voxelmap = NpVoxelMap.from_pointcloud(pc, voxel_size=1.0)

        result = benchmark(voxelmap.export_pointcloud)
        assert result.num_points == voxelmap.num_voxels

    def test_pd_voxelmap_large_export(self, benchmark, large_benchmark_size):
        """Benchmark exporting from large VoxelMap (pandas)."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "X": np.random.randn(large_benchmark_size).astype(np.float32) * 100,
                "Y": np.random.randn(large_benchmark_size).astype(np.float32) * 100,
                "Z": np.random.randn(large_benchmark_size).astype(np.float32) * 100,
            }
        )
        pc = PdPointCloud(data=df)
        voxelmap = PdVoxelMap.from_pointcloud(pc, voxel_size=1.0)

        result = benchmark(voxelmap.export_pointcloud)
        assert result.num_points == voxelmap.num_voxels
