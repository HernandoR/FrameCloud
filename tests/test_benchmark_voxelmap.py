"""Benchmark tests for VoxelMap operations using pytest-benchmark.

This module contains performance benchmarks for VoxelMap operations
using pytest-benchmark.

To run benchmarks:
    uv run pytest tests/test_benchmark_voxelmap.py -m benchmark --benchmark-only

To view previous benchmark results:
    Check the reports/benchmarks/ directory for JSON and histogram outputs
"""

import pytest

from framecloud.np.voxelmap import VoxelMap as NpVoxelMap
from framecloud.pd.voxelmap import VoxelMap as PdVoxelMap
from tests.conftest import create_voxelmap_pointcloud


@pytest.mark.benchmark(
    min_time=0.1,
    max_time=60.0,
    min_rounds=1,
    warmup=False,
)
class TestBenchmarkVoxelMap:
    """Benchmark tests for VoxelMap operations."""

    @pytest.mark.benchmark(group="voxelmap-creation")
    def test_voxelmap_creation(self, benchmark, pointcloud_impl, voxelmap_small_size):
        """Benchmark creating VoxelMap with different implementations."""
        pc = create_voxelmap_pointcloud(
            pointcloud_impl, voxelmap_small_size, with_attributes=False
        )
        VoxelMapClass = NpVoxelMap if pointcloud_impl == "np" else PdVoxelMap

        result = benchmark(VoxelMapClass.from_pointcloud, pc, voxel_size=1.0)
        assert result.num_voxels > 0

    @pytest.mark.benchmark(group="voxelmap-export")
    def test_voxelmap_export(self, benchmark, pointcloud_impl, voxelmap_small_size):
        """Benchmark exporting point cloud from VoxelMap."""
        pc = create_voxelmap_pointcloud(pointcloud_impl, voxelmap_small_size)
        VoxelMapClass = NpVoxelMap if pointcloud_impl == "np" else PdVoxelMap
        voxelmap = VoxelMapClass.from_pointcloud(pc, voxel_size=1.0)

        result = benchmark(voxelmap.export_pointcloud)
        assert result.num_points == voxelmap.num_voxels

    @pytest.mark.benchmark(group="voxelmap-export")
    def test_voxelmap_export_nearest_to_center(
        self, benchmark, pointcloud_impl, voxelmap_small_size
    ):
        """Benchmark exporting with nearest_to_center aggregation."""
        pc = create_voxelmap_pointcloud(
            pointcloud_impl, voxelmap_small_size, with_attributes=False
        )
        VoxelMapClass = NpVoxelMap if pointcloud_impl == "np" else PdVoxelMap
        voxelmap = VoxelMapClass.from_pointcloud(pc, voxel_size=1.0)

        result = benchmark(
            voxelmap.export_pointcloud, aggregation_method="nearest_to_center"
        )
        assert result.num_points == voxelmap.num_voxels


@pytest.mark.slow
@pytest.mark.benchmark(
    min_time=0.1,
    max_time=60.0,
    min_rounds=1,
    warmup=False,
)
class TestBenchmarkVoxelMapLargeScale:
    """Benchmark tests for VoxelMap with large point clouds.

    These tests are marked as 'slow' and should be run separately with:
        uv run pytest -m "slow and benchmark" --benchmark-only
    """

    @pytest.mark.benchmark(group="voxelmap-creation")
    def test_voxelmap_large_creation(
        self, benchmark, pointcloud_impl, voxelmap_large_size
    ):
        """Benchmark creating VoxelMap with large point cloud."""
        pc = create_voxelmap_pointcloud(
            pointcloud_impl, voxelmap_large_size, with_attributes=False
        )
        VoxelMapClass = NpVoxelMap if pointcloud_impl == "np" else PdVoxelMap

        result = benchmark(VoxelMapClass.from_pointcloud, pc, voxel_size=1.0)
        assert result.num_voxels > 0

    @pytest.mark.benchmark(group="voxelmap-export")
    def test_voxelmap_large_export(
        self, benchmark, pointcloud_impl, voxelmap_large_size
    ):
        """Benchmark exporting from large VoxelMap."""
        pc = create_voxelmap_pointcloud(
            pointcloud_impl, voxelmap_large_size, with_attributes=False
        )
        VoxelMapClass = NpVoxelMap if pointcloud_impl == "np" else PdVoxelMap
        voxelmap = VoxelMapClass.from_pointcloud(pc, voxel_size=1.0)

        result = benchmark(voxelmap.export_pointcloud)
        assert result.num_points == voxelmap.num_voxels
