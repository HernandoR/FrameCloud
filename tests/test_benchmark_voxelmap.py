"""Benchmark tests for VoxelMap operations using pytest-benchmark.

This module contains performance benchmarks for VoxelMap operations
using pytest-benchmark.

To run benchmarks:
    uv run pytest tests/test_benchmark_voxelmap.py -m benchmark --benchmark-only

To view previous benchmark results:
    Check the reports/benchmarks/ directory for JSON and histogram outputs
"""

from functools import lru_cache

import pytest

from framecloud.np.voxelmap import VoxelMap as NpVoxelMap
from framecloud.pd.voxelmap import VoxelMap as PdVoxelMap
from tests.conftest import create_voxelmap_pointcloud


@pytest.fixture
def voxelmap_class(pointcloud_impl):
    """Get VoxelMap class by implementation."""
    return NpVoxelMap if pointcloud_impl == "np" else PdVoxelMap


@lru_cache(maxsize=None)
def _cached_voxelmap_pointcloud(impl: str, num_points: int, with_attributes: bool):
    """Create and cache benchmark point clouds for voxel map tests."""
    return create_voxelmap_pointcloud(impl, num_points, with_attributes=with_attributes)


@pytest.fixture
def voxelmap_small_pointcloud_no_attributes(pointcloud_impl, voxelmap_small_size):
    """Reusable small point cloud for voxel map creation benchmark."""
    return _cached_voxelmap_pointcloud(pointcloud_impl, voxelmap_small_size, False)


@pytest.fixture
def voxelmap_large_pointcloud_no_attributes(pointcloud_impl, voxelmap_large_size):
    """Reusable large point cloud for voxel map benchmarks."""
    return _cached_voxelmap_pointcloud(pointcloud_impl, voxelmap_large_size, False)


@pytest.mark.benchmark(
    min_time=0.1,
    max_time=60.0,
    min_rounds=1,
    warmup=False,
)
class TestBenchmarkVoxelMap:
    """Benchmark tests for VoxelMap operations."""

    @pytest.mark.benchmark(group="voxelmap-creation")
    def test_voxelmap_creation(
        self,
        benchmark,
        voxelmap_class,
        voxelmap_small_pointcloud_no_attributes,
    ):
        """Benchmark creating VoxelMap with different implementations."""
        result = benchmark(
            voxelmap_class.from_pointcloud,
            voxelmap_small_pointcloud_no_attributes,
            voxel_size=1.0,
        )
        assert result.num_voxels > 0

    @pytest.mark.benchmark(group="voxelmap-export")
    def test_voxelmap_export(
        self,
        benchmark,
        voxelmap_class,
        voxelmap_small_pointcloud_no_attributes,
    ):
        """Benchmark exporting point cloud from VoxelMap."""

        def setup():
            return (
                voxelmap_class.from_pointcloud(
                    voxelmap_small_pointcloud_no_attributes, voxel_size=1.0
                ),
            ), {}

        result = benchmark.pedantic(
            lambda voxelmap: voxelmap.export_pointcloud(),
            setup=setup,
            rounds=1,
            iterations=1,
        )
        assert result.num_points > 0

    @pytest.mark.benchmark(group="voxelmap-export")
    def test_voxelmap_export_nearest_to_center(
        self,
        benchmark,
        voxelmap_class,
        voxelmap_small_pointcloud_no_attributes,
    ):
        """Benchmark exporting with nearest_to_center aggregation."""

        voxelmap = voxelmap_class.from_pointcloud(
            voxelmap_small_pointcloud_no_attributes, voxel_size=1.0
        )

        result = benchmark(
            voxelmap.export_pointcloud, aggregation_method="nearest_to_center"
        )
        assert result.num_points == voxelmap.num_voxels

        # def setup():
        #     return (
        #         voxelmap_class.from_pointcloud(
        #             voxelmap_small_pointcloud_no_attributes, voxel_size=1.0
        #         ),
        #     ), {}

        # result = benchmark.pedantic(
        #     lambda voxelmap: voxelmap.export_pointcloud(
        #         aggregation_method="nearest_to_center"
        #     ),
        #     setup=setup,
        #     rounds=1,
        #     iterations=1,
        # )
        # assert result.num_points > 0


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
        self,
        benchmark,
        voxelmap_class,
        voxelmap_large_pointcloud_no_attributes,
    ):
        """Benchmark creating VoxelMap with large point cloud."""
        result = benchmark(
            voxelmap_class.from_pointcloud,
            voxelmap_large_pointcloud_no_attributes,
            voxel_size=1.0,
        )
        assert result.num_voxels > 0

    @pytest.mark.benchmark(group="voxelmap-export")
    def test_voxelmap_large_export(
        self,
        benchmark,
        voxelmap_class,
        voxelmap_large_pointcloud_no_attributes,
    ):
        """Benchmark exporting from large VoxelMap."""

        def setup():
            return (
                voxelmap_class.from_pointcloud(
                    voxelmap_large_pointcloud_no_attributes, voxel_size=1.0
                ),
            ), {}

        result = benchmark.pedantic(
            lambda voxelmap: voxelmap.export_pointcloud(),
            setup=setup,
            rounds=1,
            iterations=1,
        )

        assert result.num_points > 0


def main():
    # pytest.main(["-k", "pd and 5M", "-m", "benchmark", "--benchmark-only"])
    # run directly without pytest\
    # pd 5M
    pc = create_voxelmap_pointcloud("pd", 5_000_000, with_attributes=False)
    voxelmap = PdVoxelMap.from_pointcloud(pc, voxel_size=1.0)
    result = voxelmap.export_pointcloud()
    print(f"Exported {result.num_points} points from VoxelMap with 5M input points.")


if __name__ == "__main__":
    main()
