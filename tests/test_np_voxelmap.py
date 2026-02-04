"""Tests for the numpy-based VoxelMap class."""

import numpy as np

from framecloud.np.core import PointCloud
from framecloud.np.voxelmap import VoxelMap


class TestVoxelMapInitialization:
    """Test VoxelMap initialization and validation."""

    def test_create_voxelmap_from_pointcloud(self):
        """Test creating a VoxelMap from a PointCloud."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
                [1.5, 1.5, 1.5],
                [2.0, 2.0, 2.0],
            ]
        )
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.voxel_size == 1.0
        assert voxelmap.num_voxels == 3  # Points should be grouped into 3 voxels
        assert voxelmap.point_count == 5

    def test_voxelmap_with_empty_pointcloud(self):
        """Test creating a VoxelMap from an empty PointCloud."""
        points = np.empty((0, 3))
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels == 0
        assert voxelmap.point_count == 0

    def test_voxelmap_aggregation_first(self):
        """Test VoxelMap with 'first' aggregation method."""
        points = np.array(
            [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3], [0.7, 0.7, 0.7], [1.5, 1.5, 1.5]]
        )
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(
            pc, voxel_size=1.0, aggregation_method="first"
        )

        assert voxelmap.num_voxels == 2
        # First voxel should have 3 points, second should have 1
        assert len(voxelmap.get_point_indices((0, 0, 0))) == 3
        assert len(voxelmap.get_point_indices((1, 1, 1))) == 1

    def test_voxelmap_aggregation_nearest_to_center(self):
        """Test VoxelMap with 'nearest_to_center' aggregation method."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Far from center
                [0.5, 0.5, 0.5],  # Exactly at center of voxel
                [0.9, 0.9, 0.9],  # Far from center
            ]
        )
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(
            pc, voxel_size=1.0, aggregation_method="nearest_to_center"
        )

        assert voxelmap.num_voxels == 1
        # The point at index 1 (0.5, 0.5, 0.5) should be the representative
        assert voxelmap.representative_indices[0] == 1


class TestVoxelMapProperties:
    """Test VoxelMap properties and methods."""

    def test_num_voxels_property(self):
        """Test num_voxels property."""
        points = np.random.rand(100, 3) * 10
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels > 0
        assert voxelmap.num_voxels <= 100

    def test_get_voxel_centers(self):
        """Test getting voxel centers."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        centers = voxelmap.get_voxel_centers()
        assert centers.shape[0] == voxelmap.num_voxels
        assert centers.shape[1] == 3

    def test_get_point_indices(self):
        """Test getting point indices for a voxel."""
        points = np.array(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [1.5, 1.5, 1.5]]
        )
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        # First voxel should have 3 points
        indices = voxelmap.get_point_indices((0, 0, 0))
        assert len(indices) == 3
        assert all(i in [0, 1, 2] for i in indices)

        # Second voxel should have 1 point
        indices = voxelmap.get_point_indices((1, 1, 1))
        assert len(indices) == 1
        assert indices[0] == 3


class TestVoxelMapDownsampling:
    """Test VoxelMap downsampling functionality."""

    def test_downsample_basic(self):
        """Test basic downsampling."""
        points = np.array(
            [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3], [1.0, 1.0, 1.0], [1.5, 1.5, 1.5]]
        )
        intensities = np.array([100.0, 200.0, 300.0, 400.0])
        pc = PointCloud(points=points, attributes={"intensities": intensities})

        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)
        downsampled = voxelmap.downsample(pc)

        assert downsampled.num_points == voxelmap.num_voxels
        assert "intensities" in downsampled.attribute_names

    def test_downsample_with_custom_aggregation(self):
        """Test downsampling with custom aggregation functions."""
        points = np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6]])
        intensities = np.array([100.0, 200.0, 300.0])
        pc = PointCloud(points=points, attributes={"intensities": intensities})

        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        # Use mean aggregation for intensities
        custom_agg = {"intensities": lambda x: np.mean(x)}
        downsampled = voxelmap.downsample(pc, custom_aggregation=custom_agg)

        assert downsampled.num_points == 1
        # Mean of [100, 200, 300] = 200
        assert downsampled.intensities[0] == 200.0

    def test_downsample_preserves_attributes(self):
        """Test that downsampling preserves all attributes."""
        points = np.random.rand(50, 3) * 10
        colors = np.random.randint(0, 255, size=(50, 3))
        intensities = np.random.rand(50) * 1000
        pc = PointCloud(
            points=points, attributes={"colors": colors, "intensities": intensities}
        )

        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)
        downsampled = voxelmap.downsample(pc)

        assert set(downsampled.attribute_names) == set(pc.attribute_names)


class TestVoxelMapRecalculation:
    """Test VoxelMap recalculation functionality."""

    def test_recalculate_from_modified_pointcloud(self):
        """Test recalculating voxel map from a modified point cloud."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        pc = PointCloud(points=points)

        voxelmap1 = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        # Modify the point cloud
        pc.points = pc.points + 0.5

        # Recalculate
        voxelmap2 = voxelmap1.recalculate(pc)

        assert voxelmap2.voxel_size == voxelmap1.voxel_size
        # Due to shift, the origin should be different
        assert not np.array_equal(voxelmap1.origin, voxelmap2.origin)


class TestVoxelMapStatistics:
    """Test VoxelMap statistics functionality."""

    def test_get_statistics(self):
        """Test getting statistics about the voxel map."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],  # 3 points in voxel 0
                [1.0, 1.0, 1.0],  # 1 point in voxel 1
                [2.0, 2.0, 2.0],
                [2.1, 2.1, 2.1],  # 2 points in voxel 2
            ]
        )
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        stats = voxelmap.get_statistics()

        assert stats["num_voxels"] == 3
        assert stats["num_points"] == 6
        assert stats["voxel_size"] == 1.0
        assert stats["compression_ratio"] == 2.0
        assert stats["min_points_per_voxel"] == 1
        assert stats["max_points_per_voxel"] == 3
        assert stats["mean_points_per_voxel"] == 2.0
        assert len(stats["origin"]) == 3


class TestVoxelMapCopyBehavior:
    """Test VoxelMap copy behavior."""

    def test_keep_copy_option(self):
        """Test that keep_copy option works correctly."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)

        # Without copy
        voxelmap1 = VoxelMap.from_pointcloud(pc, voxel_size=1.0, keep_copy=False)
        assert voxelmap1.pointcloud_copy is None

        # With copy
        voxelmap2 = VoxelMap.from_pointcloud(pc, voxel_size=1.0, keep_copy=True)
        assert voxelmap2.pointcloud_copy is not None
        assert "points" in voxelmap2.pointcloud_copy
        assert "attributes" in voxelmap2.pointcloud_copy


class TestVoxelMapEdgeCases:
    """Test VoxelMap edge cases."""

    def test_single_point_cloud(self):
        """Test with a single point."""
        points = np.array([[1.5, 2.5, 3.5]])
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels == 1
        assert voxelmap.point_count == 1

    def test_large_voxel_size(self):
        """Test with a very large voxel size (all points in one voxel)."""
        points = np.random.rand(20, 3) * 10
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=100.0)

        assert voxelmap.num_voxels == 1
        assert len(voxelmap.get_point_indices(tuple(voxelmap.voxel_coords[0]))) == 20

    def test_small_voxel_size(self):
        """Test with a very small voxel size (each point in its own voxel)."""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=0.001)

        # Each point should be in its own voxel
        assert voxelmap.num_voxels == 3

    def test_negative_coordinates(self):
        """Test with negative point coordinates."""
        points = np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels == 3
        assert voxelmap.origin[0] == -1.0
