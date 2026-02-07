"""Tests for the pandas-based VoxelMap class."""

import numpy as np
import pandas as pd

from framecloud.pd.core import PointCloud
from framecloud.pd.voxelmap import VoxelMap


class TestVoxelMapInitialization:
    """Test VoxelMap initialization and validation."""

    def test_create_voxelmap_from_pointcloud(self):
        """Test creating a VoxelMap from a PointCloud."""
        data = pd.DataFrame(
            {
                "X": [0.0, 0.5, 1.0, 1.5, 2.0],
                "Y": [0.0, 0.5, 1.0, 1.5, 2.0],
                "Z": [0.0, 0.5, 1.0, 1.5, 2.0],
            }
        )
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.voxel_size == 1.0
        assert voxelmap.num_voxels == 3  # Points should be grouped into 3 voxels
        assert voxelmap.pointcloud.num_points == 5

    def test_voxelmap_with_empty_pointcloud(self):
        """Test creating a VoxelMap from an empty PointCloud."""
        data = pd.DataFrame({"X": [], "Y": [], "Z": []})
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels == 0
        assert voxelmap.pointcloud.num_points == 0

    def test_voxelmap_aggregation_first(self):
        """Test VoxelMap with 'first' aggregation method."""
        data = pd.DataFrame(
            {
                "X": [0.0, 0.3, 0.7, 1.5],
                "Y": [0.0, 0.3, 0.7, 1.5],
                "Z": [0.0, 0.3, 0.7, 1.5],
            }
        )
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels == 2
        # First voxel should have 3 points, second should have 1
        assert len(voxelmap.get_point_indices((0, 0, 0))) == 3
        assert len(voxelmap.get_point_indices((1, 1, 1))) == 1

    def test_voxelmap_aggregation_nearest_to_center(self):
        """Test VoxelMap with 'nearest_to_center' aggregation method."""
        data = pd.DataFrame(
            {
                "X": [0.0, 0.5, 0.9],  # 0.5 is exactly at center
                "Y": [0.0, 0.5, 0.9],
                "Z": [0.0, 0.5, 0.9],
            }
        )
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels == 1
        # Export with nearest_to_center to check which point is selected
        downsampled = voxelmap.export_pointcloud(aggregation_method="nearest_to_center")
        # The point at index 1 (0.5, 0.5, 0.5) should be the representative
        assert downsampled.data["X"].iloc[0] == 0.5


class TestVoxelMapProperties:
    """Test VoxelMap properties and methods."""

    def test_num_voxels_property(self):
        """Test num_voxels property."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "X": np.random.rand(100) * 10,
                "Y": np.random.rand(100) * 10,
                "Z": np.random.rand(100) * 10,
            }
        )
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels > 0
        assert voxelmap.num_voxels <= 100

    def test_voxel_coords_property(self):
        """Test voxel_coords property."""
        data = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        coords = voxelmap.voxel_coords
        assert coords.shape[0] == voxelmap.num_voxels
        assert coords.shape[1] == 3

    def test_get_voxel_centers(self):
        """Test getting voxel centers."""
        data = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        centers = voxelmap.get_voxel_centers()
        assert centers.shape[0] == voxelmap.num_voxels
        assert centers.shape[1] == 3

    def test_get_point_indices(self):
        """Test getting point indices for a voxel."""
        data = pd.DataFrame(
            {
                "X": [0.1, 0.2, 0.3, 1.5],
                "Y": [0.1, 0.2, 0.3, 1.5],
                "Z": [0.1, 0.2, 0.3, 1.5],
            }
        )
        pc = PointCloud(data=data)
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
        data = pd.DataFrame(
            {
                "X": [0.0, 0.3, 1.0, 1.5],
                "Y": [0.0, 0.3, 1.0, 1.5],
                "Z": [0.0, 0.3, 1.0, 1.5],
                "intensity": [100.0, 200.0, 300.0, 400.0],
            }
        )
        pc = PointCloud(data=data)

        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)
        downsampled = voxelmap.export_pointcloud()

        assert downsampled.num_points == voxelmap.num_voxels
        assert "intensity" in downsampled.attribute_names

    def test_downsample_with_custom_aggregation(self):
        """Test downsampling with custom aggregation functions."""
        data = pd.DataFrame(
            {
                "X": [0.0, 0.3, 0.6],
                "Y": [0.0, 0.3, 0.6],
                "Z": [0.0, 0.3, 0.6],
                "intensity": [100.0, 200.0, 300.0],
            }
        )
        pc = PointCloud(data=data)

        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        # Use mean aggregation for intensity
        custom_agg = {"intensity": lambda x: x.mean()}
        downsampled = voxelmap.export_pointcloud(custom_aggregation=custom_agg)

        assert downsampled.num_points == 1
        # Mean of [100, 200, 300] = 200
        assert downsampled.data["intensity"].iloc[0] == 200.0

    def test_downsample_preserves_attributes(self):
        """Test that downsampling preserves all attributes."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "X": np.random.rand(50) * 10,
                "Y": np.random.rand(50) * 10,
                "Z": np.random.rand(50) * 10,
                "color_r": np.random.randint(0, 255, 50),
                "color_g": np.random.randint(0, 255, 50),
                "color_b": np.random.randint(0, 255, 50),
                "intensity": np.random.rand(50) * 1000,
            }
        )
        pc = PointCloud(data=data)

        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)
        downsampled = voxelmap.export_pointcloud()

        assert set(downsampled.attribute_names) == set(pc.attribute_names)


class TestVoxelMapRecalculation:
    """Test VoxelMap recalculation functionality."""

    def test_refresh_from_modified_pointcloud(self):
        """Test refreshing voxel map from a modified point cloud."""
        data = pd.DataFrame(
            {"X": [0.0, 1.0, 2.0], "Y": [0.0, 1.0, 2.0], "Z": [0.0, 1.0, 2.0]}
        )
        pc = PointCloud(data=data)

        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)
        origin1 = voxelmap.origin.copy()

        # Modify the point cloud
        pc.data[["X", "Y", "Z"]] = pc.data[["X", "Y", "Z"]] + 0.5

        # Refresh voxel map
        voxelmap.refresh()

        # Due to shift, the origin should be different
        assert not np.array_equal(origin1, voxelmap.origin)


class TestVoxelMapStatistics:
    """Test VoxelMap statistics functionality."""

    def test_get_statistics(self):
        """Test getting statistics about the voxel map."""
        data = pd.DataFrame(
            {
                "X": [0.0, 0.1, 0.2, 1.0, 2.0, 2.1],
                "Y": [0.0, 0.1, 0.2, 1.0, 2.0, 2.1],
                "Z": [0.0, 0.1, 0.2, 1.0, 2.0, 2.1],
            }
        )
        pc = PointCloud(data=data)
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
        data = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=data)

        # Without copy
        voxelmap1 = VoxelMap.from_pointcloud(pc, voxel_size=1.0, keep_copy=False)
        assert voxelmap1.pointcloud is pc  # Should be same reference
        assert not voxelmap1.is_copy  # Should not be a copy

        # With copy
        voxelmap2 = VoxelMap.from_pointcloud(pc, voxel_size=1.0, keep_copy=True)
        assert voxelmap2.pointcloud is not pc  # Should be different reference
        assert voxelmap2.is_copy  # Should be a copy
        # But data should be equal
        pd.testing.assert_frame_equal(voxelmap2.pointcloud.data, pc.data)


class TestVoxelMapEdgeCases:
    """Test VoxelMap edge cases."""

    def test_single_point_cloud(self):
        """Test with a single point."""
        data = pd.DataFrame({"X": [1.5], "Y": [2.5], "Z": [3.5]})
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels == 1
        assert voxelmap.pointcloud.num_points == 1

    def test_large_voxel_size(self):
        """Test with a very large voxel size (all points in one voxel)."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "X": np.random.rand(20) * 10,
                "Y": np.random.rand(20) * 10,
                "Z": np.random.rand(20) * 10,
            }
        )
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=100.0)

        assert voxelmap.num_voxels == 1
        assert len(voxelmap.get_point_indices(tuple(voxelmap.voxel_coords[0]))) == 20

    def test_small_voxel_size(self):
        """Test with a very small voxel size (each point in its own voxel)."""
        data = pd.DataFrame(
            {"X": [0.0, 10.0, 20.0], "Y": [0.0, 10.0, 20.0], "Z": [0.0, 10.0, 20.0]}
        )
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=0.001)

        # Each point should be in its own voxel
        assert voxelmap.num_voxels == 3

    def test_negative_coordinates(self):
        """Test with negative point coordinates."""
        data = pd.DataFrame(
            {"X": [-1.0, 0.0, 1.0], "Y": [-1.0, 0.0, 1.0], "Z": [-1.0, 0.0, 1.0]}
        )
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels == 3
        assert voxelmap.origin[0] == -1.0

    def test_empty_voxelmap_export(self):
        """Test exporting from empty voxel map."""
        data = pd.DataFrame({"X": [], "Y": [], "Z": []})
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        downsampled = voxelmap.export_pointcloud()
        assert len(downsampled.data) == 0

    def test_coordinate_column_protection(self):
        """Test that coordinate columns cannot be overridden in custom_aggregation."""
        data = pd.DataFrame(
            {
                "X": [0.0, 0.5, 1.0],
                "Y": [0.0, 0.5, 1.0],
                "Z": [0.0, 0.5, 1.0],
                "intensity": [100.0, 200.0, 300.0],
            }
        )
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        # Should raise ValueError when trying to aggregate coordinate columns
        import pytest

        with pytest.raises(ValueError, match="coordinate columns"):
            voxelmap.export_pointcloud(custom_aggregation={"X": lambda x: x.mean()})

        with pytest.raises(ValueError, match="coordinate columns"):
            voxelmap.export_pointcloud(
                custom_aggregation={
                    "Y": lambda x: x.mean(),
                    "intensity": lambda x: x.mean(),
                }
            )

        with pytest.raises(ValueError, match="coordinate columns"):
            voxelmap.export_pointcloud(custom_aggregation={"Z": lambda x: x.mean()})

    def test_dataframe_with_extra_attributes(self):
        """Test with DataFrame containing multiple attributes."""
        data = pd.DataFrame(
            {
                "X": [0.0, 0.3, 1.0],
                "Y": [0.0, 0.3, 1.0],
                "Z": [0.0, 0.3, 1.0],
                "intensity": [100, 200, 300],
                "classification": [1, 2, 3],
            }
        )
        pc = PointCloud(data=data)
        voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

        assert voxelmap.num_voxels == 2
        downsampled = voxelmap.export_pointcloud()
        assert "intensity" in downsampled.attribute_names
        assert "classification" in downsampled.attribute_names
