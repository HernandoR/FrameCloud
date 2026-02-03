"""Tests for the pd.PointCloud class."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from framecloud.pd.core import ArrayShapeError, AttributeExistsError, PointCloud


class TestPointCloudInitialization:
    """Test PointCloud initialization and validation."""

    def test_create_basic_point_cloud(self):
        """Test creating a basic point cloud with valid DataFrame."""
        df = pd.DataFrame(
            {"X": [0.0, 1.0, 2.0], "Y": [0.0, 1.0, 2.0], "Z": [0.0, 1.0, 2.0]}
        )
        pc = PointCloud(data=df)
        assert pc.num_points == 3
        assert len(pc.attribute_names) == 0

    def test_create_point_cloud_with_attributes(self):
        """Test creating a point cloud with attributes."""
        df = pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y": [0.0, 1.0],
                "Z": [0.0, 1.0],
                "colors_0": [255, 0],
                "colors_1": [0, 255],
                "colors_2": [0, 0],
            }
        )
        pc = PointCloud(data=df)
        assert pc.num_points == 2
        assert "colors_0" in pc.attribute_names

    def test_invalid_missing_xyz_columns(self):
        """Test that DataFrame must have X, Y, Z columns."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0]})  # Missing Z
        with pytest.raises(ValidationError, match="must have X, Y, Z columns"):
            PointCloud(data=df)

    def test_invalid_not_dataframe(self):
        """Test that data must be a DataFrame."""
        with pytest.raises(ValidationError):
            PointCloud(data=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def test_empty_point_cloud(self):
        """Test creating an empty point cloud."""
        df = pd.DataFrame({"X": [], "Y": [], "Z": []})
        pc = PointCloud(data=df)
        assert pc.num_points == 0


class TestPointCloudProperties:
    """Test PointCloud properties."""

    def test_num_points_property(self):
        """Test num_points property."""
        df = pd.DataFrame(
            {"X": [0.0, 1.0, 2.0], "Y": [0.0, 1.0, 2.0], "Z": [0.0, 1.0, 2.0]}
        )
        pc = PointCloud(data=df)
        assert pc.num_points == 3

    def test_attribute_names_property(self):
        """Test attribute_names property."""
        df = pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y": [0.0, 1.0],
                "Z": [0.0, 1.0],
                "colors": [255, 0],
                "intensities": [1.0, 2.0],
            }
        )
        pc = PointCloud(data=df)
        assert set(pc.attribute_names) == {"colors", "intensities"}

    def test_attribute_names_empty(self):
        """Test attribute_names property when no attributes."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)
        assert pc.attribute_names == []

    def test_points_property(self):
        """Test points property returns numpy array."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)
        points = pc.points
        assert isinstance(points, np.ndarray)
        assert points.shape == (2, 3)

    def test_attributes_property(self):
        """Test attributes property returns dict of numpy arrays."""
        df = pd.DataFrame(
            {"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0], "colors": [255, 0]}
        )
        pc = PointCloud(data=df)
        attrs = pc.attributes
        assert isinstance(attrs, dict)
        assert "colors" in attrs
        assert isinstance(attrs["colors"], np.ndarray)


class TestPointCloudAttributeOperations:
    """Test attribute management operations."""

    def test_add_attribute(self):
        """Test adding a new attribute."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)
        intensities = np.array([1.0, 2.0])
        pc.add_attribute("intensities", intensities)
        assert "intensities" in pc.attribute_names
        np.testing.assert_array_equal(pc.data["intensities"].to_numpy(), intensities)

    def test_add_duplicate_attribute(self):
        """Test that adding duplicate attribute raises error."""
        df = pd.DataFrame(
            {"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0], "colors": [255, 0]}
        )
        pc = PointCloud(data=df)
        new_colors = np.array([0, 255])
        with pytest.raises(AttributeExistsError):
            pc.add_attribute("colors", new_colors)

    def test_set_attribute_overwrites(self):
        """Test that set_attribute overwrites existing attribute."""
        df = pd.DataFrame(
            {"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0], "colors": [255, 0]}
        )
        pc = PointCloud(data=df)
        new_colors = np.array([0, 255])
        pc.set_attribute("colors", new_colors)
        np.testing.assert_array_equal(pc.data["colors"].to_numpy(), new_colors)

    def test_set_new_attribute(self):
        """Test that set_attribute can add new attributes."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)
        intensities = np.array([1.0, 2.0])
        pc.set_attribute("intensities", intensities)
        assert "intensities" in pc.attribute_names

    def test_set_attribute_wrong_length(self):
        """Test that set_attribute with wrong length raises error."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)
        wrong_intensities = np.array([1.0])  # Only 1 value for 2 points
        with pytest.raises(ArrayShapeError, match="does not match number of points"):
            pc.set_attribute("intensities", wrong_intensities)

    def test_remove_attribute(self):
        """Test removing an attribute."""
        df = pd.DataFrame(
            {"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0], "colors": [255, 0]}
        )
        pc = PointCloud(data=df)
        pc.remove_attribute("colors")
        assert "colors" not in pc.attribute_names

    def test_remove_nonexistent_attribute(self):
        """Test that removing nonexistent attribute doesn't raise error."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)
        pc.remove_attribute("nonexistent")  # Should not raise

    def test_get_attribute(self):
        """Test getting an attribute."""
        df = pd.DataFrame(
            {"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0], "colors": [255, 0]}
        )
        pc = PointCloud(data=df)
        retrieved_colors = pc.get_attribute("colors")
        np.testing.assert_array_equal(retrieved_colors, np.array([255, 0]))

    def test_get_nonexistent_attribute(self):
        """Test getting a nonexistent attribute returns None."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)
        assert pc.get_attribute("nonexistent") is None


class TestPointCloudTransformation:
    """Test transformation operations."""

    def test_translation_transform(self):
        """Test translation transformation."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)

        # Translation matrix (move by 1.0 in each direction)
        matrix = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

        transformed = pc.transform(matrix, inplace=False)
        expected = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        np.testing.assert_array_almost_equal(transformed.points, expected)

    def test_scale_transform(self):
        """Test scale transformation."""
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [1.0, 2.0], "Z": [1.0, 2.0]})
        pc = PointCloud(data=df)

        # Scale matrix (scale by 2.0)
        matrix = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])

        transformed = pc.transform(matrix, inplace=False)
        expected = np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
        np.testing.assert_array_almost_equal(transformed.points, expected)

    def test_transform_inplace(self):
        """Test in-place transformation."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)

        matrix = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

        result = pc.transform(matrix, inplace=True)
        expected = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        np.testing.assert_array_almost_equal(pc.points, expected)
        assert result is None  # inplace should return None

    def test_transform_with_attributes(self):
        """Test that attributes are preserved during transformation."""
        df = pd.DataFrame(
            {"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0], "colors": [255, 0]}
        )
        pc = PointCloud(data=df)

        matrix = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

        transformed = pc.transform(matrix, inplace=False)
        assert "colors" in transformed.attribute_names
        np.testing.assert_array_equal(
            transformed.data["colors"].to_numpy(), np.array([255, 0])
        )

    def test_invalid_transformation_matrix(self):
        """Test that invalid transformation matrix raises error."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)

        invalid_matrix = np.array([[1, 0], [0, 1]])  # 2x2 instead of 4x4
        with pytest.raises(ArrayShapeError, match="4x4"):
            pc.transform(invalid_matrix)


class TestPointCloudCopy:
    """Test copy operations."""

    def test_copy_creates_new_instance(self):
        """Test that copy creates a new instance."""
        df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]})
        pc = PointCloud(data=df)
        pc_copy = pc.copy()
        assert pc is not pc_copy
        assert pc.data is not pc_copy.data
        np.testing.assert_array_equal(pc.points, pc_copy.points)

    def test_copy_is_deep(self):
        """Test that copy is deep (modifying copy doesn't affect original)."""
        df = pd.DataFrame(
            {"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0], "colors": [255, 0]}
        )
        pc = PointCloud(data=df)
        pc_copy = pc.copy()

        # Modify the copy
        pc_copy.data.loc[0, "X"] = 999.0
        pc_copy.data.loc[0, "colors"] = 0

        # Original should be unchanged
        assert pc.data.loc[0, "X"] == 0.0
        assert pc.data.loc[0, "colors"] == 255


class TestPointCloudSampling:
    """Test sampling operations."""

    def test_sample_without_replacement(self):
        """Test sampling without replacement."""
        df = pd.DataFrame({"X": np.arange(10), "Y": np.arange(10), "Z": np.arange(10)})
        pc = PointCloud(data=df)
        sampled = pc.sample(num_samples=5, replace=False)
        assert sampled.num_points == 5

    def test_sample_with_replacement(self):
        """Test sampling with replacement."""
        df = pd.DataFrame({"X": np.arange(10), "Y": np.arange(10), "Z": np.arange(10)})
        pc = PointCloud(data=df)
        sampled = pc.sample(num_samples=15, replace=True)
        assert sampled.num_points == 15

    def test_sample_more_than_available_without_replacement(self):
        """Test that sampling more than available without replacement raises error."""
        df = pd.DataFrame({"X": np.arange(10), "Y": np.arange(10), "Z": np.arange(10)})
        pc = PointCloud(data=df)
        with pytest.raises(ValueError, match="exceeds number of points"):
            pc.sample(num_samples=15, replace=False)

    def test_sample_preserves_attributes(self):
        """Test that sampling preserves attributes."""
        df = pd.DataFrame(
            {
                "X": np.arange(10),
                "Y": np.arange(10),
                "Z": np.arange(10),
                "colors": np.arange(10),
            }
        )
        pc = PointCloud(data=df)
        sampled = pc.sample(num_samples=5, replace=False)
        assert "colors" in sampled.attribute_names
        assert len(sampled.data["colors"]) == 5


class TestPointCloudToDict:
    """Test dict conversion."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        df = pd.DataFrame(
            {"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0], "colors": [255, 0]}
        )
        pc = PointCloud(data=df)

        result = pc.to_dict()
        assert "points" in result
        assert "attributes" in result
        assert result["points"].shape == (2, 3)
        assert "colors" in result["attributes"]


class TestPointCloudCrossCheck:
    """Cross-check pd.PointCloud against np.PointCloud using fixtures."""

    def test_small_pointcloud_basic_properties(self, small_point_cloud_np):
        """Test that basic properties match between np and pd implementations."""
        from tests.conftest import np_to_pd_pointcloud

        pd_pc = np_to_pd_pointcloud(small_point_cloud_np)

        assert pd_pc.num_points == small_point_cloud_np.num_points
        np.testing.assert_array_almost_equal(pd_pc.points, small_point_cloud_np.points)

    def test_medium_pointcloud_transformation(
        self, medium_point_cloud_np, transformation_matrix
    ):
        """Test that transformation produces same results."""
        from tests.conftest import np_to_pd_pointcloud

        pd_pc = np_to_pd_pointcloud(medium_point_cloud_np)

        np_transformed = medium_point_cloud_np.transform(
            transformation_matrix, inplace=False
        )
        pd_transformed = pd_pc.transform(transformation_matrix, inplace=False)

        np.testing.assert_array_almost_equal(
            pd_transformed.points, np_transformed.points, decimal=5
        )

    def test_large_pointcloud_sampling(self, large_point_cloud_np):
        """Test that sampling works correctly on large point clouds."""
        from tests.conftest import np_to_pd_pointcloud

        pd_pc = np_to_pd_pointcloud(large_point_cloud_np)

        # Test sampling
        np.random.seed(123)
        pd_sampled = pd_pc.sample(num_samples=1000, replace=False)

        assert pd_sampled.num_points == 1000
        assert len(pd_sampled.attribute_names) > 0
