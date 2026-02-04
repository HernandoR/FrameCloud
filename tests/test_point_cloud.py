"""Tests for the PointCloud class."""

import numpy as np
import pytest
from pydantic import ValidationError

from framecloud.np.core import ArrayShapeError, AttributeExistsError, PointCloud


class TestPointCloudInitialization:
    """Test PointCloud initialization and validation."""

    def test_create_basic_point_cloud(self):
        """Test creating a basic point cloud with valid points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        pc = PointCloud(points=points)
        assert pc.points.shape == (3, 3)
        assert pc.num_points == 3
        assert len(pc.attribute_names) == 0

    def test_create_point_cloud_with_attributes(self):
        """Test creating a point cloud with attributes."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0]])
        pc = PointCloud(points=points, attributes={"colors": colors})
        assert pc.num_points == 2
        assert "colors" in pc.attribute_names
        np.testing.assert_array_equal(pc.attributes["colors"], colors)

    def test_invalid_points_shape_2d_but_wrong_columns(self):
        """Test that points array must have exactly 3 columns."""
        points = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2D points instead of 3D
        with pytest.raises(ValidationError, match="Points array must be of shape Nx3"):
            PointCloud(points=points)

    def test_invalid_points_shape_1d(self):
        """Test that points array must be 2D."""
        points = np.array([0.0, 1.0, 2.0])  # 1D array
        with pytest.raises(ValidationError, match="Points array must be of shape Nx3"):
            PointCloud(points=points)

    def test_invalid_attribute_length(self):
        """Test that attributes must have same length as points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0]])  # Only 1 color for 2 points
        with pytest.raises(ValidationError, match="does not match number of points"):
            PointCloud(points=points, attributes={"colors": colors})

    def test_empty_point_cloud(self):
        """Test creating an empty point cloud."""
        points = np.empty((0, 3))
        pc = PointCloud(points=points)
        assert pc.num_points == 0


class TestPointCloudProperties:
    """Test PointCloud properties."""

    def test_num_points_property(self):
        """Test num_points property."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        pc = PointCloud(points=points)
        assert pc.num_points == 3

    def test_attribute_names_property(self):
        """Test attribute_names property."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0]])
        intensities = np.array([1.0, 2.0])
        pc = PointCloud(
            points=points, attributes={"colors": colors, "intensities": intensities}
        )
        assert set(pc.attribute_names) == {"colors", "intensities"}

    def test_attribute_names_empty(self):
        """Test attribute_names property when no attributes."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)
        assert pc.attribute_names == []


class TestPointCloudAttributeOperations:
    """Test attribute management operations."""

    def test_add_attribute(self):
        """Test adding a new attribute."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)
        intensities = np.array([1.0, 2.0])
        pc.add_attribute("intensities", intensities)
        assert "intensities" in pc.attribute_names
        np.testing.assert_array_equal(pc.attributes["intensities"], intensities)

    def test_add_duplicate_attribute(self):
        """Test that adding duplicate attribute raises error."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0]])
        pc = PointCloud(points=points, attributes={"colors": colors})
        new_colors = np.array([[0, 0, 255], [255, 255, 0]])
        with pytest.raises(AttributeExistsError):
            pc.add_attribute("colors", new_colors)

    def test_set_attribute_overwrites(self):
        """Test that set_attribute overwrites existing attribute."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0]])
        pc = PointCloud(points=points, attributes={"colors": colors})
        new_colors = np.array([[0, 0, 255], [255, 255, 0]])
        pc.set_attribute("colors", new_colors)
        np.testing.assert_array_equal(pc.attributes["colors"], new_colors)

    def test_set_new_attribute(self):
        """Test that set_attribute can add new attributes."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)
        intensities = np.array([1.0, 2.0])
        pc.set_attribute("intensities", intensities)
        assert "intensities" in pc.attribute_names

    def test_set_attribute_wrong_length(self):
        """Test that set_attribute with wrong length raises error."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)
        wrong_intensities = np.array([1.0])  # Only 1 value for 2 points
        with pytest.raises(ArrayShapeError, match="does not match number of points"):
            pc.set_attribute("intensities", wrong_intensities)

    def test_remove_attribute(self):
        """Test removing an attribute."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0]])
        pc = PointCloud(points=points, attributes={"colors": colors})
        pc.remove_attribute("colors")
        assert "colors" not in pc.attribute_names

    def test_remove_nonexistent_attribute(self):
        """Test that removing nonexistent attribute doesn't raise error."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)
        pc.remove_attribute("nonexistent")  # Should not raise

    def test_get_attribute(self):
        """Test getting an attribute."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0]])
        pc = PointCloud(points=points, attributes={"colors": colors})
        retrieved_colors = pc.get_attribute("colors")
        np.testing.assert_array_equal(retrieved_colors, colors)

    def test_get_nonexistent_attribute(self):
        """Test getting a nonexistent attribute returns None."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)
        assert pc.get_attribute("nonexistent") is None


class TestPointCloudTransformation:
    """Test transformation operations."""

    def test_translation_transform(self):
        """Test translation transformation."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)

        # Translation matrix (move by 1.0 in each direction)
        matrix = np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ]
        )

        transformed = pc.transform(matrix, inplace=False)
        expected = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        np.testing.assert_array_almost_equal(transformed.points, expected)

    def test_scale_transform(self):
        """Test scale transformation."""
        points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        pc = PointCloud(points=points)

        # Scale matrix (scale by 2.0)
        matrix = np.array(
            [
                [2, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 1],
            ]
        )

        transformed = pc.transform(matrix, inplace=False)
        expected = np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
        np.testing.assert_array_almost_equal(transformed.points, expected)

    def test_transform_inplace(self):
        """Test in-place transformation."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)

        matrix = np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ]
        )

        result = pc.transform(matrix, inplace=True)
        expected = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        np.testing.assert_array_almost_equal(pc.points, expected)
        assert result is None  # inplace should return None

    def test_transform_with_attributes(self):
        """Test that attributes are preserved during transformation."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0]])
        pc = PointCloud(points=points, attributes={"colors": colors})

        matrix = np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ]
        )

        transformed = pc.transform(matrix, inplace=False)
        assert "colors" in transformed.attribute_names
        np.testing.assert_array_equal(transformed.attributes["colors"], colors)

    def test_invalid_transformation_matrix(self):
        """Test that invalid transformation matrix raises error."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)

        invalid_matrix = np.array([[1, 0], [0, 1]])  # 2x2 instead of 4x4
        with pytest.raises(ArrayShapeError, match="4x4"):
            pc.transform(invalid_matrix)


class TestPointCloudCopy:
    """Test copy operations."""

    def test_copy_creates_new_instance(self):
        """Test that copy creates a new instance."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pc = PointCloud(points=points)
        pc_copy = pc.model_copy(deep=True)
        assert pc is not pc_copy
        np.testing.assert_array_equal(pc.points, pc_copy.points)

    def test_copy_is_deep(self):
        """Test that copy is deep (modifying copy doesn't affect original)."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0]])
        pc = PointCloud(points=points, attributes={"colors": colors})
        pc_copy = pc.model_copy(deep=True)

        # Modify the copy
        pc_copy.points[0, 0] = 999.0
        pc_copy.attributes["colors"][0, 0] = 0

        # Original should be unchanged
        assert pc.points[0, 0] == 0.0
        assert pc.attributes["colors"][0, 0] == 255


class TestPointCloudSampling:
    """Test sampling operations."""

    def test_sample_without_replacement(self):
        """Test sampling without replacement."""
        points = np.arange(30).reshape(10, 3).astype(float)
        pc = PointCloud(points=points)
        sampled = pc.sample(num_samples=5, replace=False)
        assert sampled.num_points == 5

    def test_sample_with_replacement(self):
        """Test sampling with replacement."""
        points = np.arange(30).reshape(10, 3).astype(float)
        pc = PointCloud(points=points)
        sampled = pc.sample(num_samples=15, replace=True)
        assert sampled.num_points == 15

    def test_sample_more_than_available_without_replacement(self):
        """Test that sampling more than available without replacement raises error."""
        points = np.arange(30).reshape(10, 3).astype(float)
        pc = PointCloud(points=points)
        with pytest.raises(ValueError, match="exceeds number of points"):
            pc.sample(num_samples=15, replace=False)

    def test_sample_preserves_attributes(self):
        """Test that sampling preserves attributes."""
        points = np.arange(30).reshape(10, 3).astype(float)
        colors = np.array([[i, i, i] for i in range(10)])
        pc = PointCloud(points=points, attributes={"colors": colors})
        sampled = pc.sample(num_samples=5, replace=False)
        assert "colors" in sampled.attribute_names
        assert sampled.attributes["colors"].shape[0] == 5


class TestPointCloudToDict:
    """Test dict conversion."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0]])
        pc = PointCloud(points=points, attributes={"colors": colors})

        result = pc.to_dict()
        assert "points" in result
        assert "attributes" in result
        np.testing.assert_array_equal(result["points"], points)
        np.testing.assert_array_equal(result["attributes"]["colors"], colors)
