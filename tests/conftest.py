"""Fixtures for point cloud testing."""

import numpy as np
import pytest

from framecloud.np.core import PointCloud as NpPointCloud
from framecloud.pd.core import PointCloud as PdPointCloud


def human_readable_number(num_str):
    """Convert a number to a human-readable format with unit suffix.

    Args:
        num_str: A string or number to convert.

    Returns:
        A human-readable string (e.g., '10M', '1.5K').
    """
    try:
        num = float(num_str)
    except (ValueError, TypeError):
        return num_str

    if num == 0:
        return "0"

    abs_num = abs(num)
    units = [(1e9, "B"), (1e6, "M"), (1e3, "K")]

    for threshold, unit in units:
        if abs_num >= threshold:
            value = num / threshold
            # Format with 1 decimal place, then remove trailing ".0"
            formatted = f"{value:.1f}"
            if formatted.endswith(".0"):
                formatted = formatted[:-2]
            return formatted + unit

    # For numbers < 1000, return as integer if whole, else as float
    return str(int(num)) if isinstance(num, int) or num.is_integer() else str(num)


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, (int, float)):
        return human_readable_number(val)
    elif isinstance(val, str) and val.replace(".", "").replace("-", "").isdigit():
        return human_readable_number(val)
    return str(val)


@pytest.fixture
def small_point_cloud_np():
    """Create a small point cloud (10 points) using numpy."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
        ]
    )
    colors = np.array([[i * 10, i * 20, i * 30] for i in range(10)])
    intensities = np.array([i * 100.0 for i in range(10)])
    return NpPointCloud(
        points=points, attributes={"colors": colors, "intensities": intensities}
    )


@pytest.fixture
def medium_point_cloud_np():
    """Create a medium point cloud (20k points) using numpy."""
    np.random.seed(42)
    num_points = 20000
    points = np.random.randn(num_points, 3).astype(np.float32) * 100
    colors = np.random.randint(0, 255, size=(num_points, 3), dtype=np.uint8)
    intensities = np.random.rand(num_points).astype(np.float32) * 1000
    return NpPointCloud(
        points=points, attributes={"colors": colors, "intensities": intensities}
    )


@pytest.fixture
def large_point_cloud_np():
    """Create a large point cloud (200k points) using numpy."""
    np.random.seed(42)
    num_points = 200000
    points = np.random.randn(num_points, 3).astype(np.float32) * 100
    colors = np.random.randint(0, 255, size=(num_points, 3), dtype=np.uint8)
    intensities = np.random.rand(num_points).astype(np.float32) * 1000
    classifications = np.random.randint(0, 20, size=num_points, dtype=np.uint8)
    return NpPointCloud(
        points=points,
        attributes={
            "colors": colors,
            "intensities": intensities,
            "classifications": classifications,
        },
    )


@pytest.fixture
def transformation_matrix():
    """Create a transformation matrix for testing."""
    # Translation by (10, 20, 30) and scale by 2
    return np.array(
        [
            [2, 0, 0, 10],
            [0, 2, 0, 20],
            [0, 0, 2, 30],
            [0, 0, 0, 1],
        ]
    )


@pytest.fixture(params=[20000, 50000, 100000, 200000])
def varied_size_point_cloud_np(request):
    """Create point clouds of various sizes for parametric testing."""
    np.random.seed(42)
    num_points = request.param
    points = np.random.randn(num_points, 3).astype(np.float32) * 100
    colors = np.random.randint(0, 255, size=(num_points, 3), dtype=np.uint8)
    intensities = np.random.rand(num_points).astype(np.float32) * 1000
    return NpPointCloud(
        points=points, attributes={"colors": colors, "intensities": intensities}
    )


def np_to_pd_pointcloud(np_pc: NpPointCloud) -> PdPointCloud:
    """Convert a numpy-based PointCloud to pandas-based PointCloud."""
    import pandas as pd

    data = {
        "X": np_pc.points[:, 0],
        "Y": np_pc.points[:, 1],
        "Z": np_pc.points[:, 2],
    }
    for attr_name, attr_values in np_pc.attributes.items():
        if attr_values.ndim == 1:
            data[attr_name] = attr_values
        else:
            # Handle multi-dimensional attributes (like colors)
            for i in range(attr_values.shape[1]):
                data[f"{attr_name}_{i}"] = attr_values[:, i]

    df = pd.DataFrame(data)
    return PdPointCloud(data=df)
