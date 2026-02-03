"""Common utility functions for I/O operations."""

import numpy as np
from loguru import logger


def validate_xyz_in_attribute_names(attribute_names: list[str]) -> dict[str, int]:
    """Validate that X, Y, Z are in attribute names and return their positions.

    Args:
        attribute_names: List of attribute names

    Returns:
        dict: Mapping of X, Y, Z to their positions

    Raises:
        ValueError: If X, Y, Z are not all present
    """
    point_attrs_pos = {}
    for i, name in enumerate(attribute_names):
        if name in ["X", "Y", "Z"]:
            point_attrs_pos[name] = i

    if len(point_attrs_pos) < 3:
        logger.error(
            f"Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}, "
            f"found positions: {point_attrs_pos}"
        )
        raise ValueError(
            f"Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}, "
            f"found positions: {point_attrs_pos}"
        )

    return point_attrs_pos


def validate_buffer_size(array_size: int, num_attributes: int):
    """Validate that buffer size is compatible with number of attributes.

    Args:
        array_size: Size of the array from buffer
        num_attributes: Expected number of attributes

    Raises:
        ValueError: If sizes are incompatible
    """
    if array_size % num_attributes != 0:
        logger.error(
            "Binary buffer size is not compatible with the number of attributes."
        )
        raise ValueError(
            "Binary buffer size is not compatible with the number of attributes."
        )


def default_attribute_names(attribute_names: list[str] | None) -> list[str]:
    """Return default attribute names if None provided.

    Args:
        attribute_names: Optional list of attribute names

    Returns:
        list[str]: The provided list or default ["X", "Y", "Z"]
    """
    return attribute_names if attribute_names is not None else ["X", "Y", "Z"]


def extract_xyz_arrays(
    array: np.ndarray, point_attrs_pos: dict[str, int]
) -> np.ndarray:
    """Extract X, Y, Z columns from array and stack into points array.

    Args:
        array: Input array with all attributes
        point_attrs_pos: Mapping of X, Y, Z to their column positions

    Returns:
        np.ndarray: Nx3 array of points
    """
    return np.vstack(
        (
            array[:, point_attrs_pos["X"]],
            array[:, point_attrs_pos["Y"]],
            array[:, point_attrs_pos["Z"]],
        )
    ).T


def extract_attributes_dict(
    array: np.ndarray, attribute_names: list[str]
) -> dict[str, np.ndarray]:
    """Extract non-XYZ attributes from array into a dictionary.

    Args:
        array: Input array with all attributes
        attribute_names: List of all attribute names

    Returns:
        dict: Mapping of attribute name to values (excluding X, Y, Z)
    """
    attributes = {}
    for i, name in enumerate(attribute_names):
        if name not in ["X", "Y", "Z"]:
            attributes[name] = array[:, i]
    return attributes
