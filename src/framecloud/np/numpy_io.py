"""NumPy file format I/O implementation for numpy-based PointCloud."""

from pathlib import Path

import numpy as np
from loguru import logger

from framecloud.np.core import PointCloud


class NumpyIO:
    """Implementation of NumPy file format (.npy, .npz) I/O operations for numpy PointCloud."""

    @staticmethod
    def from_numpy_file(
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> PointCloud:
        """Load a PointCloud from a NumPy .npy file.

        Args:
            file_path (Path): Path to the NumPy .npy file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        array = np.load(file_path).astype(dtype)
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        # [X, Y, Z, ...] must in the attribute_names
        point_attrs_pos = {}
        for i, name in enumerate(attribute_names):
            if name in ["X", "Y", "Z"]:
                point_attrs_pos[name] = i
        if len(point_attrs_pos) < 3:
            logger.error(
                f"""Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}
                found positions: {point_attrs_pos}"""
            )
            raise ValueError(
                f"""Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}
                found positions: {point_attrs_pos}"""
            )

        logger.info(f"Loading PointCloud from NumPy file: {file_path}")
        points = np.vstack(
            (
                array[:, point_attrs_pos["X"]],
                array[:, point_attrs_pos["Y"]],
                array[:, point_attrs_pos["Z"]],
            )
        ).T
        attributes = {}
        for i, name in enumerate(attribute_names):
            if name not in ["X", "Y", "Z"]:
                attributes[name] = array[:, i]
        pc = PointCloud(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    @staticmethod
    def to_numpy_file(
        point_cloud: PointCloud,
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a NumPy .npy file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output NumPy .npy file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        logger.info(f"Saving PointCloud to NumPy file: {file_path}")
        arrays = []
        for name in attribute_names:
            if name == "X":
                arrays.append(point_cloud.points[:, 0])
            elif name == "Y":
                arrays.append(point_cloud.points[:, 1])
            elif name == "Z":
                arrays.append(point_cloud.points[:, 2])
            else:
                arrays.append(point_cloud.attributes[name])
        combined_array = np.vstack(arrays).T.astype(dtype)
        np.save(file_path, combined_array)
        logger.info(f"PointCloud saved to {file_path} successfully.")

    @staticmethod
    def from_npz_file(
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> PointCloud:
        """Load a PointCloud from a NumPy .npz file.

        Args:
            file_path (Path): Path to the NumPy .npz file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        npz_data = np.load(file_path)
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]
        point_attrs_pos = {}
        for i, name in enumerate(attribute_names):
            if name in ["X", "Y", "Z"]:
                point_attrs_pos[name] = i
        if len(point_attrs_pos) < 3:
            logger.error(
                f"""Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}
                found positions: {point_attrs_pos}"""
            )
            raise ValueError(
                f"""Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}
                found positions: {point_attrs_pos}"""
            )
        logger.info(f"Loading PointCloud from NumPy .npz file: {file_path}")
        for name in attribute_names:
            if name not in npz_data:
                logger.error(f"Attribute '{name}' not found in .npz file.")
                raise ValueError(f"Attribute '{name}' not found in .npz file.")
        array = np.vstack([npz_data[name] for name in attribute_names]).T.astype(dtype)
        points = np.vstack(
            (
                array[:, point_attrs_pos["X"]],
                array[:, point_attrs_pos["Y"]],
                array[:, point_attrs_pos["Z"]],
            )
        ).T
        attributes = {}
        for i, name in enumerate(attribute_names):
            if name not in ["X", "Y", "Z"]:
                attributes[name] = array[:, i]
        pc = PointCloud(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    @staticmethod
    def to_npz_file(
        point_cloud: PointCloud,
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a NumPy .npz file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output NumPy .npz file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        logger.info(f"Saving PointCloud to NumPy .npz file: {file_path}")
        arrays = {}
        for name in attribute_names:
            if name == "X":
                arrays[name] = point_cloud.points[:, 0].astype(dtype)
            elif name == "Y":
                arrays[name] = point_cloud.points[:, 1].astype(dtype)
            elif name == "Z":
                arrays[name] = point_cloud.points[:, 2].astype(dtype)
            else:
                arrays[name] = point_cloud.attributes[name].astype(dtype)
        np.savez(file_path, **arrays)
        logger.info(f"PointCloud saved to {file_path} successfully.")
