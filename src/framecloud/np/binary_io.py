"""Binary file I/O implementation for numpy-based PointCloud."""

from pathlib import Path

import numpy as np
from loguru import logger

from framecloud.np.core import PointCloud


class BinaryIO:
    """Implementation of binary buffer/file I/O operations for numpy PointCloud."""

    @staticmethod
    def from_binary_buffer(
        bytes_buffer: bytes,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> PointCloud:
        """Load a PointCloud from a NumPy binary file.

        Args:
            bytes_buffer (bytes): Bytes buffer containing the NumPy binary data.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
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

        logger.info("Loading PointCloud from binary buffer.")
        array = np.frombuffer(bytes_buffer, dtype=dtype)
        num_attributes = len(attribute_names)
        if array.size % num_attributes != 0:
            logger.error(
                "Binary buffer size is not compatible with the number of attributes."
            )
            raise ValueError(
                "Binary buffer size is not compatible with the number of attributes."
            )
        array = array.reshape((-1, num_attributes))
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
    def to_binary_buffer(
        point_cloud: PointCloud,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> bytes:
        """Save a PointCloud to a NumPy binary buffer.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            bytes: Bytes buffer containing the NumPy binary data.
        """
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        logger.info("Saving PointCloud to binary buffer.")
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
        bytes_buffer = combined_array.tobytes()
        logger.info("PointCloud saved to binary buffer successfully.")
        return bytes_buffer

    @staticmethod
    def from_binary_file(
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> PointCloud:
        """Load a PointCloud from a NumPy binary file.

        Args:
            file_path (Path): Path to the NumPy binary file end with .bin.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        buffer = Path(file_path).read_bytes()
        return BinaryIO.from_binary_buffer(buffer, attribute_names, dtype)

    @staticmethod
    def to_binary_file(
        point_cloud: PointCloud,
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a NumPy binary file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output NumPy binary file end with .bin.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        bytes_buffer = BinaryIO.to_binary_buffer(point_cloud, attribute_names, dtype)
        Path(file_path).write_bytes(bytes_buffer)
        logger.info(f"PointCloud saved to {file_path} successfully.")
