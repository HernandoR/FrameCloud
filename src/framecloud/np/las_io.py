"""LAS/LAZ file I/O implementation for numpy-based PointCloud."""

from pathlib import Path

import laspy
import numpy as np
from loguru import logger

from framecloud.np.core import PointCloud


class LasIO:
    """Implementation of LAS/LAZ file I/O operations for numpy PointCloud."""

    @staticmethod
    def from_las(file_path: Path | str) -> PointCloud:
        """Load a PointCloud from a LAS/LAZ file.

        Args:
            file_path (Path): Path to the LAS/LAZ file.
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        logger.info(f"Loading PointCloud from LAS/LAZ file: {file_path}")
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).T

        attributes = {}
        for dimension in las.point_format.dimensions:
            if dimension.name not in ["X", "Y", "Z"]:
                attributes[dimension.name] = las[dimension.name]

        pc = PointCloud(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    @staticmethod
    def to_las(point_cloud: PointCloud, file_path: Path | str):
        """Save a PointCloud to a LAS file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output LAS file.

        Note:
            Please refer to https://laspy.readthedocs.io/en/latest/intro.html#point-format-6
            and https://laspy.readthedocs.io/en/latest/intro.html#point-format-7
            for supported attributes and their names.
        """
        file_path = str(file_path)
        logger.info(f"Saving PointCloud to LAS file: {file_path}")
        header = laspy.LasHeader(point_format=7, version="1.4")
        las = laspy.LasData(header)

        las.x = point_cloud.points[:, 0]
        las.y = point_cloud.points[:, 1]
        las.z = point_cloud.points[:, 2]

        for attr_name, values in point_cloud.attributes.items():
            las[attr_name] = values

        las.write(file_path)
        logger.info(f"PointCloud saved to {file_path} successfully.")
