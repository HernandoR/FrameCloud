"""LAS/LAZ file I/O implementation for numpy-based PointCloud."""

from pathlib import Path

import laspy
import numpy as np
from loguru import logger


class LasIO:
    """Mixin providing LAS/LAZ file I/O operations for numpy PointCloud."""

    @classmethod
    def from_las(cls, file_path: Path | str):
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

        pc = cls(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_las(self, file_path: Path | str):
        """Save this PointCloud to a LAS file.

        Args:
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

        las.x = self.points[:, 0]
        las.y = self.points[:, 1]
        las.z = self.points[:, 2]

        for attr_name, values in self.attributes.items():
            las[attr_name] = values

        las.write(file_path)
        logger.info(f"PointCloud saved to {file_path} successfully.")
