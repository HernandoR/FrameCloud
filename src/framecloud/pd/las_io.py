"""LAS/LAZ file I/O implementation for pandas-based PointCloud."""

from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from loguru import logger

from framecloud.pd.core import PointCloud


class LasIO:
    """Implementation of LAS/LAZ file I/O operations for pandas PointCloud."""

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

        data = {
            "X": np.array(las.x),
            "Y": np.array(las.y),
            "Z": np.array(las.z),
        }

        for dimension in las.point_format.dimensions:
            if dimension.name not in ["X", "Y", "Z"]:
                data[dimension.name] = np.array(las[dimension.name])

        df = pd.DataFrame(data)
        pc = PointCloud(data=df)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    @staticmethod
    def to_las(point_cloud: PointCloud, file_path: Path | str):
        """Save a PointCloud to a LAS file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output LAS file.
        """
        file_path = str(file_path)
        logger.info(f"Saving PointCloud to LAS file: {file_path}")
        header = laspy.LasHeader(point_format=7, version="1.4")
        las = laspy.LasData(header)

        las.x = point_cloud.data["X"].to_numpy()
        las.y = point_cloud.data["Y"].to_numpy()
        las.z = point_cloud.data["Z"].to_numpy()

        for attr_name in point_cloud.attribute_names:
            las[attr_name] = point_cloud.data[attr_name].to_numpy()

        las.write(file_path)
        logger.info(f"PointCloud saved to {file_path} successfully.")
