"""Parquet file I/O implementation for numpy-based PointCloud."""

from pathlib import Path

import polars as pl
from loguru import logger


class ParquetIO:
    """Mixin providing Parquet file I/O operations for numpy PointCloud."""

    @classmethod
    def from_parquet(cls, file_path: Path | str,
        position_cols: list[str] = None,
    ):
        """Load a PointCloud from a Parquet file.

        Args:
            file_path (Path): Path to the Parquet file.
            position_cols (list[str]): List of column names for point positions. Defaults to ["X", "Y", "Z"].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        if position_cols is None:
            position_cols = ["X", "Y", "Z"]
        logger.info(f"Loading PointCloud from Parquet file: {file_path}")
        df = pl.read_parquet(file_path)
        points = df.select(position_cols).to_numpy()

        attributes = {}
        for col in df.columns:
            if col not in position_cols:
                attributes[col] = df[col].to_numpy()

        pc = cls(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_parquet(self, file_path: Path | str, position_cols: list[str] = None):
        """Save this PointCloud to a Parquet file.

        Args:
            file_path (Path): Path to the output Parquet file.
            position_cols (list[str]): List of column names for point positions. Defaults to ["X", "Y", "Z"].
        """
        if position_cols is None:
            position_cols = ["X", "Y", "Z"]
        logger.info(f"Saving PointCloud to Parquet file: {file_path}")
        data = {}
        data[position_cols[0]] = self.points[:, 0]
        data[position_cols[1]] = self.points[:, 1]
        data[position_cols[2]] = self.points[:, 2]

        for attr_name, values in self.attributes.items():
            data[attr_name] = values

        df = pl.DataFrame(data)
        df.write_parquet(file_path)
        logger.info(f"PointCloud saved to {file_path} successfully.")
