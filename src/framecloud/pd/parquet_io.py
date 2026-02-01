"""Parquet file I/O implementation for pandas-based PointCloud."""

from pathlib import Path

import polars as pl
from loguru import logger

from framecloud.pd.core import PointCloud


class ParquetIO:
    """Implementation of Parquet file I/O operations for pandas PointCloud."""

    @staticmethod
    def from_parquet(
        file_path: Path | str,
        position_cols: list[str] = None,
    ) -> PointCloud:
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
        df_pl = pl.read_parquet(file_path)
        df = df_pl.to_pandas()

        # Rename position columns to X, Y, Z if needed
        if position_cols != ["X", "Y", "Z"]:
            df = df.rename(
                columns={
                    position_cols[0]: "X",
                    position_cols[1]: "Y",
                    position_cols[2]: "Z",
                }
            )

        pc = PointCloud(data=df)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    @staticmethod
    def to_parquet(
        point_cloud: PointCloud, file_path: Path | str, position_cols: list[str] = None
    ):
        """Save a PointCloud to a Parquet file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output Parquet file.
            position_cols (list[str]): List of column names for point positions. Defaults to ["X", "Y", "Z"].
        """
        if position_cols is None:
            position_cols = ["X", "Y", "Z"]
        logger.info(f"Saving PointCloud to Parquet file: {file_path}")

        df = point_cloud.data.copy()
        # Rename X, Y, Z to custom position columns if needed
        if position_cols != ["X", "Y", "Z"]:
            df = df.rename(
                columns={
                    "X": position_cols[0],
                    "Y": position_cols[1],
                    "Z": position_cols[2],
                }
            )

        df_pl = pl.from_pandas(df)
        df_pl.write_parquet(file_path)
        logger.info(f"PointCloud saved to {file_path} successfully.")
