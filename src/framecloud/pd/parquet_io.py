"""Parquet file I/O implementation for pandas-based PointCloud."""

from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger



class ParquetIO:
    """Mixin providing Parquet file I/O operations for pandas PointCloud."""

    @classmethod
    def from_parquet(cls, 
        file_path: Path | str,
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

        pc = cls(data=df)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_parquet(
        self, file_path: Path | str, position_cols: list[str] = None
    ):
        """Save a PointCloud to a Parquet file.

        Args:
            file_path (Path): Path to the output Parquet file.
            position_cols (list[str]): List of column names for point positions. Defaults to ["X", "Y", "Z"].
        """
        if position_cols is None:
            position_cols = ["X", "Y", "Z"]
        logger.info(f"Saving PointCloud to Parquet file: {file_path}")

        df = self.data.copy()
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
