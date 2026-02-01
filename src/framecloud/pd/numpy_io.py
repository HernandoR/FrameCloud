"""NumPy file format I/O implementation for pandas-based PointCloud."""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger



class NumpyIO:
    """Mixin providing NumPy file format (.npy, .npz) I/O operations for pandas PointCloud."""

    @classmethod
    def from_numpy_file(cls, 
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
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

        if not all(col in attribute_names for col in ["X", "Y", "Z"]):
            logger.error(f"Attribute names must include 'X', 'Y', and 'Z'.")
            raise ValueError(f"Attribute names must include 'X', 'Y', and 'Z'.")

        logger.info(f"Loading PointCloud from NumPy file: {file_path}")
        data = {name: array[:, i] for i, name in enumerate(attribute_names)}
        df = pd.DataFrame(data)
        pc = cls(data=df)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_numpy_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a NumPy .npy file.

        Args:
            file_path (Path): Path to the output NumPy .npy file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        logger.info(f"Saving PointCloud to NumPy file: {file_path}")
        arrays = []
        for name in attribute_names:
            if name in self.data.columns:
                arrays.append(self.data[name].to_numpy())
            else:
                logger.error(f"Attribute '{name}' not found in point cloud.")
                raise ValueError(f"Attribute '{name}' not found in point cloud.")

        combined_array = np.vstack(arrays).T.astype(dtype)
        np.save(file_path, combined_array)
        logger.info(f"PointCloud saved to {file_path} successfully.")

    @classmethod
    def from_npz_file(cls, 
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
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

        if not all(col in attribute_names for col in ["X", "Y", "Z"]):
            logger.error(f"Attribute names must include 'X', 'Y', and 'Z'.")
            raise ValueError(f"Attribute names must include 'X', 'Y', and 'Z'.")

        logger.info(f"Loading PointCloud from NumPy .npz file: {file_path}")
        for name in attribute_names:
            if name not in npz_data:
                logger.error(f"Attribute '{name}' not found in .npz file.")
                raise ValueError(f"Attribute '{name}' not found in .npz file.")

        data = {name: npz_data[name].astype(dtype) for name in attribute_names}
        df = pd.DataFrame(data)
        pc = cls(data=df)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_npz_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a NumPy .npz file.

        Args:
            file_path (Path): Path to the output NumPy .npz file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        logger.info(f"Saving PointCloud to NumPy .npz file: {file_path}")
        arrays = {}
        for name in attribute_names:
            if name in self.data.columns:
                arrays[name] = self.data[name].to_numpy().astype(dtype)
            else:
                logger.error(f"Attribute '{name}' not found in point cloud.")
                raise ValueError(f"Attribute '{name}' not found in point cloud.")

        np.savez(file_path, **arrays)
        logger.info(f"PointCloud saved to {file_path} successfully.")
