from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from framecloud._io_utils import default_attribute_names, validate_buffer_size
from framecloud.exceptions import ArrayShapeError, AttributeExistsError


class PointCloud(BaseModel):
    """A point cloud representation using pandas DataFrame.

    Attributes:
        data (pd.DataFrame): A DataFrame with columns for X, Y, Z coordinates and additional attributes.
    """

    model_config = {"arbitrary_types_allowed": True}

    data: pd.DataFrame = Field(
        ...,
        description="A DataFrame with X, Y, Z columns for 3D coordinates and optional attribute columns.",
    )

    @field_validator("data")
    def validate_data_columns(cls, v):
        if not isinstance(v, pd.DataFrame):
            logger.error("Data must be a pandas DataFrame.")
            raise TypeError("Data must be a pandas DataFrame.")
        if not all(col in v.columns for col in ["X", "Y", "Z"]):
            logger.error("DataFrame must have X, Y, Z columns.")
            raise ArrayShapeError("DataFrame must have X, Y, Z columns.")
        return v

    @property
    def points(self) -> np.ndarray:
        """Returns the points as an Nx3 numpy array."""
        return self.data[["X", "Y", "Z"]].to_numpy()

    @property
    def num_points(self) -> int:
        """Returns the number of points in the point cloud."""
        return len(self.data)

    @property
    def attribute_names(self) -> list[str]:
        """Returns a list of attribute names in the point cloud (excluding X, Y, Z)."""
        return [col for col in self.data.columns if col not in ["X", "Y", "Z"]]

    @property
    def attributes(self) -> dict:
        """Returns attributes as a dictionary of numpy arrays."""
        return {
            col: self.data[col].to_numpy()
            for col in self.data.columns
            if col not in ["X", "Y", "Z"]
        }

    def set_attribute(self, name: str, values: np.ndarray | pd.Series):
        """Sets an attribute for the point cloud. Overwrites if exists.

        Args:
            name (str): The name of the attribute.
            values (np.ndarray | pd.Series): An array of values for the attribute.

        Raises:
            ArrayShapeError: If the length of values does not match the number of points.
        """
        if len(values) != self.num_points:
            logger.error(f"Attribute '{name}' length does not match number of points.")
            raise ArrayShapeError(
                f"Attribute '{name}' length does not match number of points."
            )
        self.data[name] = values
        logger.debug(f"Attribute '{name}' set successfully.")

    def add_attribute(self, name: str, values: np.ndarray | pd.Series):
        """Adds an attribute to the point cloud.

        Args:
            name (str): The name of the attribute.
            values (np.ndarray | pd.Series): An array of values for the attribute.

        Raises:
            AttributeExistsError: If the attribute already exists.
            ArrayShapeError: If the length of values does not match the number of points.
        """
        if name in self.data.columns:
            logger.error(f"Attribute '{name}' already exists.")
            raise AttributeExistsError(f"Attribute '{name}' already exists.")
        if len(values) != self.num_points:
            logger.error(f"Attribute '{name}' length does not match number of points.")
            raise ArrayShapeError(
                f"Attribute '{name}' length does not match number of points."
            )
        self.data[name] = values
        logger.debug(f"Attribute '{name}' added successfully.")

    def remove_attribute(self, name: str):
        """Removes an attribute from the point cloud. Does nothing if not exists.

        Args:
            name (str): The name of the attribute.
        """
        if name in self.data.columns and name not in ["X", "Y", "Z"]:
            self.data.drop(columns=[name], inplace=True)
            logger.debug(f"Attribute '{name}' removed successfully.")
        else:
            logger.warning(f"Attribute '{name}' does not exist. No action taken.")

    def get_attribute(self, name: str) -> np.ndarray | None:
        """Retrieves an attribute from the point cloud.

        Args:
            name (str): The name of the attribute.

        Returns:
            np.ndarray: The array of values for the attribute.
            None: If the attribute does not exist.
        """
        if name in self.data.columns and name not in ["X", "Y", "Z"]:
            return self.data[name].to_numpy()
        return None

    def to_dict(self) -> dict:
        """Converts the PointCloud to a dictionary representation.

        Returns:
            dict: A dictionary containing points and attributes.
        """
        return {
            "points": self.points,
            "attributes": self.attributes,
        }

    def transform(self, matrix: np.ndarray, inplace: bool = False):
        """Applies a transformation matrix to the point cloud.

        Args:
            matrix (np.ndarray): A 4x4 transformation matrix.
            inplace (bool): If True, modifies the point cloud in place. Otherwise, returns a new PointCloud.

        Returns:
            PointCloud: The transformed point cloud (if inplace is False).
        """
        if matrix.shape != (4, 4):
            logger.error("Transformation matrix must be of shape 4x4.")
            raise ArrayShapeError("Transformation matrix must be of shape 4x4.")

        # Convert points to homogeneous coordinates
        points = self.points
        ones = np.ones((self.num_points, 1))
        homogeneous_points = np.hstack((points, ones))

        # Apply transformation
        transformed_points = homogeneous_points @ matrix.T

        if inplace:
            self.data[["X", "Y", "Z"]] = transformed_points[:, :3]
            logger.debug("Point cloud transformed in place.")
        else:
            new_data = self.data.copy()
            new_data[["X", "Y", "Z"]] = transformed_points[:, :3]
            new_pc = PointCloud(data=new_data)
            logger.debug("New transformed point cloud created.")
            return new_pc

    def sample(self, num_samples: int, replace: bool = False) -> "PointCloud":
        """Randomly samples points from the point cloud.

        Args:
            num_samples (int): The number of points to sample.
            replace (bool): Whether to sample with replacement.

        Returns:
            PointCloud: A new PointCloud instance containing the sampled points and their attributes.
        """
        if num_samples > self.num_points and not replace:
            logger.error(
                "Number of samples exceeds number of points without replacement."
            )
            raise ValueError(
                "Number of samples exceeds number of points without replacement."
            )

        sampled_data = self.data.sample(n=num_samples, replace=replace)
        logger.debug(f"Sampled {num_samples} points from the point cloud.")
        return PointCloud(data=sampled_data.reset_index(drop=True))

    # ========================================================================
    # LAS/LAZ File I/O Operations
    # ========================================================================

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

        data = {
            "X": np.array(las.x),
            "Y": np.array(las.y),
            "Z": np.array(las.z),
        }

        for dimension in las.point_format.dimensions:
            if dimension.name not in ["X", "Y", "Z"]:
                data[dimension.name] = np.array(las[dimension.name])

        df = pd.DataFrame(data)
        pc = cls(data=df)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_las(self, file_path: Path | str):
        """Save a PointCloud to a LAS file.

        Args:
            file_path (Path): Path to the output LAS file.
        """
        file_path = str(file_path)
        logger.info(f"Saving PointCloud to LAS file: {file_path}")
        header = laspy.LasHeader(point_format=7, version="1.4")
        las = laspy.LasData(header)

        las.x = self.data["X"].to_numpy()
        las.y = self.data["Y"].to_numpy()
        las.z = self.data["Z"].to_numpy()

        for attr_name in self.attribute_names:
            las[attr_name] = self.data[attr_name].to_numpy()

        las.write(file_path)
        logger.info(f"PointCloud saved to {file_path} successfully.")

    # ========================================================================
    # Parquet File I/O Operations
    # ========================================================================

    @classmethod
    def from_parquet(
        cls,
        file_path: Path | str,
        position_cols: list[str] | None = None,
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

    def to_parquet(self, file_path: Path | str, position_cols: list[str] | None = None):
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

    # ========================================================================
    # Binary Buffer/File I/O Operations
    # ========================================================================

    @classmethod
    def from_binary_buffer(
        cls,
        bytes_buffer: bytes,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ):
        """Load a PointCloud from a binary buffer.

        Args:
            bytes_buffer (bytes): Bytes buffer containing the binary data.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        attribute_names = default_attribute_names(attribute_names)

        # [X, Y, Z, ...] must be in the attribute_names
        if not all(col in attribute_names for col in ["X", "Y", "Z"]):
            logger.error(f"Attribute names must include 'X', 'Y', and 'Z'.")
            raise ValueError(f"Attribute names must include 'X', 'Y', and 'Z'.")

        logger.info("Loading PointCloud from binary buffer.")
        array = np.frombuffer(bytes_buffer, dtype=dtype)
        num_attributes = len(attribute_names)
        validate_buffer_size(array.size, num_attributes)

        array = array.reshape((-1, num_attributes))

        data = {name: array[:, i] for i, name in enumerate(attribute_names)}
        df = pd.DataFrame(data)
        pc = cls(data=df)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_binary_buffer(
        self,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> bytes:
        """Save a PointCloud to a binary buffer.

        Args:
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            bytes: Bytes buffer containing the binary data.
        """
        attribute_names = default_attribute_names(attribute_names)

        logger.info("Saving PointCloud to binary buffer.")
        arrays = []
        for name in attribute_names:
            if name in self.data.columns:
                arrays.append(self.data[name].to_numpy())
            else:
                logger.error(f"Attribute '{name}' not found in point cloud.")
                raise ValueError(f"Attribute '{name}' not found in point cloud.")

        combined_array = np.vstack(arrays).T.astype(dtype)
        bytes_buffer = combined_array.tobytes()
        logger.info("PointCloud saved to binary buffer successfully.")
        return bytes_buffer

    @classmethod
    def from_binary_file(
        cls,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ):
        """Load a PointCloud from a binary file.

        Args:
            file_path (Path): Path to the binary file ending with .bin.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        buffer = Path(file_path).read_bytes()
        return cls.from_binary_buffer(buffer, attribute_names, dtype)

    def to_binary_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a binary file.

        Args:
            file_path (Path): Path to the output binary file ending with .bin.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        bytes_buffer = self.to_binary_buffer(attribute_names, dtype)
        Path(file_path).write_bytes(bytes_buffer)
        logger.info(f"PointCloud saved to {file_path} successfully.")

    # ========================================================================
    # NumPy File Format I/O Operations
    # ========================================================================

    @classmethod
    def from_numpy_file(
        cls,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
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
        attribute_names = default_attribute_names(attribute_names)

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
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a NumPy .npy file.

        Args:
            file_path (Path): Path to the output NumPy .npy file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        attribute_names = default_attribute_names(attribute_names)

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
    def from_npz_file(
        cls,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
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
        attribute_names = default_attribute_names(attribute_names)

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
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a NumPy .npz file.

        Args:
            file_path (Path): Path to the output NumPy .npz file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        attribute_names = default_attribute_names(attribute_names)

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
