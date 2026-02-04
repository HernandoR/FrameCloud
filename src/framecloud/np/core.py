"""PointCloud core implementation with integrated I/O operations.

This module contains the PointCloud class for numpy-based point cloud data,
with all I/O operations (LAS, Parquet, Binary, NumPy) integrated directly.
"""

from pathlib import Path

import laspy
import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from framecloud._io_utils import (
    default_attribute_names,
    extract_attributes_dict,
    extract_xyz_arrays,
    validate_buffer_size,
    validate_xyz_in_attribute_names,
)
from framecloud.exceptions import ArrayShapeError, AttributeExistsError


class PointCloud(BaseModel):
    """A simple point cloud representation using numpy arrays.

    Attributes:
        points (np.ndarray): An Nx3 array representing the 3D coordinates of N points.
        attributes (dict): A dictionary containing additional attributes for the points.
    """

    model_config = {"arbitrary_types_allowed": True}

    points: np.ndarray = Field(
        ...,
        description="An Nx3 array representing the 3D coordinates of N points.",
        example=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )
    attributes: dict = Field(
        default_factory=dict,
        description="A dictionary containing additional attributes for the points.",
        example={"colors": np.array([[255, 0, 0], [0, 255, 0]])},
    )

    @field_validator("points")
    def validate_points_shape(cls, v):
        if v.ndim != 2 or v.shape[1] != 3:
            logger.error("Points array must be of shape Nx3.")
            raise ArrayShapeError("Points array must be of shape Nx3.")
        return v

    def __post_init__(self):
        for attr_name, attr_value in self.attributes.items():
            if attr_value.shape[0] != self.points.shape[0]:
                logger.error(
                    f"Attribute '{attr_name}' length does not match number of points."
                )
                raise ArrayShapeError(
                    f"Attribute '{attr_name}' length does not match number of points."
                )

    @model_validator(mode="after")
    def validate_attributes(self):
        """Validate that all attributes have the same length as points."""
        for attr_name, attr_value in self.attributes.items():
            if attr_value.shape[0] != self.points.shape[0]:
                logger.error(
                    f"Attribute '{attr_name}' length does not match number of points."
                )
                raise ArrayShapeError(
                    f"Attribute '{attr_name}' length does not match number of points."
                )
        return self

    def __getattribute__(self, name):
        # Use super().__getattribute__ to access 'attributes' to avoid recursion
        if name not in ["attributes", "points", "model_config"] and name[0] != "_":
            try:
                attributes = super().__getattribute__("attributes")
                if name in attributes:
                    return attributes[name]
            except AttributeError:
                pass
        return super().__getattribute__(name)

    @property
    def num_points(self) -> int:
        """Returns the number of points in the point cloud."""
        return self.points.shape[0]

    @property
    def attribute_names(self) -> list[str]:
        """Returns a list of attribute names in the point cloud."""
        return list(self.attributes.keys())

    def set_attribute(self, name: str, values: np.ndarray):
        """Sets an attribute for the point cloud.overwrite if exists.

        Args:
            name (str): The name of the attribute.
            values (np.ndarray): An array of values for the attribute.

        Raises:
            ArrayShapeError: If the length of values does not match the number of points.
        """
        if values.shape[0] != self.num_points:
            logger.error(f"Attribute '{name}' length does not match number of points.")
            raise ArrayShapeError(
                f"Attribute '{name}' length does not match number of points."
            )
        self.attributes[name] = values
        logger.debug(f"Attribute '{name}' set successfully.")

    def add_attribute(self, name: str, values: np.ndarray):
        """Adds an attribute to the point cloud.

        Args:
            name (str): The name of the attribute.
            values (np.ndarray): An array of values for the attribute.

        Raises:
            AttributeExistsError: If the attribute already exists.
            ArrayShapeError: If the length of values does not match the number of points.
        """
        if name in self.attributes:
            logger.error(f"Attribute '{name}' already exists.")
            raise AttributeExistsError(f"Attribute '{name}' already exists.")
        if values.shape[0] != self.num_points:
            logger.error(f"Attribute '{name}' length does not match number of points.")
            raise ArrayShapeError(
                f"Attribute '{name}' length does not match number of points."
            )
        self.attributes[name] = values
        logger.debug(f"Attribute '{name}' added successfully.")

    def remove_attribute(self, name: str):
        """Removes an attribute from the point cloud. does nothing if not exists.
        Args:
            name (str): The name of the attribute.
        """
        if name in self.attributes:
            del self.attributes[name]
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
        return self.attributes.get(name, None)

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
        ones = np.ones((self.num_points, 1))
        homogeneous_points = np.hstack((self.points, ones))

        # Apply transformation
        transformed_points = homogeneous_points @ matrix.T

        if inplace:
            self.points = transformed_points[:, :3]
            logger.debug("Point cloud transformed in place.")
        else:
            new_pc = PointCloud(
                points=transformed_points[:, :3], attributes=self.attributes.copy()
            )
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

        sampled_indices = np.random.choice(
            self.num_points, size=num_samples, replace=replace
        )
        sampled_points = self.points[sampled_indices]

        sampled_attributes = {
            name: values[sampled_indices] for name, values in self.attributes.items()
        }

        logger.debug(f"Sampled {num_samples} points from the point cloud.")
        return PointCloud(points=sampled_points, attributes=sampled_attributes)

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
        points = np.vstack((las.x, las.y, las.z)).T

        attributes = {}
        for dimension in las.point_format.dimensions:
            if dimension.name not in ["X", "Y", "Z"]:
                attributes[dimension.name] = np.array(las[dimension.name])

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
        df = pl.read_parquet(file_path)
        points = df.select(position_cols).to_numpy()

        attributes = {}
        for col in df.columns:
            if col not in position_cols:
                attributes[col] = df[col].to_numpy()

        pc = cls(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_parquet(self, file_path: Path | str, position_cols: list[str] | None = None):
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
        point_attrs_pos = validate_xyz_in_attribute_names(attribute_names)

        logger.info("Loading PointCloud from binary buffer.")
        array = np.frombuffer(bytes_buffer, dtype=dtype)
        num_attributes = len(attribute_names)
        validate_buffer_size(array.size, num_attributes)

        array = array.reshape((-1, num_attributes))
        points = extract_xyz_arrays(array, point_attrs_pos)
        attributes = extract_attributes_dict(array, attribute_names)

        pc = cls(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_binary_buffer(
        self,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ) -> bytes:
        """Save this PointCloud to a binary buffer.

        Args:
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            bytes: Bytes buffer containing the binary data.
        """
        attribute_names = default_attribute_names(attribute_names)

        logger.info("Saving PointCloud to binary buffer.")
        arrays = []
        for name in attribute_names:
            if name == "X":
                arrays.append(self.points[:, 0])
            elif name == "Y":
                arrays.append(self.points[:, 1])
            elif name == "Z":
                arrays.append(self.points[:, 2])
            else:
                arrays.append(self.attributes[name])
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
        """Save this PointCloud to a binary file.

        Args:
            file_path (Path): Path to the output binary file ending with .bin.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        bytes_buffer = self.to_binary_buffer(attribute_names, dtype)
        Path(file_path).write_bytes(bytes_buffer)
        logger.info(f"PointCloud saved to {file_path} successfully.")

    # ========================================================================
    # NumPy File Format I/O Operations (.npy and .npz)
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
        point_attrs_pos = validate_xyz_in_attribute_names(attribute_names)

        logger.info(f"Loading PointCloud from NumPy file: {file_path}")
        points = extract_xyz_arrays(array, point_attrs_pos)
        attributes = extract_attributes_dict(array, attribute_names)

        pc = cls(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_numpy_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ):
        """Save this PointCloud to a NumPy .npy file.

        Args:
            file_path (Path): Path to the output NumPy .npy file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        attribute_names = default_attribute_names(attribute_names)

        logger.info(f"Saving PointCloud to NumPy file: {file_path}")
        arrays = []
        for name in attribute_names:
            if name == "X":
                arrays.append(self.points[:, 0])
            elif name == "Y":
                arrays.append(self.points[:, 1])
            elif name == "Z":
                arrays.append(self.points[:, 2])
            else:
                arrays.append(self.attributes[name])
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
        point_attrs_pos = validate_xyz_in_attribute_names(attribute_names)

        logger.info(f"Loading PointCloud from NumPy .npz file: {file_path}")
        for name in attribute_names:
            if name not in npz_data:
                logger.error(f"Attribute '{name}' not found in .npz file.")
                raise ValueError(f"Attribute '{name}' not found in .npz file.")

        array = np.vstack([npz_data[name] for name in attribute_names]).T.astype(dtype)
        points = extract_xyz_arrays(array, point_attrs_pos)
        attributes = extract_attributes_dict(array, attribute_names)

        pc = cls(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_npz_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] | None = None,
        dtype=np.float32,
    ):
        """Save this PointCloud to a NumPy .npz file.

        Args:
            file_path (Path): Path to the output NumPy .npz file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        attribute_names = default_attribute_names(attribute_names)

        logger.info(f"Saving PointCloud to NumPy .npz file: {file_path}")
        arrays = {}
        for name in attribute_names:
            if name == "X":
                arrays[name] = self.points[:, 0].astype(dtype)
            elif name == "Y":
                arrays[name] = self.points[:, 1].astype(dtype)
            elif name == "Z":
                arrays[name] = self.points[:, 2].astype(dtype)
            else:
                arrays[name] = self.attributes[name].astype(dtype)
        np.savez(file_path, **arrays)
        logger.info(f"PointCloud saved to {file_path} successfully.")
