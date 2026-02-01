import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_validator


class AttributeExistsError(Exception):
    """Custom exception raised when an attribute already exists in the point cloud."""

    name: str


class ArrayShapeError(ValueError):
    """Custom exception raised when a numpy array has an unexpected shape."""

    info: str


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

    def copy(self):
        """Creates a deep copy of the PointCloud.

        Returns:
            PointCloud: A new instance of PointCloud with copied data.
        """
        new_pc = PointCloud(data=self.data.copy())
        logger.debug("Point cloud copied successfully.")
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
