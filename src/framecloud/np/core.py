import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from framecloud.np.binary_io import BinaryIO
from framecloud.np.las_io import LasIO
from framecloud.np.numpy_io import NumpyIO
from framecloud.np.parquet_io import ParquetIO


class AttributeExistsError(Exception):
    """Custom exception raised when an attribute already exists in the point cloud."""

    name: str


class ArrayShapeError(ValueError):
    """Custom exception raised when a numpy array has an unexpected shape."""

    info: str


class PointCloud(LasIO, ParquetIO, BinaryIO, NumpyIO, BaseModel):
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
            logger.debug("New transformed point cloud created.")
            return new_pc

    def copy(self):
        """Creates a deep copy of the PointCloud.

        Returns:
            PointCloud: A new instance of PointCloud with copied data.
        """
        new_pc = PointCloud(
            points=self.points.copy(),
            attributes={k: v.copy() for k, v in self.attributes.items()},
        )
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

        sampled_indices = np.random.choice(
            self.num_points, size=num_samples, replace=replace
        )
        sampled_points = self.points[sampled_indices]

        sampled_attributes = {
            name: values[sampled_indices] for name, values in self.attributes.items()
        }

        logger.debug(f"Sampled {num_samples} points from the point cloud.")
        return PointCloud(points=sampled_points, attributes=sampled_attributes)
