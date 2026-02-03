"""Common exceptions used across framecloud implementations."""


class AttributeExistsError(Exception):
    """Custom exception raised when an attribute already exists in the point cloud."""

    name: str


class ArrayShapeError(ValueError):
    """Custom exception raised when a numpy array has an unexpected shape."""

    info: str
