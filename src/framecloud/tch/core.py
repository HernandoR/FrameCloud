"""Torch-based PointCloud implementation."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from framecloud.exceptions import ArrayShapeError, AttributeExistsError


class PointCloud:
    """A simple point cloud representation using torch tensors."""

    def __init__(
        self,
        points: torch.Tensor,
        attributes: dict[str, torch.Tensor] | None = None,
    ):
        if points.ndim != 2 or points.shape[1] != 3:
            logger.error("Points tensor must be of shape Nx3.")
            raise ArrayShapeError("Points tensor must be of shape Nx3.")

        attributes = attributes or {}
        for attr_name, attr_value in attributes.items():
            if attr_value.shape[0] != points.shape[0]:
                logger.error(
                    f"Attribute '{attr_name}' length does not match number of points."
                )
                raise ArrayShapeError(
                    f"Attribute '{attr_name}' length does not match number of points."
                )

        self.points = points
        self.attributes = attributes

    @property
    def num_points(self) -> int:
        """Returns the number of points in the point cloud."""
        return int(self.points.shape[0])

    @property
    def attribute_names(self) -> list[str]:
        """Returns a list of attribute names in the point cloud."""
        return list(self.attributes.keys())

    @property
    def device(self) -> torch.device:
        """Return the device backing this point cloud."""
        return self.points.device

    def set_attribute(self, name: str, values: torch.Tensor):
        """Sets an attribute for the point cloud, overwriting if exists."""
        if values.shape[0] != self.num_points:
            logger.error(f"Attribute '{name}' length does not match number of points.")
            raise ArrayShapeError(
                f"Attribute '{name}' length does not match number of points."
            )
        self.attributes[name] = values

    def add_attribute(self, name: str, values: torch.Tensor):
        """Adds an attribute to the point cloud."""
        if name in self.attributes:
            logger.error(f"Attribute '{name}' already exists.")
            raise AttributeExistsError(f"Attribute '{name}' already exists.")
        self.set_attribute(name, values)

    def remove_attribute(self, name: str):
        """Removes an attribute if present."""
        self.attributes.pop(name, None)

    def get_attribute(self, name: str) -> torch.Tensor | None:
        """Retrieve an attribute if present."""
        return self.attributes.get(name)

    def clone(self) -> "PointCloud":
        """Deep copy the point cloud and its attributes."""
        return PointCloud(
            points=self.points.clone(),
            attributes={k: v.clone() for k, v in self.attributes.items()},
        )

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "PointCloud":
        """Move the point cloud to a device."""
        return PointCloud(
            points=self.points.to(device, non_blocking=non_blocking),
            attributes={
                k: v.to(device, non_blocking=non_blocking)
                for k, v in self.attributes.items()
            },
        )

    def cpu(self) -> "PointCloud":
        """Return a CPU copy of the point cloud."""
        return self.to("cpu")

    def cuda(self, non_blocking: bool = False) -> "PointCloud":
        """Return a CUDA copy of the point cloud."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system.")
        return self.to("cuda", non_blocking=non_blocking)

    def __getattr__(self, name: str) -> Any:
        if "attributes" in self.__dict__ and name in self.attributes:
            return self.attributes[name]
        raise AttributeError(f"{name} is not a PointCloud attribute")
