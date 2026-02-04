"""VoxelMap implementation for spatial downsampling with numpy arrays.

This module provides a VoxelMap class that voxelizes point clouds for spatial
downsampling while tracking the indices of points within each voxel.
"""

from typing import Any, Callable

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from framecloud.exceptions import ArrayShapeError


class VoxelMap(BaseModel):
    """A voxel map for spatial downsampling of point clouds.

    The VoxelMap aggregates points into voxels based on a specified voxel size.
    It tracks which points belong to each voxel and can aggregate attributes
    using various strategies (nearest to center by default).

    Attributes:
        voxel_size (float): The size of each voxel (uniform in all dimensions).
        voxel_coords (np.ndarray): Nx3 array of voxel coordinates for each unique voxel.
        voxel_indices (dict): Mapping from voxel coordinate tuple to list of point indices.
        representative_indices (np.ndarray): Index of the representative point for each voxel.
        origin (np.ndarray): The origin point of the voxel grid (3D coordinates).
        point_count (int): Original number of points used to create the voxel map.
        keep_copy (bool): Whether to keep a deep copy of the point cloud data.
        pointcloud_copy (dict | None): Optional deep copy of point cloud data.
    """

    model_config = {"arbitrary_types_allowed": True}

    voxel_size: float = Field(..., gt=0, description="Size of each voxel")
    voxel_coords: np.ndarray = Field(
        ..., description="Unique voxel coordinates (Nx3 array)"
    )
    voxel_indices: dict[tuple[int, int, int], np.ndarray] = Field(
        default_factory=dict, description="Mapping from voxel coords to point indices"
    )
    representative_indices: np.ndarray = Field(
        ..., description="Representative point index for each voxel"
    )
    origin: np.ndarray = Field(..., description="Origin of the voxel grid")
    point_count: int = Field(..., ge=0, description="Number of points in source cloud")
    keep_copy: bool = Field(
        default=False, description="Whether to keep a copy of point cloud data"
    )
    pointcloud_copy: dict[str, Any] | None = Field(
        default=None, description="Optional copy of point cloud data"
    )

    @field_validator("voxel_coords")
    def validate_voxel_coords(cls, v):
        if v.ndim != 2 or v.shape[1] != 3:
            logger.error("Voxel coordinates must be of shape Nx3.")
            raise ArrayShapeError("Voxel coordinates must be of shape Nx3.")
        return v

    @field_validator("origin")
    def validate_origin(cls, v):
        if v.shape != (3,):
            logger.error("Origin must be a 3D coordinate.")
            raise ArrayShapeError("Origin must be a 3D coordinate.")
        return v

    @classmethod
    def from_pointcloud(
        cls,
        pointcloud: "PointCloud",  # type: ignore # noqa: F821
        voxel_size: float,
        aggregation_method: str = "nearest_to_center",
        keep_copy: bool = False,
    ) -> "VoxelMap":
        """Create a VoxelMap from a PointCloud.

        Args:
            pointcloud: The input PointCloud object.
            voxel_size: Size of each voxel.
            aggregation_method: Method to select representative point.
                - "nearest_to_center": Select point nearest to voxel center (default)
                - "first": Select first point in each voxel
            keep_copy: Whether to keep a deep copy of the point cloud data.

        Returns:
            VoxelMap: The created voxel map.
        """
        logger.info(
            f"Creating VoxelMap with voxel_size={voxel_size}, "
            f"aggregation_method={aggregation_method}"
        )

        points = pointcloud.points
        num_points = points.shape[0]

        if num_points == 0:
            logger.warning("Empty point cloud provided.")
            return cls(
                voxel_size=voxel_size,
                voxel_coords=np.empty((0, 3), dtype=np.int32),
                voxel_indices={},
                representative_indices=np.empty(0, dtype=np.int32),
                origin=np.zeros(3),
                point_count=0,
                keep_copy=keep_copy,
                pointcloud_copy=None,
            )

        # Calculate origin (minimum coordinates)
        origin = points.min(axis=0)

        # Convert points to voxel coordinates
        voxel_coords_all = np.floor((points - origin) / voxel_size).astype(np.int32)

        # Find unique voxels and their indices
        unique_voxels, inverse_indices = np.unique(
            voxel_coords_all, axis=0, return_inverse=True
        )

        # Build mapping from voxel coordinates to point indices
        voxel_indices = {}
        for i in range(len(unique_voxels)):
            voxel_tuple = tuple(unique_voxels[i])
            point_mask = inverse_indices == i
            voxel_indices[voxel_tuple] = np.where(point_mask)[0]

        # Select representative points for each voxel
        representative_indices = cls._compute_representative_indices(
            points,
            unique_voxels,
            voxel_indices,
            origin,
            voxel_size,
            aggregation_method,
        )

        # Optionally keep a copy of the point cloud
        pointcloud_copy = None
        if keep_copy:
            pointcloud_copy = {
                "points": points.copy(),
                "attributes": {k: v.copy() for k, v in pointcloud.attributes.items()},
            }

        logger.info(
            f"Created VoxelMap with {len(unique_voxels)} voxels "
            f"from {num_points} points"
        )

        return cls(
            voxel_size=voxel_size,
            voxel_coords=unique_voxels,
            voxel_indices=voxel_indices,
            representative_indices=representative_indices,
            origin=origin,
            point_count=num_points,
            keep_copy=keep_copy,
            pointcloud_copy=pointcloud_copy,
        )

    @staticmethod
    def _compute_representative_indices(
        points: np.ndarray,
        unique_voxels: np.ndarray,
        voxel_indices: dict[tuple[int, int, int], np.ndarray],
        origin: np.ndarray,
        voxel_size: float,
        method: str,
    ) -> np.ndarray:
        """Compute representative point indices for each voxel.

        Args:
            points: Point coordinates (Nx3 array).
            unique_voxels: Unique voxel coordinates.
            voxel_indices: Mapping from voxel coords to point indices.
            origin: Origin of the voxel grid.
            voxel_size: Size of each voxel.
            method: Aggregation method ("nearest_to_center" or "first").

        Returns:
            Array of representative point indices.
        """
        representative_indices = np.zeros(len(unique_voxels), dtype=np.int32)

        for i, voxel_coord in enumerate(unique_voxels):
            voxel_tuple = tuple(voxel_coord)
            point_idx = voxel_indices[voxel_tuple]

            if method == "first":
                representative_indices[i] = point_idx[0]
            elif method == "nearest_to_center":
                # Calculate voxel center
                voxel_center = origin + (voxel_coord + 0.5) * voxel_size
                # Find point nearest to voxel center
                distances = np.linalg.norm(points[point_idx] - voxel_center, axis=1)
                nearest_idx = point_idx[np.argmin(distances)]
                representative_indices[i] = nearest_idx
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

        return representative_indices

    @property
    def num_voxels(self) -> int:
        """Returns the number of voxels."""
        return len(self.voxel_coords)

    def get_voxel_centers(self) -> np.ndarray:
        """Get the center coordinates of all voxels.

        Returns:
            Nx3 array of voxel center coordinates.
        """
        return self.origin + (self.voxel_coords + 0.5) * self.voxel_size

    def get_point_indices(self, voxel_coord: tuple[int, int, int]) -> np.ndarray:
        """Get point indices for a specific voxel.

        Args:
            voxel_coord: Voxel coordinate tuple (i, j, k).

        Returns:
            Array of point indices in the specified voxel.
        """
        return self.voxel_indices.get(voxel_coord, np.array([], dtype=np.int32))

    def downsample(
        self,
        pointcloud: "PointCloud",  # type: ignore # noqa: F821
        custom_aggregation: dict[str, Callable] | None = None,
    ) -> "PointCloud":  # type: ignore # noqa: F821
        """Create a downsampled point cloud using the voxel map.

        Args:
            pointcloud: The original point cloud (must match the one used to create the map).
            custom_aggregation: Optional dict mapping attribute names to aggregation functions.
                Each function should take an array of attribute values and return a single value.

        Returns:
            A new downsampled PointCloud.
        """
        from framecloud.np.core import PointCloud

        if pointcloud.num_points != self.point_count:
            logger.warning(
                "Point cloud size mismatch. This may indicate the point cloud "
                "has been modified since the voxel map was created."
            )

        # Get representative points
        downsampled_points = pointcloud.points[self.representative_indices]

        # Aggregate attributes
        downsampled_attributes = {}
        for attr_name, attr_values in pointcloud.attributes.items():
            if custom_aggregation and attr_name in custom_aggregation:
                # Use custom aggregation function
                aggregated_values = []
                for voxel_tuple in [tuple(vc) for vc in self.voxel_coords]:  # type: ignore
                    point_idx = self.voxel_indices[voxel_tuple]
                    aggregated_value = custom_aggregation[attr_name](
                        attr_values[point_idx]
                    )
                    aggregated_values.append(aggregated_value)
                downsampled_attributes[attr_name] = np.array(aggregated_values)
            else:
                # Use representative point's attribute
                downsampled_attributes[attr_name] = attr_values[
                    self.representative_indices
                ]

        logger.info(
            f"Downsampled point cloud from {self.point_count} to "
            f"{self.num_voxels} points"
        )

        return PointCloud(points=downsampled_points, attributes=downsampled_attributes)

    def recalculate(
        self,
        pointcloud: "PointCloud",  # type: ignore # noqa: F821
        aggregation_method: str = "nearest_to_center",
    ) -> "VoxelMap":
        """Recalculate the voxel map from a point cloud using saved parameters.

        Args:
            pointcloud: The point cloud to voxelize.
            aggregation_method: Method to select representative point.

        Returns:
            A new VoxelMap instance.
        """
        return VoxelMap.from_pointcloud(
            pointcloud=pointcloud,
            voxel_size=self.voxel_size,
            aggregation_method=aggregation_method,
            keep_copy=self.keep_copy,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the voxel map.

        Returns:
            Dictionary containing statistics.
        """
        points_per_voxel = [len(indices) for indices in self.voxel_indices.values()]

        return {
            "num_voxels": self.num_voxels,
            "num_points": self.point_count,
            "voxel_size": self.voxel_size,
            "compression_ratio": (
                self.point_count / self.num_voxels if self.num_voxels > 0 else 0
            ),
            "min_points_per_voxel": min(points_per_voxel) if points_per_voxel else 0,
            "max_points_per_voxel": max(points_per_voxel) if points_per_voxel else 0,
            "mean_points_per_voxel": (
                np.mean(points_per_voxel) if points_per_voxel else 0
            ),
            "origin": self.origin.tolist(),
        }
