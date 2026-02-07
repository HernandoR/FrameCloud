"""VoxelMap implementation for spatial downsampling with numpy arrays.

This module provides a VoxelMap class that voxelizes point clouds for spatial
downsampling while tracking the indices of points within each voxel.
"""

from typing import Any, Callable

import numpy as np
from loguru import logger

from framecloud.exceptions import ArrayShapeError
from framecloud.np.core import PointCloud


class VoxelMap:
    """A voxel map for spatial downsampling of point clouds.

    The VoxelMap aggregates points into voxels based on a specified voxel size.
    It tracks which points belong to each voxel and can aggregate attributes
    using various strategies during export.

    Attributes:
        voxel_size (float): The size of each voxel (uniform in all dimensions).
        voxel_coords (np.ndarray): Nx3 array of voxel coordinates for each unique voxel.
        inverse_indices (np.ndarray): Array mapping each point to its voxel index.
        origin (np.ndarray): The origin point of the voxel grid (3D coordinates).
        pointcloud: Reference to the source PointCloud (mutable reference or deep copy).
        is_copy (bool): Whether the pointcloud is a deep copy (immutable from outside).
    """

    def __init__(
        self,
        voxel_size: float,
        voxel_coords: np.ndarray,
        inverse_indices: np.ndarray,
        origin: np.ndarray,
        pointcloud: PointCloud,
        is_copy: bool = False,
    ):
        """Initialize a VoxelMap.

        Args:
            voxel_size: Size of each voxel (must be > 0).
            voxel_coords: Nx3 array of unique voxel coordinates.
            inverse_indices: Array mapping each point to its voxel index.
            origin: Origin of the voxel grid (3D coordinates).
            pointcloud: Reference to the PointCloud (either mutable ref or deep copy).
            is_copy: Whether the pointcloud is a deep copy.
        """
        if voxel_size <= 0:
            raise ValueError("voxel_size must be greater than 0")
        if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 3:
            logger.error("Voxel coordinates must be of shape Nx3.")
            raise ArrayShapeError("Voxel coordinates must be of shape Nx3.")
        if origin.shape != (3,):
            logger.error("Origin must be a 3D coordinate.")
            raise ArrayShapeError("Origin must be a 3D coordinate.")

        self.voxel_size = voxel_size
        self.voxel_coords = voxel_coords
        self.inverse_indices = inverse_indices
        self.origin = origin
        self.pointcloud = pointcloud
        self._is_copy = is_copy

    @property
    def is_copy(self) -> bool:
        """Returns whether the pointcloud is a deep copy (read-only)."""
        return self._is_copy

    @classmethod
    def from_pointcloud(
        cls,
        pointcloud: PointCloud,
        voxel_size: float,
        keep_copy: bool = False,
    ) -> "VoxelMap":
        """Create a VoxelMap from a PointCloud.

        Args:
            pointcloud: The input PointCloud object.
            voxel_size: Size of each voxel.
            keep_copy: Whether to keep a deep copy of the point cloud data.

        Returns:
            VoxelMap: The created voxel map.
        """
        logger.debug(f"Creating VoxelMap with voxel_size={voxel_size}")

        points = pointcloud.points
        num_points = points.shape[0]

        if num_points == 0:
            logger.warning("Empty point cloud provided.")
            empty_pc = PointCloud(points=np.empty((0, 3)), attributes={})
            return cls(
                voxel_size=voxel_size,
                voxel_coords=np.empty((0, 3), dtype=np.int32),
                inverse_indices=np.empty(0, dtype=np.intp),
                origin=np.zeros(3),
                pointcloud=empty_pc,
                is_copy=True,
            )

        # Calculate origin (minimum coordinates)
        origin = points.min(axis=0)

        # Convert points to voxel coordinates
        voxel_coords_all = np.floor((points - origin) / voxel_size).astype(np.int32)

        # Find unique voxels and their indices (vectorized)
        unique_voxels, inverse_indices = np.unique(
            voxel_coords_all, axis=0, return_inverse=True
        )
        logger.debug(f"Grouped {num_points} points into {len(unique_voxels)} voxels")

        # Handle point cloud reference
        if keep_copy:
            # Create a deep copy
            copied_points = points.copy()
            copied_attributes = {k: v.copy() for k, v in pointcloud.attributes.items()}
            pc_ref = PointCloud(points=copied_points, attributes=copied_attributes)
            is_copy = True
        else:
            # Keep mutable reference
            pc_ref = pointcloud
            is_copy = False

        logger.debug(
            f"Created VoxelMap with {len(unique_voxels)} voxels "
            f"from {num_points} points"
        )

        return cls(
            voxel_size=voxel_size,
            voxel_coords=unique_voxels,
            inverse_indices=inverse_indices,
            origin=origin,
            pointcloud=pc_ref,
            is_copy=is_copy,
        )

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
        # Find the voxel index matching the coordinate
        voxel_arr = np.array(voxel_coord, dtype=np.int32)
        matches = np.all(self.voxel_coords == voxel_arr, axis=1)
        voxel_idx = np.where(matches)[0]
        if len(voxel_idx) == 0:
            return np.array([], dtype=np.intp)
        return np.where(self.inverse_indices == voxel_idx[0])[0]

    def export_pointcloud(
        self,
        aggregation_method: str = "nearest_to_center",
        custom_aggregation: dict[str, Callable] | None = None,
    ) -> PointCloud:
        """Export a downsampled point cloud using the voxel map.

        Args:
            aggregation_method: Method to select representative point.
                - "nearest_to_center": Select point nearest to voxel center (default)
                - "first": Select first point in each voxel
            custom_aggregation: Optional dict mapping attribute names to aggregation functions.
                Each function should take an array of attribute values and return a single value.

        Returns:
            A new downsampled PointCloud.
        """
        # Handle empty voxel map
        if self.num_voxels == 0:
            logger.warning("Empty voxel map, returning empty point cloud")
            return PointCloud(points=np.empty((0, 3)), attributes={})

        points = self.pointcloud.points

        # Determine representative indices for all voxels (vectorized where possible)
        representative_indices = np.zeros(self.num_voxels, dtype=np.int32)

        if aggregation_method == "first":
            # Get first index from each voxel using argsort (vectorized)
            sorted_order = np.argsort(self.inverse_indices, kind="stable")
            _, first_occurrence = np.unique(
                self.inverse_indices[sorted_order], return_index=True
            )
            representative_indices = sorted_order[first_occurrence]
        elif aggregation_method == "nearest_to_center":
            # Calculate all voxel centers at once
            voxel_centers = self.origin + (self.voxel_coords + 0.5) * self.voxel_size

            # Compute squared distances from each point to its voxel center
            point_voxel_centers = voxel_centers[self.inverse_indices]
            squared_distances = np.sum((points - point_voxel_centers) ** 2, axis=1)

            # For each voxel, find the point with minimum distance
            # Use a vectorized approach with bincount-like logic
            for i in range(self.num_voxels):
                point_mask = self.inverse_indices == i
                point_idx = np.where(point_mask)[0]
                representative_indices[i] = point_idx[
                    np.argmin(squared_distances[point_idx])
                ]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Get representative points (vectorized)
        downsampled_points = points[representative_indices]

        # Aggregate attributes (vectorized where possible)
        downsampled_attributes = {}
        for attr_name, attr_values in self.pointcloud.attributes.items():
            if custom_aggregation and attr_name in custom_aggregation:
                # Use custom aggregation function (requires iteration per voxel)
                aggregated_values = np.zeros(self.num_voxels, dtype=attr_values.dtype)
                for i in range(self.num_voxels):
                    point_idx = np.where(self.inverse_indices == i)[0]
                    aggregated_values[i] = custom_aggregation[attr_name](
                        attr_values[point_idx]
                    )
                downsampled_attributes[attr_name] = aggregated_values
            else:
                # Use representative point's attribute (vectorized)
                downsampled_attributes[attr_name] = attr_values[representative_indices]

        logger.debug(
            f"Exported point cloud from {self.pointcloud.num_points} to "
            f"{self.num_voxels} points"
        )

        return PointCloud(points=downsampled_points, attributes=downsampled_attributes)

    def refresh(self) -> None:
        """Refresh the voxel map based on the current state of the point cloud.

        This recalculates voxel assignments if the point cloud has been modified.
        """
        points = self.pointcloud.points
        num_points = points.shape[0]

        if num_points == 0:
            logger.warning("Empty point cloud.")
            self.voxel_coords = np.empty((0, 3), dtype=np.int32)
            self.inverse_indices = np.empty(0, dtype=np.intp)
            self.origin = np.zeros(3)
            return

        # Recalculate origin
        self.origin = points.min(axis=0)

        # Convert points to voxel coordinates
        voxel_coords_all = np.floor((points - self.origin) / self.voxel_size).astype(
            np.int32
        )

        # Find unique voxels and their indices
        unique_voxels, inverse_indices = np.unique(
            voxel_coords_all, axis=0, return_inverse=True
        )

        self.voxel_coords = unique_voxels
        self.inverse_indices = inverse_indices

        logger.debug(f"Refreshed VoxelMap with {len(unique_voxels)} voxels")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the voxel map.

        Returns:
            Dictionary containing statistics.
        """
        # Count points per voxel using bincount
        if self.num_voxels > 0:
            points_per_voxel = np.bincount(
                self.inverse_indices, minlength=self.num_voxels
            )
        else:
            points_per_voxel = np.array([], dtype=np.intp)

        return {
            "num_voxels": self.num_voxels,
            "num_points": self.pointcloud.num_points,
            "voxel_size": self.voxel_size,
            "compression_ratio": (
                self.pointcloud.num_points / self.num_voxels
                if self.num_voxels > 0
                else 0
            ),
            "min_points_per_voxel": int(points_per_voxel.min())
            if len(points_per_voxel) > 0
            else 0,
            "max_points_per_voxel": int(points_per_voxel.max())
            if len(points_per_voxel) > 0
            else 0,
            "mean_points_per_voxel": (
                float(points_per_voxel.mean()) if len(points_per_voxel) > 0 else 0.0
            ),
            "origin": self.origin.tolist(),
        }
