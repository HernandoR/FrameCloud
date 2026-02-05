"""VoxelMap implementation for spatial downsampling with pandas DataFrames.

This module provides a VoxelMap class that voxelizes point clouds for spatial
downsampling while tracking the indices of points within each voxel, optimized
for pandas DataFrames.
"""

from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from framecloud.exceptions import ArrayShapeError


class VoxelMap:
    """A voxel map for spatial downsampling of point clouds using pandas.

    The VoxelMap aggregates points into voxels based on a specified voxel size.
    It tracks which points belong to each voxel and can aggregate attributes
    using various strategies during export.

    Attributes:
        voxel_size (float): The size of each voxel (uniform in all dimensions).
        voxel_data (pd.DataFrame): DataFrame with voxel info (coords, point_indices).
        origin (np.ndarray): The origin point of the voxel grid (3D coordinates).
        pointcloud: Reference to the source PointCloud (mutable reference or deep copy).
    """

    def __init__(
        self,
        voxel_size: float,
        voxel_data: pd.DataFrame,
        origin: np.ndarray,
        pointcloud: "PointCloud",  # type: ignore # noqa: F821
    ):
        """Initialize a VoxelMap.

        Args:
            voxel_size: Size of each voxel (must be > 0).
            voxel_data: DataFrame with voxel coordinates and point indices.
            origin: Origin of the voxel grid (3D coordinates).
            pointcloud: Reference to the PointCloud (either mutable ref or deep copy).
        """
        if voxel_size <= 0:
            raise ValueError("voxel_size must be greater than 0")
        if origin.shape != (3,):
            logger.error("Origin must be a 3D coordinate.")
            raise ArrayShapeError("Origin must be a 3D coordinate.")

        self.voxel_size = voxel_size
        self.voxel_data = voxel_data
        self.origin = origin
        self.pointcloud = pointcloud

    @classmethod
    def from_pointcloud(
        cls,
        pointcloud: "PointCloud",  # type: ignore # noqa: F821
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
        logger.info(f"Creating VoxelMap with voxel_size={voxel_size}")

        data = pointcloud.data.copy()
        num_points = len(data)

        if num_points == 0:
            logger.warning("Empty point cloud provided.")
            from framecloud.pd.core import PointCloud

            empty_data = pd.DataFrame({"X": [], "Y": [], "Z": []})
            empty_pc = PointCloud(data=empty_data)
            empty_voxel_data = pd.DataFrame(
                {
                    "voxel_x": pd.Series(dtype=np.int32),
                    "voxel_y": pd.Series(dtype=np.int32),
                    "voxel_z": pd.Series(dtype=np.int32),
                    "point_indices": pd.Series(dtype=object),
                }
            )
            return cls(
                voxel_size=voxel_size,
                voxel_data=empty_voxel_data,
                origin=np.zeros(3),
                pointcloud=empty_pc,
            )

        # Calculate origin (minimum coordinates)
        origin = np.array([data["X"].min(), data["Y"].min(), data["Z"].min()])

        # Add original indices to the dataframe
        data["_original_idx"] = np.arange(num_points)

        # Convert points to voxel coordinates (vectorized)
        data["voxel_x"] = ((data["X"] - origin[0]) / voxel_size).astype(np.int32)
        data["voxel_y"] = ((data["Y"] - origin[1]) / voxel_size).astype(np.int32)
        data["voxel_z"] = ((data["Z"] - origin[2]) / voxel_size).astype(np.int32)

        # Group by voxel coordinates and collect point indices
        grouped = data.groupby(["voxel_x", "voxel_y", "voxel_z"])
        voxel_data = (
            grouped.agg(point_indices=("_original_idx", lambda x: [x.to_numpy()]))
            .reset_index()
        )
        # Extract arrays from lists (pandas agg requires scalar or list return)
        voxel_data["point_indices"] = voxel_data["point_indices"].apply(lambda x: x[0])

        # Handle point cloud reference
        if keep_copy:
            from framecloud.pd.core import PointCloud

            # Create a deep copy
            pc_ref = PointCloud(data=pointcloud.data.copy())
        else:
            # Keep mutable reference
            pc_ref = pointcloud

        logger.info(f"Created VoxelMap with {len(voxel_data)} voxels from {num_points} points")

        return cls(
            voxel_size=voxel_size,
            voxel_data=voxel_data,
            origin=origin,
            pointcloud=pc_ref,
        )

    def _compute_representative_indices(
        self, aggregation_method: str = "nearest_to_center"
    ) -> np.ndarray:
        """Compute representative point indices for each voxel.

        Args:
            aggregation_method: Method to select representative point.
                - "nearest_to_center": Select point nearest to voxel center (default)
                - "first": Select first point in each voxel

        Returns:
            Array of representative point indices.
        """
        representative_indices = []

        if aggregation_method == "first":
            # Simply get first index from each voxel
            for point_idx_array in self.voxel_data["point_indices"]:
                representative_indices.append(point_idx_array[0])
        elif aggregation_method == "nearest_to_center":
            # For each voxel, find nearest point using squared distance
            for _, row in self.voxel_data.iterrows():
                voxel_coord = np.array([row["voxel_x"], row["voxel_y"], row["voxel_z"]])
                voxel_center = self.origin + (voxel_coord + 0.5) * self.voxel_size
                point_idx_array = row["point_indices"]

                # Get points in this voxel
                points = self.pointcloud.data.iloc[point_idx_array][["X", "Y", "Z"]].values
                # Use squared distance (avoid sqrt for performance)
                squared_distances = np.sum((points - voxel_center) ** 2, axis=1)
                nearest_idx = point_idx_array[np.argmin(squared_distances)]
                representative_indices.append(nearest_idx)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        return np.array(representative_indices, dtype=np.int32)

    @property
    def num_voxels(self) -> int:
        """Returns the number of voxels."""
        return len(self.voxel_data)

    @property
    def voxel_coords(self) -> np.ndarray:
        """Get voxel coordinates as Nx3 array."""
        return self.voxel_data[["voxel_x", "voxel_y", "voxel_z"]].to_numpy()

    def get_voxel_centers(self) -> np.ndarray:
        """Get the center coordinates of all voxels.

        Returns:
            Nx3 array of voxel center coordinates.
        """
        coords = self.voxel_coords
        return self.origin + (coords + 0.5) * self.voxel_size

    def get_point_indices(self, voxel_coord: tuple[int, int, int]) -> np.ndarray:
        """Get point indices for a specific voxel.

        Args:
            voxel_coord: Voxel coordinate tuple (i, j, k).

        Returns:
            Array of point indices in the specified voxel.
        """
        mask = (
            (self.voxel_data["voxel_x"] == voxel_coord[0])
            & (self.voxel_data["voxel_y"] == voxel_coord[1])
            & (self.voxel_data["voxel_z"] == voxel_coord[2])
        )
        if mask.any():
            return self.voxel_data.loc[mask, "point_indices"].iloc[0]
        return np.array([], dtype=np.int32)

    def export_pointcloud(
        self,
        aggregation_method: str = "nearest_to_center",
        custom_aggregation: dict[str, Callable] | None = None,
    ) -> "PointCloud":  # type: ignore # noqa: F821
        """Export a downsampled point cloud using the voxel map.

        Args:
            aggregation_method: Method to select representative point.
                - "nearest_to_center": Select point nearest to voxel center (default)
                - "first": Select first point in each voxel
            custom_aggregation: Optional dict mapping attribute names to aggregation functions.
                Each function should take a pandas Series and return a single value.

        Returns:
            A new downsampled PointCloud.
        """
        from framecloud.pd.core import PointCloud

        # Compute representative indices at export time
        representative_indices = self._compute_representative_indices(aggregation_method)

        # Get representative points
        representative_data = self.pointcloud.data.iloc[representative_indices].copy()

        # Apply custom aggregation if provided
        if custom_aggregation:
            for attr_name, agg_func in custom_aggregation.items():
                if attr_name in representative_data.columns:
                    # Aggregate for each voxel
                    aggregated_values = []
                    for point_indices in self.voxel_data["point_indices"]:
                        aggregated_value = agg_func(
                            self.pointcloud.data.loc[point_indices, attr_name]
                        )
                        aggregated_values.append(aggregated_value)
                    representative_data[attr_name] = aggregated_values

        representative_data.reset_index(drop=True, inplace=True)

        logger.info(
            f"Exported point cloud from {len(self.pointcloud.data)} to "
            f"{self.num_voxels} points"
        )

        return PointCloud(data=representative_data)

    def refresh(self) -> None:
        """Refresh the voxel map based on the current state of the point cloud.

        This recalculates voxel assignments if the point cloud has been modified.
        """
        data = self.pointcloud.data.copy()
        num_points = len(data)

        if num_points == 0:
            logger.warning("Empty point cloud.")
            self.voxel_data = pd.DataFrame(
                {
                    "voxel_x": pd.Series(dtype=np.int32),
                    "voxel_y": pd.Series(dtype=np.int32),
                    "voxel_z": pd.Series(dtype=np.int32),
                    "point_indices": pd.Series(dtype=object),
                }
            )
            self.origin = np.zeros(3)
            return

        # Recalculate origin
        self.origin = np.array([data["X"].min(), data["Y"].min(), data["Z"].min()])

        # Add original indices
        data["_original_idx"] = np.arange(num_points)

        # Convert points to voxel coordinates
        data["voxel_x"] = ((data["X"] - self.origin[0]) / self.voxel_size).astype(np.int32)
        data["voxel_y"] = ((data["Y"] - self.origin[1]) / self.voxel_size).astype(np.int32)
        data["voxel_z"] = ((data["Z"] - self.origin[2]) / self.voxel_size).astype(np.int32)

        # Group by voxel coordinates
        grouped = data.groupby(["voxel_x", "voxel_y", "voxel_z"])
        voxel_data = (
            grouped.agg(point_indices=("_original_idx", lambda x: [x.to_numpy()]))
            .reset_index()
        )
        voxel_data["point_indices"] = voxel_data["point_indices"].apply(lambda x: x[0])

        self.voxel_data = voxel_data

        logger.info(f"Refreshed VoxelMap with {len(voxel_data)} voxels")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the voxel map.

        Returns:
            Dictionary containing statistics.
        """
        points_per_voxel = self.voxel_data["point_indices"].apply(len)

        return {
            "num_voxels": self.num_voxels,
            "num_points": len(self.pointcloud.data),
            "voxel_size": self.voxel_size,
            "compression_ratio": (
                len(self.pointcloud.data) / self.num_voxels if self.num_voxels > 0 else 0
            ),
            "min_points_per_voxel": (
                int(points_per_voxel.min()) if len(points_per_voxel) > 0 else 0
            ),
            "max_points_per_voxel": (
                int(points_per_voxel.max()) if len(points_per_voxel) > 0 else 0
            ),
            "mean_points_per_voxel": (
                float(points_per_voxel.mean()) if len(points_per_voxel) > 0 else 0
            ),
            "origin": self.origin.tolist(),
        }

