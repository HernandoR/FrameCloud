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
from framecloud.pd.core import PointCloud


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
        is_copy (bool): Whether the pointcloud is a deep copy (immutable from outside).
    """

    def __init__(
        self,
        voxel_size: float,
        voxel_data: pd.DataFrame,
        origin: np.ndarray,
        pointcloud: PointCloud,
        is_copy: bool = False,
    ):
        """Initialize a VoxelMap.

        Args:
            voxel_size: Size of each voxel (must be > 0).
            voxel_data: DataFrame with voxel coordinates and point indices.
            origin: Origin of the voxel grid (3D coordinates).
            pointcloud: Reference to the PointCloud (either mutable ref or deep copy).
            is_copy: Whether the pointcloud is a deep copy.
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

        data = pointcloud.data.copy()
        num_points = len(data)

        if num_points == 0:
            logger.warning("Empty point cloud provided.")
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
                is_copy=True,
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
        logger.debug(f"Grouping {num_points} points by voxel coordinates")
        grouped = data.groupby(["voxel_x", "voxel_y", "voxel_z"])

        # Use list aggregation and then convert to numpy arrays
        voxel_data = grouped.agg(point_indices=("_original_idx", list)).reset_index()
        # Convert lists to numpy arrays
        voxel_data["point_indices"] = voxel_data["point_indices"].apply(np.array)

        logger.debug(f"Created {len(voxel_data)} voxels")

        # Handle point cloud reference
        if keep_copy:
            # Create a deep copy
            pc_ref = PointCloud(data=pointcloud.data.copy())
            is_copy = True
        else:
            # Keep mutable reference
            pc_ref = pointcloud
            is_copy = False

        logger.debug(
            f"Created VoxelMap with {len(voxel_data)} voxels from {num_points} points"
        )

        return cls(
            voxel_size=voxel_size,
            voxel_data=voxel_data,
            origin=origin,
            pointcloud=pc_ref,
            is_copy=is_copy,
        )

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
    ) -> PointCloud:
        """Export a downsampled point cloud using the voxel map.

        Args:
            aggregation_method: Method to select representative point.
                - "nearest_to_center": Select point nearest to voxel center (default)
                - "first": Select first point in each voxel
            custom_aggregation: Optional dict mapping attribute names to aggregation functions.
                Each function should take a pandas Series and return a single value.
                Cannot include coordinate columns (X, Y, Z).

        Returns:
            A new downsampled PointCloud.
        """
        # Handle empty voxel map
        if self.num_voxels == 0:
            logger.warning("Empty voxel map, returning empty point cloud")
            return PointCloud(data=pd.DataFrame({"X": [], "Y": [], "Z": []}))

        # Validate custom_aggregation doesn't include coordinate columns
        if custom_aggregation:
            forbidden_coord_cols = {"X", "Y", "Z"}
            forbidden_in_agg = forbidden_coord_cols.intersection(
                custom_aggregation.keys()
            )
            if forbidden_in_agg:
                raise ValueError(
                    f"custom_aggregation cannot contain coordinate columns "
                    f"{sorted(forbidden_in_agg)}; these are determined by the "
                    "representative point selection."
                )

        data = self.pointcloud.data

        # Calculate voxel centers for all voxels
        voxel_coords = self.voxel_data[["voxel_x", "voxel_y", "voxel_z"]].values
        voxel_centers = self.origin + (voxel_coords + 0.5) * self.voxel_size

        # Determine representative indices based on aggregation method (vectorized)
        representative_indices = np.zeros(self.num_voxels, dtype=np.int32)

        if aggregation_method == "first":
            # Simply get first index from each voxel
            representative_indices = np.array(
                [idx_array[0] for idx_array in self.voxel_data["point_indices"]]
            )
        elif aggregation_method == "nearest_to_center":
            # Vectorized calculation avoiding iterrows
            point_indices_series = self.voxel_data["point_indices"]

            # Get counts to build expanded arrays
            counts = point_indices_series.apply(len).to_numpy()

            # Flatten all point indices
            all_point_indices = np.concatenate(point_indices_series.to_numpy())

            # For each point, record which voxel it belongs to
            voxel_idx_for_points = np.repeat(
                np.arange(len(point_indices_series)), counts
            )

            # Get coordinates for all points
            points = data.iloc[all_point_indices][["X", "Y", "Z"]].to_numpy()

            # Corresponding voxel centers for each point
            centers_for_points = voxel_centers[voxel_idx_for_points]

            # Use squared distance (avoid sqrt for performance)
            squared_distances = np.sum((points - centers_for_points) ** 2, axis=1)

            # Build DataFrame to find nearest point per voxel
            distance_df = pd.DataFrame(
                {
                    "voxel_idx": voxel_idx_for_points,
                    "point_idx": all_point_indices,
                    "sqdist": squared_distances,
                }
            )

            # For each voxel, select the point with minimum squared distance
            nearest_rows = distance_df.loc[
                distance_df.groupby("voxel_idx")["sqdist"].idxmin()
            ]

            # Ensure ordering by voxel index to match voxel_data order
            nearest_rows = nearest_rows.sort_values("voxel_idx")
            representative_indices = nearest_rows["point_idx"].to_numpy()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Get representative points data
        representative_data = data.iloc[representative_indices].copy()

        # Apply custom aggregation using groupby if provided
        if custom_aggregation:
            # Create a temporary mapping from original indices to voxel indices
            idx_to_voxel: dict[int, int] = {}
            for voxel_idx, point_indices in enumerate(self.voxel_data["point_indices"]):
                for point_idx in point_indices:
                    idx_to_voxel[point_idx] = voxel_idx

            # Add voxel_idx column to original data for grouping
            temp_data = data.copy()
            temp_data["_voxel_idx"] = temp_data.index.map(idx_to_voxel)

            # Apply custom aggregation for each attribute using groupby
            for attr_name, agg_func in custom_aggregation.items():
                if attr_name in temp_data.columns:
                    aggregated = temp_data.groupby("_voxel_idx")[attr_name].apply(
                        agg_func
                    )
                    # Map back to representative_data order
                    representative_data[attr_name] = [
                        aggregated.loc[voxel_idx]
                        for voxel_idx in range(len(self.voxel_data))
                    ]

        representative_data.reset_index(drop=True, inplace=True)

        logger.debug(
            f"Exported point cloud from {len(data)} to {self.num_voxels} points"
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
        data["voxel_x"] = ((data["X"] - self.origin[0]) / self.voxel_size).astype(
            np.int32
        )
        data["voxel_y"] = ((data["Y"] - self.origin[1]) / self.voxel_size).astype(
            np.int32
        )
        data["voxel_z"] = ((data["Z"] - self.origin[2]) / self.voxel_size).astype(
            np.int32
        )

        # Group by voxel coordinates
        grouped = data.groupby(["voxel_x", "voxel_y", "voxel_z"])
        voxel_data = grouped.agg(
            point_indices=("_original_idx", lambda x: [x.to_numpy()])
        ).reset_index()
        voxel_data["point_indices"] = voxel_data["point_indices"].apply(lambda x: x[0])

        self.voxel_data = voxel_data

        logger.debug(f"Refreshed VoxelMap with {len(voxel_data)} voxels")

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
                len(self.pointcloud.data) / self.num_voxels
                if self.num_voxels > 0
                else 0
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
