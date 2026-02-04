"""VoxelMap implementation for spatial downsampling with pandas DataFrames.

This module provides a VoxelMap class that voxelizes point clouds for spatial
downsampling while tracking the indices of points within each voxel, optimized
for pandas DataFrames.
"""

from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from framecloud.exceptions import ArrayShapeError


class VoxelMap(BaseModel):
    """A voxel map for spatial downsampling of point clouds using pandas.

    The VoxelMap aggregates points into voxels based on a specified voxel size.
    It tracks which points belong to each voxel and can aggregate attributes
    using various strategies (nearest to center by default).

    Attributes:
        voxel_size (float): The size of each voxel (uniform in all dimensions).
        voxel_data (pd.DataFrame): DataFrame with voxel info (coords, representative_idx, etc).
        origin (np.ndarray): The origin point of the voxel grid (3D coordinates).
        point_count (int): Original number of points used to create the voxel map.
        keep_copy (bool): Whether to keep a deep copy of the point cloud data.
        pointcloud_copy (pd.DataFrame | None): Optional deep copy of point cloud data.
    """

    model_config = {"arbitrary_types_allowed": True}

    voxel_size: float = Field(..., gt=0, description="Size of each voxel")
    voxel_data: pd.DataFrame = Field(
        ...,
        description="DataFrame with voxel coordinates and metadata",
    )
    origin: np.ndarray = Field(..., description="Origin of the voxel grid")
    point_count: int = Field(..., ge=0, description="Number of points in source cloud")
    keep_copy: bool = Field(
        default=False, description="Whether to keep a copy of point cloud data"
    )
    pointcloud_copy: pd.DataFrame | None = Field(
        default=None, description="Optional copy of point cloud data"
    )

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

        data = pointcloud.data.copy()
        num_points = len(data)

        if num_points == 0:
            logger.warning("Empty point cloud provided.")
            empty_voxel_data = pd.DataFrame(
                {
                    "voxel_x": pd.Series(dtype=np.int32),
                    "voxel_y": pd.Series(dtype=np.int32),
                    "voxel_z": pd.Series(dtype=np.int32),
                    "representative_idx": pd.Series(dtype=np.int32),
                    "point_indices": pd.Series(dtype=object),
                }
            )
            return cls(
                voxel_size=voxel_size,
                voxel_data=empty_voxel_data,
                origin=np.zeros(3),
                point_count=0,
                keep_copy=keep_copy,
                pointcloud_copy=None,
            )

        # Calculate origin (minimum coordinates)
        origin = np.array([data["X"].min(), data["Y"].min(), data["Z"].min()])

        # Add original indices to the dataframe
        data["_original_idx"] = np.arange(num_points)

        # Convert points to voxel coordinates (vectorized)
        data["voxel_x"] = ((data["X"] - origin[0]) / voxel_size).astype(np.int32)
        data["voxel_y"] = ((data["Y"] - origin[1]) / voxel_size).astype(np.int32)
        data["voxel_z"] = ((data["Z"] - origin[2]) / voxel_size).astype(np.int32)

        # Group by voxel coordinates
        grouped = data.groupby(["voxel_x", "voxel_y", "voxel_z"])

        if aggregation_method == "first":
            # Select first point in each voxel
            voxel_data = (
                grouped.agg(
                    representative_idx=("_original_idx", "first"),
                    point_indices=("_original_idx", lambda x: [x.to_numpy()]),
                )
                .reset_index()
                .astype({"representative_idx": np.int32})
            )
            # Extract arrays from lists (pandas agg requires scalar or list return)
            voxel_data["point_indices"] = voxel_data["point_indices"].apply(
                lambda x: x[0]
            )
        elif aggregation_method == "nearest_to_center":
            # Select point nearest to voxel center (vectorized approach)
            def select_nearest_to_center(group, name):
                # name contains the (voxel_x, voxel_y, voxel_z) tuple
                voxel_coord = np.array(name)
                voxel_center = origin + (voxel_coord + 0.5) * voxel_size

                # Calculate distances to center
                points = group[["X", "Y", "Z"]].values
                distances = np.linalg.norm(points - voxel_center, axis=1)

                # Find nearest point
                nearest_idx = group["_original_idx"].iloc[np.argmin(distances)]

                return pd.Series(
                    {
                        "representative_idx": nearest_idx,
                        "point_indices": group["_original_idx"].to_numpy(),
                    }
                )

            # Apply with group_keys=True to pass the multi-index name
            voxel_data = []
            for name, group in grouped:
                result = select_nearest_to_center(group, name)
                voxel_data.append(
                    {
                        "voxel_x": name[0],
                        "voxel_y": name[1],
                        "voxel_z": name[2],
                        "representative_idx": result["representative_idx"],
                        "point_indices": result["point_indices"],
                    }
                )
            voxel_data = pd.DataFrame(voxel_data)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Optionally keep a copy of the point cloud
        pointcloud_copy = None
        if keep_copy:
            pointcloud_copy = pointcloud.data.copy()

        logger.info(
            f"Created VoxelMap with {len(voxel_data)} voxels from {num_points} points"
        )

        return cls(
            voxel_size=voxel_size,
            voxel_data=voxel_data,
            origin=origin,
            point_count=num_points,
            keep_copy=keep_copy,
            pointcloud_copy=pointcloud_copy,
        )

    @property
    def num_voxels(self) -> int:
        """Returns the number of voxels."""
        return len(self.voxel_data)

    @property
    def voxel_coords(self) -> np.ndarray:
        """Get voxel coordinates as Nx3 array."""
        return self.voxel_data[["voxel_x", "voxel_y", "voxel_z"]].to_numpy()

    @property
    def representative_indices(self) -> np.ndarray:
        """Get representative point indices."""
        return self.voxel_data["representative_idx"].to_numpy()

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

    def downsample(
        self,
        pointcloud: "PointCloud",  # type: ignore # noqa: F821
        custom_aggregation: dict[str, Callable] | None = None,
    ) -> "PointCloud":  # type: ignore # noqa: F821
        """Create a downsampled point cloud using the voxel map.

        Args:
            pointcloud: The original point cloud (must match the one used to create the map).
            custom_aggregation: Optional dict mapping attribute names to aggregation functions.
                Each function should take a pandas Series and return a single value.

        Returns:
            A new downsampled PointCloud.
        """
        from framecloud.pd.core import PointCloud

        if len(pointcloud.data) != self.point_count:
            logger.warning(
                "Point cloud size mismatch. This may indicate the point cloud "
                "has been modified since the voxel map was created."
            )

        # Get representative points
        representative_data = pointcloud.data.iloc[self.representative_indices].copy()

        # Apply custom aggregation if provided
        if custom_aggregation:
            for attr_name, agg_func in custom_aggregation.items():
                if attr_name in representative_data.columns:
                    # Aggregate for each voxel
                    aggregated_values = []
                    for point_indices in self.voxel_data["point_indices"]:
                        aggregated_value = agg_func(
                            pointcloud.data.loc[point_indices, attr_name]
                        )
                        aggregated_values.append(aggregated_value)
                    representative_data[attr_name] = aggregated_values

        representative_data.reset_index(drop=True, inplace=True)

        logger.info(
            f"Downsampled point cloud from {self.point_count} to "
            f"{self.num_voxels} points"
        )

        return PointCloud(data=representative_data)

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
        points_per_voxel = self.voxel_data["point_indices"].apply(len)

        return {
            "num_voxels": self.num_voxels,
            "num_points": self.point_count,
            "voxel_size": self.voxel_size,
            "compression_ratio": (
                self.point_count / self.num_voxels if self.num_voxels > 0 else 0
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
