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


def _to_numpy_array(series):
    """直接将分组后的Series转换为NumPy数组（聚合函数）"""
    return np.asarray(series.values)


class VoxelMap:
    """A voxel map for spatial downsampling of point clouds using pandas.

    The VoxelMap aggregates points into voxels based on a specified voxel size.
    It tracks which points belong to each voxel and can aggregate attributes
    using various strategies during export.

    Attributes:
        voxel_size (float): The size of each voxel (uniform in all dimensions).
        voxel_coords (np.ndarray): Nx3 array of voxel coordinates for each unique voxel.
        voxel_indices_per_point (np.ndarray): Array mapping each point to its voxel index.
        origin (np.ndarray): The origin point of the voxel grid (3D coordinates).
        pointcloud: Reference to the source PointCloud (mutable reference or deep copy).
        is_copy (bool): Whether the pointcloud is a deep copy (immutable from outside).
    """

    def __init__(
        self,
        voxel_size: float,
        voxel_coords: np.ndarray,
        voxel_indices_per_point: np.ndarray,
        origin: np.ndarray,
        pointcloud: PointCloud,
        is_copy: bool = False,
    ):
        """Initialize a VoxelMap.

        Args:
            voxel_size: Size of each voxel (must be > 0).
            voxel_coords: Nx3 array of unique voxel coordinates.
            voxel_indices_per_point: Array mapping each point to its voxel index.
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
        self._voxel_coords = voxel_coords
        self._voxel_indices_per_point = voxel_indices_per_point
        self.origin = origin
        self.pointcloud = pointcloud
        self._is_copy = is_copy

    @property
    def is_copy(self) -> bool:
        """Returns whether the pointcloud is a deep copy (read-only)."""
        return self._is_copy

    @classmethod
    def _build_origin_and_voxel_data(
        cls,
        data: pd.DataFrame,
        voxel_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        origin = data[["X", "Y", "Z"]].min().to_numpy()
        voxel_coords_all = np.floor(
            (data[["X", "Y", "Z"]].to_numpy() - origin) / voxel_size
        ).astype(np.int32)
        unique_voxels, voxel_indices_per_point = np.unique(
            voxel_coords_all, axis=0, return_inverse=True
        )
        return origin, unique_voxels, voxel_indices_per_point

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

        data = pointcloud.data
        num_points = len(data)

        if num_points == 0:
            logger.warning("Empty point cloud provided.")
            empty_data = pd.DataFrame({"X": [], "Y": [], "Z": []})
            empty_pc = PointCloud(data=empty_data)
            return cls(
                voxel_size=voxel_size,
                voxel_coords=np.empty((0, 3), dtype=np.int32),
                voxel_indices_per_point=np.empty(0, dtype=np.intp),
                origin=np.zeros(3),
                pointcloud=empty_pc,
                is_copy=True,
            )

        origin, voxel_coords, voxel_indices_per_point = cls._build_origin_and_voxel_data(
            data, voxel_size
        )

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
            f"Created VoxelMap with {len(voxel_coords)} voxels from {num_points} points"
        )

        return cls(
            voxel_size=voxel_size,
            voxel_coords=voxel_coords,
            voxel_indices_per_point=voxel_indices_per_point,
            origin=origin,
            pointcloud=pc_ref,
            is_copy=is_copy,
        )

    @property
    def num_voxels(self) -> int:
        """Returns the number of voxels."""
        return len(self._voxel_coords)

    @property
    def voxel_coords(self) -> np.ndarray:
        """Get voxel coordinates as Nx3 array."""
        return self._voxel_coords

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
        voxel_arr = np.array(voxel_coord, dtype=np.int32)
        matches = np.all(self._voxel_coords == voxel_arr, axis=1)
        voxel_idx = np.where(matches)[0]
        if len(voxel_idx) == 0:
            return np.array([], dtype=np.int32)
        return np.where(self._voxel_indices_per_point == voxel_idx[0])[0]

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

        points = data[["X", "Y", "Z"]].to_numpy()

        representative_indices = np.zeros(self.num_voxels, dtype=np.int32)

        if aggregation_method == "first":
            sorted_order = np.argsort(self._voxel_indices_per_point, kind="stable")
            _, first_occurrence = np.unique(
                self._voxel_indices_per_point[sorted_order], return_index=True
            )
            representative_indices = sorted_order[first_occurrence]
        elif aggregation_method == "nearest_to_center":
            voxel_centers = self.origin + (self._voxel_coords + 0.5) * self.voxel_size
            point_voxel_centers = voxel_centers[self._voxel_indices_per_point]
            squared_distances = np.sum((points - point_voxel_centers) ** 2, axis=1)

            sorted_order = np.lexsort((squared_distances, self._voxel_indices_per_point))
            _, first_occurrence = np.unique(
                self._voxel_indices_per_point[sorted_order], return_index=True
            )
            representative_indices = sorted_order[first_occurrence]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Get representative points data
        representative_data = data.iloc[representative_indices].copy()

        if custom_aggregation:
            temp_data = data.copy()
            temp_data["_voxel_idx"] = self._voxel_indices_per_point
            for attr_name, agg_func in custom_aggregation.items():
                if attr_name in temp_data.columns:
                    aggregated = temp_data.groupby("_voxel_idx")[attr_name].apply(
                        agg_func
                    )
                    aggregated = aggregated.reindex(
                        range(self.num_voxels), fill_value=np.nan
                    )
                    representative_data[attr_name] = aggregated.values

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
            self._voxel_coords = np.empty((0, 3), dtype=np.int32)
            self._voxel_indices_per_point = np.empty(0, dtype=np.intp)
            self.origin = np.zeros(3)
            return

        self.origin, self._voxel_coords, self._voxel_indices_per_point = (
            self._build_origin_and_voxel_data(
                data,
                self.voxel_size,
            )
        )

        logger.debug(f"Refreshed VoxelMap with {len(self._voxel_coords)} voxels")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the voxel map.

        Returns:
            Dictionary containing statistics.
        """
        points_per_voxel = (
            np.bincount(self._voxel_indices_per_point, minlength=self.num_voxels)
            if self.num_voxels > 0
            else np.array([], dtype=np.intp)
        )

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


def make_paser():
    import argparse

    parser = argparse.ArgumentParser(description="Run VoxelMap pipeline")
    parser.add_argument(
        "--num_points",
        type=int,
        default=10_000_000,
        help="Number of points in the point cloud",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=1.0,
        help="Voxel size for the VoxelMap",
    )
    return parser


def main():
    parser = make_paser()
    args = parser.parse_args()

    num_points = args.num_points
    voxel_size = args.voxel_size

    # Create random point cloud
    data = pd.DataFrame(
        {
            "X": np.random.randn(num_points).astype(np.float32),
            "Y": np.random.randn(num_points).astype(np.float32),
            "Z": np.random.randn(num_points).astype(np.float32),
        }
    )
    pc = PointCloud(data=data)
    logger.info(f"Created point cloud with {num_points} points")
    # Create VoxelMap
    voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=voxel_size)
    logger.info(
        f"Created VoxelMap with {voxelmap.num_voxels} voxels from {num_points} points"
    )
    # Export downsampled point cloud
    downsampled_pc = voxelmap.export_pointcloud()
    logger.info(
        f"Exported downsampled point cloud with {len(downsampled_pc.data)} points"
    )


if __name__ == "__main__":
    main()
