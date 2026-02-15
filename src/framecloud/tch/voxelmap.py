"""VoxelMap implementation using torch tensors."""

from __future__ import annotations

from typing import Any, Callable

import torch
from loguru import logger

from framecloud.exceptions import ArrayShapeError
from framecloud.tch.core import PointCloud


class VoxelMap:
    """Torch-based voxel map supporting optional GPU execution."""

    def __init__(
        self,
        voxel_size: float,
        voxel_coords: torch.Tensor,
        voxel_indice_per_point: torch.Tensor,
        origin: torch.Tensor,
        pointcloud: PointCloud,
        is_copy: bool = False,
    ):
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
        self.voxel_indice_per_point = voxel_indice_per_point
        self.origin = origin
        self.pointcloud = pointcloud
        self._is_copy = is_copy

    @property
    def is_copy(self) -> bool:
        return self._is_copy

    @classmethod
    def from_pointcloud(
        cls,
        pointcloud: PointCloud,
        voxel_size: float,
        keep_copy: bool = False,
        device: torch.device | str | None = None,
    ) -> "VoxelMap":
        logger.debug(f"Creating torch VoxelMap with voxel_size={voxel_size}")

        pc_ref = pointcloud.clone() if keep_copy else pointcloud
        if device is not None:
            pc_ref = pc_ref.to(device)

        points = pc_ref.points
        num_points = points.shape[0]

        if num_points == 0:
            logger.warning("Empty point cloud provided.")
            empty_points = torch.empty((0, 3), device=points.device)
            empty_pc = PointCloud(points=empty_points)
            return cls(
                voxel_size=voxel_size,
                voxel_coords=torch.empty((0, 3), dtype=torch.int32, device=points.device),
                voxel_indice_per_point=torch.empty(0, dtype=torch.int64, device=points.device),
                origin=torch.zeros(3, device=points.device),
                pointcloud=empty_pc,
                is_copy=True,
            )

        origin = torch.min(points, dim=0).values
        voxel_coords_all = torch.floor((points - origin) / voxel_size).to(torch.int32)

        unique_voxels, voxel_indice_per_point = torch.unique(
            voxel_coords_all, dim=0, return_inverse=True
        )

        logger.debug(
            f"Created torch VoxelMap with {len(unique_voxels)} voxels from {num_points} points"
        )

        pc_result = pc_ref
        is_copy_flag = keep_copy or (pc_result is not pointcloud)

        return cls(
            voxel_size=voxel_size,
            voxel_coords=unique_voxels,
            voxel_indice_per_point=voxel_indice_per_point,
            origin=origin,
            pointcloud=pc_result,
            is_copy=is_copy_flag,
        )

    @property
    def num_voxels(self) -> int:
        return int(self.voxel_coords.shape[0])

    def get_voxel_centers(self) -> torch.Tensor:
        return self.origin + (self.voxel_coords.to(self.origin.dtype) + 0.5) * self.voxel_size

    def get_point_indices(self, voxel_coord: tuple[int, int, int]) -> torch.Tensor:
        voxel_arr = torch.tensor(
            voxel_coord, device=self.voxel_coords.device, dtype=self.voxel_coords.dtype
        )
        matches = torch.all(self.voxel_coords == voxel_arr, dim=1)
        voxel_idx = torch.nonzero(matches, as_tuple=False).flatten()
        if len(voxel_idx) == 0:
            return torch.tensor([], device=self.voxel_coords.device, dtype=torch.int64)
        return torch.nonzero(
            self.voxel_indice_per_point == voxel_idx[0], as_tuple=False
        ).flatten()

    def export_pointcloud(
        self,
        aggregation_method: str = "nearest_to_center",
        custom_aggregation: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> PointCloud:
        if self.num_voxels == 0:
            logger.warning("Empty voxel map, returning empty point cloud")
            empty = torch.empty((0, 3), device=self.origin.device)
            return PointCloud(points=empty)

        points = self.pointcloud.points
        voxel_idx = self.voxel_indice_per_point

        if aggregation_method == "first":
            sorted_order = torch.argsort(voxel_idx)
            _, counts = torch.unique(voxel_idx[sorted_order], return_counts=True)
            starts = torch.cumsum(
                torch.cat(
                    [
                        torch.zeros(
                            1, device=counts.device, dtype=counts.dtype
                        ),
                        counts[:-1],
                    ]
                ),
                dim=0,
            ).to(torch.long)
            representative_indices = sorted_order[starts]
        elif aggregation_method == "nearest_to_center":
            voxel_centers = self.get_voxel_centers()
            point_centers = voxel_centers[voxel_idx]
            squared_distances = torch.sum((points - point_centers) ** 2, dim=1)

            distance_order = torch.argsort(squared_distances)
            chosen_indices: list[int] = []
            seen_voxels: set[int] = set()
            for idx in distance_order.tolist():
                voxel_value = int(voxel_idx[idx])
                if voxel_value in seen_voxels:
                    continue
                seen_voxels.add(voxel_value)
                chosen_indices.append(idx)
                if len(chosen_indices) == self.num_voxels:
                    break
            representative_indices = torch.tensor(
                chosen_indices, device=points.device, dtype=torch.long
            )
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        downsampled_points = points[representative_indices]
        downsampled_attributes: dict[str, torch.Tensor] = {}

        for attr_name, attr_values in self.pointcloud.attributes.items():
            if custom_aggregation and attr_name in custom_aggregation:
                aggregated_values = []
                for voxel_index in range(self.num_voxels):
                    mask = voxel_idx == voxel_index
                    aggregated_values.append(custom_aggregation[attr_name](attr_values[mask]))
                stacked = (
                    torch.stack(aggregated_values)
                    if isinstance(aggregated_values[0], torch.Tensor)
                    else torch.tensor(aggregated_values, device=downsampled_points.device)
                )
                downsampled_attributes[attr_name] = stacked
            else:
                downsampled_attributes[attr_name] = attr_values[representative_indices]

        logger.debug(
            f"Exported point cloud from {self.pointcloud.num_points} to {self.num_voxels} points"
        )

        return PointCloud(points=downsampled_points, attributes=downsampled_attributes)

    def refresh(self) -> None:
        points = self.pointcloud.points
        num_points = points.shape[0]

        if num_points == 0:
            logger.warning("Empty point cloud.")
            self.voxel_coords = torch.empty((0, 3), dtype=torch.int32, device=points.device)
            self.voxel_indice_per_point = torch.empty(0, dtype=torch.int64, device=points.device)
            self.origin = torch.zeros(3, device=points.device)
            return

        self.origin = torch.min(points, dim=0).values
        voxel_coords_all = torch.floor((points - self.origin) / self.voxel_size).to(torch.int32)

        unique_voxels, voxel_indice_per_point = torch.unique(
            voxel_coords_all, dim=0, return_inverse=True
        )

        self.voxel_coords = unique_voxels
        self.voxel_indice_per_point = voxel_indice_per_point

        logger.debug(f"Refreshed torch VoxelMap with {len(unique_voxels)} voxels")

    def get_statistics(self) -> dict[str, Any]:
        if self.num_voxels > 0:
            points_per_voxel = torch.bincount(
                self.voxel_indice_per_point.to(torch.int64),
                minlength=self.num_voxels,
            )
        else:
            points_per_voxel = torch.tensor([], device=self.origin.device)

        return {
            "num_voxels": self.num_voxels,
            "num_points": self.pointcloud.num_points,
            "voxel_size": self.voxel_size,
            "compression_ratio": (
                self.pointcloud.num_points / self.num_voxels if self.num_voxels > 0 else 0
            ),
            "min_points_per_voxel": int(points_per_voxel.min()) if len(points_per_voxel) > 0 else 0,
            "max_points_per_voxel": int(points_per_voxel.max()) if len(points_per_voxel) > 0 else 0,
            "mean_points_per_voxel": (
                float(points_per_voxel.float().mean()) if len(points_per_voxel) > 0 else 0.0
            ),
            "origin": self.origin.tolist(),
        }
