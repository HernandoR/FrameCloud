"""Shared voxel aggregation utilities."""

from __future__ import annotations

import numpy as np


def aggregate_voxels_numpy(
    voxel_x: np.ndarray,
    voxel_y: np.ndarray,
    voxel_z: np.ndarray,
    point_indices: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Group points into voxels using a fast lexsort-based approach.

    Returns unique voxel coordinates, the per-voxel point index slices,
    and a voxel index for every point (aligned to the original order).
    """
    if len(point_indices) == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            [],
            np.empty(0, dtype=np.intp),
        )

    sort_idx = np.lexsort([voxel_z, voxel_y, voxel_x])
    sorted_origins = point_indices[sort_idx]
    sorted_keys = np.stack(
        (voxel_x[sort_idx], voxel_y[sort_idx], voxel_z[sort_idx]), axis=1
    )

    mask = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    split_indices = np.flatnonzero(mask) + 1

    point_indices_per_voxel = np.split(sorted_origins, split_indices)
    unique_coords = sorted_keys[np.r_[0, split_indices]]

    voxel_index_per_point = np.empty_like(sort_idx, dtype=np.intp)
    voxel_index_per_point[sort_idx] = np.repeat(
        np.arange(len(point_indices_per_voxel), dtype=np.intp),
        [len(indices) for indices in point_indices_per_voxel],
    )

    return unique_coords, point_indices_per_voxel, voxel_index_per_point
