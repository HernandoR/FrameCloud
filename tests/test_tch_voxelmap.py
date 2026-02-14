import pytest
import torch

from framecloud.tch.core import PointCloud
from framecloud.tch.voxelmap import VoxelMap


def _sample_pointcloud(device: torch.device | str | None = None) -> PointCloud:
    points = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [1.2, 1.2, 1.2],
            [2.6, 2.6, 2.6],
        ],
        dtype=torch.float32,
        device=device,
    )
    intensities = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    return PointCloud(points=points, attributes={"intensity": intensities})


def test_voxelmap_creation_cpu():
    pc = _sample_pointcloud()
    voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

    assert voxelmap.num_voxels == 3
    assert voxelmap.get_point_indices((0, 0, 0)).numel() == 2
    centers = voxelmap.get_voxel_centers()
    assert centers.shape == (voxelmap.num_voxels, 3)


def test_voxelmap_export_cpu():
    pc = _sample_pointcloud()
    voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

    downsampled = voxelmap.export_pointcloud()
    assert downsampled.num_points == voxelmap.num_voxels

    downsampled_first = voxelmap.export_pointcloud(aggregation_method="first")
    assert downsampled_first.num_points == voxelmap.num_voxels

    aggregated = voxelmap.export_pointcloud(
        custom_aggregation={"intensity": lambda values: values.mean()}
    )
    assert aggregated.intensity.shape[0] == voxelmap.num_voxels


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_voxelmap_creation_gpu():
    pc = _sample_pointcloud(device="cuda")
    voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0, device="cuda")

    assert voxelmap.voxel_coords.is_cuda
    exported = voxelmap.export_pointcloud()
    assert exported.points.is_cuda
