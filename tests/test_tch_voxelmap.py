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


def test_voxelmap_refresh_cpu():
    pc = _sample_pointcloud()
    voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

    # Capture initial state
    initial_num_voxels = voxelmap.num_voxels
    initial_centers = voxelmap.get_voxel_centers().clone()

    # Calling refresh should not raise and should preserve core invariants
    voxelmap.refresh()

    assert voxelmap.num_voxels == initial_num_voxels
    refreshed_centers = voxelmap.get_voxel_centers()
    assert refreshed_centers.shape == initial_centers.shape


def test_voxelmap_get_statistics_cpu():
    pc = _sample_pointcloud()
    voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

    stats = voxelmap.get_statistics()

    # At minimum, statistics should report the number of voxels correctly
    assert isinstance(stats, dict)
    assert "num_voxels" in stats
    assert stats["num_voxels"] == voxelmap.num_voxels
    assert "num_points" in stats
    assert "voxel_size" in stats
    assert "compression_ratio" in stats


def test_voxelmap_custom_aggregation_multiple_attributes_cpu():
    # Construct a point cloud with multiple attributes
    points = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [1.2, 1.2, 1.2],
            [2.6, 2.6, 2.6],
        ],
        dtype=torch.float32,
    )
    intensities = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    reflectance = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)

    pc = PointCloud(
        points=points,
        attributes={
            "intensity": intensities,
            "reflectance": reflectance,
        },
    )

    voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

    aggregated = voxelmap.export_pointcloud(
        custom_aggregation={
            "intensity": lambda values: values.mean(),
            "reflectance": lambda values: values.max(),
        }
    )

    assert aggregated.intensity.shape[0] == voxelmap.num_voxels
    assert aggregated.reflectance.shape[0] == voxelmap.num_voxels


def test_single_point_voxels_cpu():
    # Create points far enough apart so each falls into its own voxel
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    intensities = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    pc = PointCloud(points=points, attributes={"intensity": intensities})

    voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

    # Each point should be in its own voxel
    assert voxelmap.num_voxels == pc.num_points

    downsampled = voxelmap.export_pointcloud()
    assert downsampled.num_points == voxelmap.num_voxels


def test_invalid_aggregation_method_cpu():
    pc = _sample_pointcloud()
    voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

    with pytest.raises(ValueError):
        voxelmap.export_pointcloud(aggregation_method="invalid_method")
