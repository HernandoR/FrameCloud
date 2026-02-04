# VoxelMap Usage Guide

## Overview

The `VoxelMap` class provides efficient spatial downsampling of point clouds by aggregating points into voxels. Two implementations are provided:
- `framecloud.np.VoxelMap` - NumPy-based implementation
- `framecloud.pd.VoxelMap` - Pandas-based implementation

## Basic Usage

### NumPy-based VoxelMap

```python
import numpy as np
from framecloud.np import PointCloud, VoxelMap

# Create a point cloud
points = np.random.rand(1000, 3) * 100  # 1000 random points
pc = PointCloud(points=points)

# Create a voxel map with 1.0 unit voxel size
voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

print(f"Original points: {pc.num_points}")
print(f"Voxels created: {voxelmap.num_voxels}")
print(f"Compression ratio: {voxelmap.get_statistics()['compression_ratio']:.2f}x")

# Downsample the point cloud
downsampled_pc = voxelmap.downsample(pc)
print(f"Downsampled points: {downsampled_pc.num_points}")
```

### Pandas-based VoxelMap

```python
import pandas as pd
from framecloud.pd import PointCloud, VoxelMap

# Create a point cloud from a DataFrame
data = pd.DataFrame({
    'X': [0.1, 0.2, 0.9, 1.1, 2.5],
    'Y': [0.1, 0.2, 0.9, 1.1, 2.5],
    'Z': [0.1, 0.2, 0.9, 1.1, 2.5],
    'intensity': [100, 200, 300, 400, 500],
})
pc = PointCloud(data=data)

# Create voxel map
voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)
downsampled_pc = voxelmap.downsample(pc)
```

## Aggregation Methods

Two aggregation methods are supported:

### 1. Nearest to Center (default)

Selects the point nearest to the voxel center as the representative point.

```python
voxelmap = VoxelMap.from_pointcloud(
    pc, 
    voxel_size=1.0, 
    aggregation_method="nearest_to_center"
)
```

### 2. First

Selects the first point encountered in each voxel.

```python
voxelmap = VoxelMap.from_pointcloud(
    pc, 
    voxel_size=1.0, 
    aggregation_method="first"
)
```

## Custom Attribute Aggregation

You can provide custom aggregation functions for attributes:

```python
import numpy as np
from framecloud.np import PointCloud, VoxelMap

# Create point cloud with attributes
points = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
intensities = np.array([100.0, 200.0, 300.0])
pc = PointCloud(points=points, attributes={'intensity': intensities})

# Create voxel map
voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

# Downsample with custom aggregation (mean intensity)
custom_agg = {
    'intensity': lambda x: np.mean(x)  # Average intensity in each voxel
}
downsampled = voxelmap.downsample(pc, custom_aggregation=custom_agg)

print(f"Original intensities: {pc.attributes['intensity']}")
print(f"Downsampled intensity: {downsampled.attributes['intensity']}")
```

For pandas:

```python
import pandas as pd
from framecloud.pd import PointCloud, VoxelMap

# Create point cloud
data = pd.DataFrame({
    'X': [0.1, 0.2, 0.3],
    'Y': [0.1, 0.2, 0.3],
    'Z': [0.1, 0.2, 0.3],
    'intensity': [100, 200, 300],
})
pc = PointCloud(data=data)

# Create voxel map and downsample with custom aggregation
voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0)
custom_agg = {
    'intensity': lambda x: x.mean()  # Pandas Series mean
}
downsampled = voxelmap.downsample(pc, custom_aggregation=custom_agg)
```

## Accessing Voxel Information

### Get voxel centers

```python
centers = voxelmap.get_voxel_centers()
print(f"Voxel centers shape: {centers.shape}")
```

### Get point indices in a specific voxel

```python
# Get indices of points in voxel (0, 0, 0)
point_indices = voxelmap.get_point_indices((0, 0, 0))
print(f"Points in voxel (0,0,0): {point_indices}")
```

### Get statistics

```python
stats = voxelmap.get_statistics()
print(f"Number of voxels: {stats['num_voxels']}")
print(f"Number of points: {stats['num_points']}")
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"Min points per voxel: {stats['min_points_per_voxel']}")
print(f"Max points per voxel: {stats['max_points_per_voxel']}")
print(f"Mean points per voxel: {stats['mean_points_per_voxel']:.2f}")
```

## Recalculation

If the point cloud is modified, you can recalculate the voxel map:

```python
# Create initial voxel map
voxelmap1 = VoxelMap.from_pointcloud(pc, voxel_size=1.0)

# Modify point cloud
pc.points = pc.points + 0.5

# Recalculate voxel map with same parameters
voxelmap2 = voxelmap1.recalculate(pc)
```

## Optional Deep Copy

By default, VoxelMap doesn't keep a deep copy of the point cloud (for memory efficiency). You can optionally keep a copy:

```python
voxelmap = VoxelMap.from_pointcloud(pc, voxel_size=1.0, keep_copy=True)

# Check if copy is kept
print(f"Has copy: {voxelmap.pointcloud_copy is not None}")
```

## Important Notes

1. **Memory Efficiency**: VoxelMap tracks point indices rather than copying point data, making it memory-efficient for large point clouds.

2. **Mutable Point Clouds**: Since VoxelMap stores references (indices) rather than copies, modifying the original point cloud after creating a VoxelMap may cause mismatches. Use `recalculate()` if you modify the point cloud.

3. **Voxel Size Selection**: Choose voxel size based on your application:
   - Smaller voxels: Less downsampling, more detail preserved
   - Larger voxels: More downsampling, less memory usage

4. **Vectorization**: Both implementations use vectorized operations to avoid loops and maximize performance.

## Performance Comparison

- **NumPy implementation**: Best for point clouds that fit in memory as numpy arrays
- **Pandas implementation**: Best when working with DataFrames or need additional pandas functionality

Both implementations are optimized for performance and avoid explicit loops where possible.
