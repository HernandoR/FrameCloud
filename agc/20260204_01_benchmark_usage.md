# pytest-benchmark Usage Guide

## Overview

The project now uses `pytest-benchmark` for automated performance benchmarking of point cloud operations. Benchmark results are automatically saved with reports generated in multiple formats.

## Running Benchmarks

### Basic Usage

Run all benchmarks (excluding slow tests):
```bash
just benchmark
# or
uv run pytest tests/test_benchmark.py -m "benchmark and not slow" --benchmark-only
```

### Run All Benchmarks (Including Large-Scale Tests)

Run all benchmarks including slow/large-scale tests (10M+ points):
```bash
just benchmark-all
# or
uv run pytest tests/test_benchmark.py -m benchmark --benchmark-only
```

### Compare with Previous Results

Compare current benchmark results with previously saved results:
```bash
just benchmark-compare
# or
uv run pytest tests/test_benchmark.py -m "benchmark and not slow" --benchmark-only --benchmark-compare
```

### Save Baseline for Future Comparisons

Save current benchmark results as a baseline:
```bash
just benchmark-save
# or
uv run pytest tests/test_benchmark.py -m "benchmark and not slow" --benchmark-only --benchmark-save=baseline
```

## Benchmark Reports

After running benchmarks, reports are automatically generated in the `reports/benchmarks/` directory:

### Report Locations

- **JSON Report**: `reports/benchmarks/benchmark.json`
  - Contains detailed statistics for the latest benchmark run
  - Includes min, max, mean, median, stddev, IQR, and outliers
  
- **Historical Data**: `reports/benchmarks/Linux-CPython-3.12-64bit/*.json`
  - Individual JSON files for each benchmark run
  - Organized by commit hash and timestamp
  
- **Histogram**: `reports/benchmarks/histogram-*.svg`
  - Visual representation of benchmark results
  - One SVG file per benchmark group (creation, transformation, sampling, io, attributes)

### View Reports

To see available benchmark reports:
```bash
just benchmark-view
```

## Benchmark Test Organization

The benchmarks are organized into several test groups:

### 1. Creation (`group="creation"`)
- `test_np_create_pointcloud` - Create point clouds with numpy
- `test_pd_create_pointcloud` - Create point clouds with pandas

### 2. Transformation (`group="transformation"`)
- `test_np_transform_pointcloud` - Transform point clouds with numpy
- `test_pd_transform_pointcloud` - Transform point clouds with pandas

### 3. Sampling (`group="sampling"`)
- `test_np_sample_pointcloud` - Sample points with numpy
- `test_pd_sample_pointcloud` - Sample points with pandas

### 4. I/O (`group="io"`)
- `test_np_parquet_write` - Write parquet with numpy
- `test_np_parquet_read` - Read parquet with numpy
- `test_pd_parquet_write` - Write parquet with pandas
- `test_pd_parquet_read` - Read parquet with pandas

### 5. Attributes (`group="attributes"`)
- `test_np_add_attribute` - Add attributes with numpy
- `test_pd_add_attribute` - Add attributes with pandas

### 6. Large-Scale (`group="large-scale"`)
- Marked with `@pytest.mark.slow`
- Tests with 10M+ points
- Only run with `just benchmark-all`

## Configuration

### pytest.ini Configuration

The following pytest-benchmark options are configured in `pytest.ini`:

```ini
--benchmark-autosave          # Automatically save benchmark data
--benchmark-storage=reports/benchmarks  # Where to store benchmark data
--benchmark-json=reports/benchmarks/benchmark.json  # JSON report location
--benchmark-histogram=reports/benchmarks/histogram  # Histogram SVG location
```

### Benchmark Sizes

- **Default benchmarks**: 100K and 1M points (fast feedback)
- **Large-scale benchmarks**: 10M points (marked as `slow`)

## Best Practices

1. **Run benchmarks regularly**: After making performance-related changes
2. **Compare with baseline**: Use `just benchmark-compare` to detect regressions
3. **Save important baselines**: Use `just benchmark-save` before major changes
4. **Check histograms**: Visual inspection of SVG files helps identify distributions
5. **Monitor outliers**: High outlier counts may indicate system instability

## Understanding Benchmark Output

Example output:
```
Name (time in ms)                          Min      Max     Mean    StdDev   Median     IQR    Outliers    OPS    Rounds
-------------------------------------------------------------------------------------------------------------------------
test_np_create_pointcloud[100000]       6.54     7.52     6.60     0.11     6.58     0.04        2;6   151.52    121
test_np_create_pointcloud[1000000]     66.30    70.04    66.80     0.94    66.54     0.15        1;2    14.97     14
```

- **Min/Max/Mean**: Time statistics in milliseconds
- **StdDev**: Standard deviation (lower is better - more consistent)
- **Median**: Middle value (more robust than mean for skewed distributions)
- **IQR**: Interquartile range (measure of spread)
- **Outliers**: Number of outliers (format: mild;extreme)
- **OPS**: Operations per second (1 / Mean)
- **Rounds**: Number of times the test was executed

## Troubleshooting

### Issue: Tests take too long
**Solution**: Use `just benchmark` instead of `just benchmark-all` to skip slow tests

### Issue: Reports directory not found
**Solution**: The directory is automatically created on first run. It's in `.gitignore` and won't be committed.

### Issue: Histogram generation fails
**Solution**: Ensure `pytest-benchmark[histogram]` is installed:
```bash
uv add --dev 'pytest-benchmark[histogram]'
```

## Dependencies

- `pytest-benchmark[histogram]>=5.2.3` - Main benchmarking framework
- `pygal>=3.1.0` - Histogram generation (installed with [histogram] extra)
- `pygaljs>=1.0.2` - JavaScript for interactive histograms
