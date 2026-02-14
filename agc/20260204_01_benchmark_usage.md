# pytest-benchmark Usage Guide

## Overview

The project uses `pytest-benchmark` for automated performance benchmarking of point cloud operations. Benchmark results are automatically saved with reports generated in multiple formats.

## Running Benchmarks

### Basic Usage

Run fast benchmarks (excludes slow 50M+ cases):
```bash
just benchmark
# or
uv run pytest tests/test_benchmark.py -m "benchmark and not slow" --benchmark-only
```

Run large/slow benchmarks and merge with the latest fast results:
```bash
just benchmark-large
# or
uv run pytest tests/test_benchmark.py -m "slow and benchmark" --benchmark-only --benchmark-json=reports/benchmarks/benchmark-large.json
uv run python scripts/merge_benchmark_results.py reports/benchmarks/benchmark.json reports/benchmarks/benchmark-large.json --output reports/benchmarks/benchmark.json
```

### Advanced Options

For advanced users who want to use pytest-benchmark's built-in features:

**Compare with previous results:**
```bash
uv run pytest tests/test_benchmark.py -m benchmark --benchmark-only --benchmark-compare
```

**Save baseline for future comparisons:**
```bash
uv run pytest tests/test_benchmark.py -m benchmark --benchmark-only --benchmark-save=baseline
```

**Compare against a saved baseline:**
```bash
uv run pytest tests/test_benchmark.py -m benchmark --benchmark-only --benchmark-compare=baseline
```

## Benchmark Reports

After running benchmarks, reports are generated in the `reports/benchmarks/` directory:

### Report Locations

- **Combined JSON Report**: `reports/benchmarks/benchmark.json`
  - Contains detailed statistics for the latest merged benchmark run
  - Includes min, max, mean, median, stddev, IQR, and outliers

- **Large-only JSON Report**: `reports/benchmarks/benchmark-large.json`
  - Output from the latest slow benchmark run before merging

- **Historical Data**: `reports/benchmarks/Linux-CPython-3.12-64bit/*.json`
  - Individual JSON files for each benchmark run
  - Organized by commit hash and timestamp

- **Plotly charts**:
  - Interactive HTML dashboard: `reports/benchmarks/benchmark_dashboard.html`
  - SVG per benchmark group: `reports/benchmarks/plots/*.svg`

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
- Tests with 50M+ points
- Run with `just benchmark-large`

## Configuration

### Benchmark-Specific Options

Benchmark options are configured in `pytest.ini` and overridden in the Justfile commands to keep regular test runs unaffected. Each benchmark command includes:

```bash
--benchmark-autosave          # Automatically save benchmark data
--benchmark-storage=reports/benchmarks  # Where to store benchmark data
--benchmark-json=reports/benchmarks/benchmark.json  # JSON report location (overridden for large runs)
```

The `reports/benchmarks` directory is automatically created when running any benchmark command.

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
**Solution**: Use `just benchmark` for the fast suite and run `just benchmark-large` separately for slow cases.

### Issue: Reports directory not found
**Solution**: The directory is automatically created on first run. It's in `.gitignore` and won't be committed.

### Issue: Plot export fails
**Solution**: Ensure Plotly export dependencies are installed:
```bash
uv sync --dev
```

## Dependencies

- `pytest-benchmark[histogram]>=5.2.3` - Benchmarking framework
- `plotly>=6.5.2` and `kaleido>=1.2.0` - Plot generation and SVG export
