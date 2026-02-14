# Custom Plotly Benchmark Visualization System

**Date:** 2026-02-13
**Author:** Claude Code
**Purpose:** Documentation for the custom benchmark plotting system using Plotly

## Overview

This document describes the custom benchmark visualization system that replaces pytest-benchmark's built-in histogram generation. The new system uses Plotly to create interactive, professional visualizations that are:

- **Interactive**: HTML dashboards with hover information and zoom capabilities
- **Customizable**: All styling parameters can be easily modified
- **Automatic**: Discovers test groups from JSON without code changes
- **Scalable**: Handles any number of test groups and implementations

## Architecture

### Components

1. **pytest-benchmark**: Runs tests and generates JSON output (`reports/benchmarks/benchmark.json`)
2. **scripts/benchmark_plot.py**: Python script that reads JSON and generates plots
3. **Justfile integration**: `just benchmark` command runs tests and plotting automatically

### Data Flow

```
Benchmark Tests → JSON Output → Plotting Script → Visualizations
(pytest)        (benchmark.json)  (Plotly)        (SVG + HTML)
```

## Features

### 1. Automatic Test Grouping

The plotting system automatically groups tests by their `group` attribute in pytest-benchmark:

```python
@pytest.mark.benchmark(group="pcl-creation")
def test_create_pointcloud(...):
    ...
```

Each group gets:
- An individual SVG file (e.g., `pcl-creation.svg`)
- A section in the comprehensive HTML dashboard

### 2. Consistent Color Scheme

Different implementations are assigned consistent colors across all plots:

- NumPy (`np`): Blue (#1f77b4)
- Pandas (`pd`): Orange (#ff7f0e)
- Default: Green (#2ca02c) for any new implementations

This makes it easy to compare performance across different implementations at a glance.

### 3. Horizontal Bar Charts

Test case labels are displayed horizontally on the Y-axis, which:
- Allows long test names to be fully visible
- Improves readability for multiple tests
- Makes comparisons easier

### 4. Interactive Features

The HTML dashboard includes:
- Hover tooltips with detailed statistics (mean, median, min, max, stddev, OPS, rounds)
- Zoom and pan capabilities
- Responsive layout that adapts to screen size

### 5. Configurable Styling

All styling parameters are centralized in the `PlotConfig` dataclass:

```python
@dataclass
class PlotConfig:
    # Color palette for different implementations
    colors: dict[str, str] = field(
        default_factory=lambda: {
            "np": "#1f77b4",
            "pd": "#ff7f0e",
            "default": "#2ca02c",
        }
    )

    # Chart dimensions
    height_per_test: int = 30
    min_height: int = 400
    width: int = 1200

    # Fonts and margins...
```

## Usage

### Running Benchmarks with Plotting

```bash
# Run all benchmarks and generate plots
just benchmark

# View the generated reports
just benchmark-view
```

### Manual Plotting

If you already have a `benchmark.json` file:

```bash
# Use default paths
uv run python scripts/benchmark_plot.py

# Or specify custom paths
uv run python scripts/benchmark_plot.py path/to/benchmark.json path/to/output/dir
```

### Customizing Plot Appearance

To customize colors, fonts, or layout:

1. Edit `scripts/benchmark_plot.py`
2. Modify the `PlotConfig` dataclass defaults
3. No other code changes needed!

Example - changing NumPy color to red:

```python
@dataclass
class PlotConfig:
    colors: dict[str, str] = field(
        default_factory=lambda: {
            "np": "#ff0000",  # Changed to red
            "pd": "#ff7f0e",
            "default": "#2ca02c",
        }
    )
```

## Output Files

After running benchmarks, the following files are generated:

### 1. JSON Data
- **Location**: `reports/benchmarks/benchmark.json`
- **Purpose**: Raw benchmark data in pytest-benchmark format
- **Content**: Detailed statistics, machine info, commit info

### 2. Individual Group SVGs
- **Location**: `reports/benchmarks/plots/<group-name>.svg`
- **Purpose**: Static visualizations for each benchmark group
- **Examples**: `reports/benchmarks/plots/pcl-creation.svg`, `reports/benchmarks/plots/pcl-io.svg`, `reports/benchmarks/plots/pcl-transformation.svg`
- **Use case**: Embedding in documentation, presentations

### 3. HTML Dashboard
- **Location**: `reports/benchmarks/benchmark_dashboard.html`
- **Purpose**: Comprehensive interactive view of all benchmarks
- **Features**: All groups in subplots, interactive tooltips, zoom/pan
- **Use case**: Detailed analysis, sharing with team

### 4. Dashboard SVG (optional)
- **Location**: `reports/benchmarks/benchmark_dashboard.svg`
- **Purpose**: Static version of dashboard (only created for single-group benchmarks)
- **Use case**: Quick preview without opening HTML

## Implementation Details

### Data Structure

Each benchmark result is parsed into a `BenchmarkResult` dataclass:

```python
@dataclass
class BenchmarkResult:
    name: str           # Full test name
    group: str          # Benchmark group
    impl: str           # Implementation (np/pd)
    size: str           # Data size (100K, 1M, etc.)
    mean: float         # Mean execution time
    stddev: float       # Standard deviation
    min_time: float     # Minimum time
    max_time: float     # Maximum time
    median: float       # Median time
    ops: float          # Operations per second
    rounds: int         # Number of rounds
```

### Test Sorting

Tests within each group are sorted by:
1. Data size (ascending: 100K → 1M → 10M → 100M)
2. Implementation name (alphabetically: np → pd)

This ensures consistent ordering across all plots.

### Size Parsing

The system automatically parses size suffixes:
- K → thousands (e.g., "100K" = 100,000)
- M → millions (e.g., "10M" = 10,000,000)
- B → billions (e.g., "1B" = 1,000,000,000)

This allows proper numerical sorting regardless of how sizes are specified.

## Design Decisions

### Why Replace pytest-benchmark's Built-in Plotting?

The built-in histogram generation has limitations:
- Static SVG histograms only
- Limited customization
- No comprehensive dashboard
- Difficult to compare across groups

### Why Plotly?

Plotly was chosen because it:
- Generates both SVG and interactive HTML
- Has excellent documentation and API
- Supports complex layouts (subplots, custom styling)
- Creates professional-looking visualizations out of the box
- Has good Python support with type hints

### Why Separate Script Instead of pytest Plugin?

A standalone script provides:
- Easier customization (no pytest plugin API to learn)
- Can be run independently of tests
- Simpler maintenance
- Better IDE support for development

## Extending the System

### Adding New Implementations

To add support for a new implementation (e.g., "polars"):

1. Add the implementation to fixtures in `tests/conftest.py`
2. Add a color to `PlotConfig`:
   ```python
   colors: dict[str, str] = field(
       default_factory=lambda: {
           "np": "#1f77b4",
           "pd": "#ff7f0e",
           "polars": "#9467bd",  # New!
           "default": "#2ca02c",
       }
   )
   ```

That's it! The plotting system will automatically include the new implementation.

### Adding New Test Groups

Simply use the `@pytest.mark.benchmark(group="new-group")` decorator on your test:

```python
@pytest.mark.benchmark(group="pcl-filtering")
def test_filter_pointcloud(...):
    ...
```

The plotting system will automatically:
- Create `pcl-filtering.svg`
- Add a section to the dashboard
- Apply consistent styling

### Customizing Plot Types

To change from horizontal bars to another chart type:

1. Modify the `create_group_plot` method in `BenchmarkPlotter`
2. Replace `go.Bar` with another Plotly graph object (e.g., `go.Scatter`)
3. Adjust layout parameters as needed

Example - box plots instead of bars:

```python
fig.add_trace(
    go.Box(
        y=test_labels,
        x=mean_times,
        orientation="h",
        marker=dict(color=colors[0]),
        # ... other parameters
    )
)
```

## Troubleshooting

### No Plots Generated

**Problem**: Running the script produces no output files.

**Solution**: Ensure `benchmark.json` exists and contains data:
```bash
ls -lh reports/benchmarks/benchmark.json
cat reports/benchmarks/benchmark.json | jq '.benchmarks | length'
```

### SVG Export Fails

**Problem**: HTML dashboard is created but SVG files fail.

**Solution**: Ensure kaleido is installed:
```bash
uv add --dev kaleido
```

### Colors Not Consistent

**Problem**: Different plots show different colors for same implementation.

**Solution**: Check that `PlotConfig.colors` dictionary has entries for all implementations used in tests.

### Dashboard Too Large

**Problem**: HTML file is very large (>10MB).

**Solution**: This is normal for dashboards with many data points. The Plotly HTML includes all JavaScript libraries. For smaller files, use the SVG outputs.

## Future Enhancements

Possible improvements for the plotting system:

1. **Comparison Mode**: Overlay multiple benchmark runs to show performance trends
2. **Filtering**: Command-line options to plot only specific groups or implementations
3. **Statistics Panel**: Add summary statistics panel above plots
4. **Export Options**: Support for additional formats (PNG, PDF)
5. **Theme Support**: Light/dark theme toggle
6. **Performance Regression Detection**: Automatic highlighting of performance regressions

## References

- [Plotly Python Documentation](https://plotly.com/python/)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- Project benchmarks: `tests/test_benchmark.py`, `tests/test_benchmark_voxelmap.py`
