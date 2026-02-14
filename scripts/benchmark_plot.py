"""Custom benchmark plotting module using Plotly.

This module provides functionality to generate interactive visualizations for
pytest-benchmark results. It reads the benchmark JSON output and creates:
1. Individual SVG/HTML plots per test group
2. A comprehensive HTML dashboard showing all benchmark results

The module automatically discovers test groups and configurations from the JSON
file, making it adaptable to new tests without code changes.

Features:
- Automatic test case grouping by benchmark group and implementation
- Consistent color scheme across all plots
- Horizontal test case labels for better readability
- Interactive plots with hover information
- SVG output for individual groups and HTML dashboard for overview
- Configurable color palette and layout parameters
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class PlotConfig:
    """Configuration for plot styling and layout.

    This dataclass centralizes all styling parameters to make customization easy.
    Users can modify these values to change colors, fonts, and layout without
    touching the plotting logic.
    """

    # Color palette for different implementations
    # Map implementation names to colors (extendable for new implementations)
    colors: dict[str, str] = field(
        default_factory=lambda: {
            "np": "#1f77b4",  # Blue
            "pd": "#ff7f0e",  # Orange
            "default": "#2ca02c",  # Green (fallback for unknown implementations)
        }
    )

    # Chart dimensions
    height_per_test: int = 30  # Height allocated per test case in pixels
    min_height: int = 400  # Minimum chart height
    width: int = 1200  # Chart width

    # Font settings
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    title_font_size: int = 16

    # Layout margins
    margin_left: int = 300  # Left margin for test case labels
    margin_right: int = 50
    margin_top: int = 100
    margin_bottom: int = 80

    # Bar chart settings
    bar_height: float = 0.6  # Bar thickness (0-1)

    def get_color(self, impl: str) -> str:
        """Get color for a specific implementation.

        Args:
            impl: Implementation name (e.g., 'np', 'pd')

        Returns:
            Hex color code for the implementation
        """
        return self.colors.get(impl, self.colors["default"])


@dataclass
class BenchmarkResult:
    """Represents a single benchmark test result.

    This structure holds all relevant information from the pytest-benchmark
    JSON output for a single test case, making it easy to process and visualize.
    """

    name: str  # Full test name
    group: str  # Benchmark group (e.g., 'pcl-creation')
    impl: str  # Implementation type (e.g., 'np', 'pd')
    size: str  # Data size (e.g., '100K', '1M')
    mean: float  # Mean execution time in seconds
    stddev: float  # Standard deviation in seconds
    min_time: float  # Minimum execution time
    max_time: float  # Maximum execution time
    median: float  # Median execution time
    ops: float  # Operations per second
    rounds: int  # Number of benchmark rounds

    @classmethod
    def from_json(cls, bench_data: dict[str, Any]) -> "BenchmarkResult":
        """Create BenchmarkResult from pytest-benchmark JSON data.

        Args:
            bench_data: Single benchmark entry from the JSON file

        Returns:
            BenchmarkResult instance with extracted data
        """
        stats = bench_data["stats"]
        params = bench_data.get("params", {})

        # Extract implementation and size from params or name
        impl = params.get("pointcloud_impl", "unknown")
        param_str = bench_data.get("param", "")

        # Parse size from param string (e.g., "np-100K" -> "100K")
        parts = param_str.split("-")
        size = parts[-1] if len(parts) > 1 else "unknown"

        return cls(
            name=bench_data["name"],
            group=bench_data["group"],
            impl=impl,
            size=size,
            mean=stats["mean"],
            stddev=stats["stddev"],
            min_time=stats["min"],
            max_time=stats["max"],
            median=stats["median"],
            ops=stats["ops"],
            rounds=stats["rounds"],
        )


class BenchmarkPlotter:
    """Main class for creating benchmark visualizations.

    This class handles the entire plotting workflow:
    1. Loading and parsing benchmark JSON data
    2. Grouping tests by category and implementation
    3. Creating individual plots per group
    4. Generating a comprehensive dashboard
    """

    def __init__(
        self, json_path: Path, output_dir: Path, config: PlotConfig | None = None
    ):
        """Initialize the plotter with paths and configuration.

        Args:
            json_path: Path to pytest-benchmark JSON output file
            output_dir: Directory where plots will be saved
            config: Optional custom configuration (uses defaults if None)
        """
        self.json_path = json_path
        self.output_dir = output_dir
        self.config = config or PlotConfig()
        self.results: list[BenchmarkResult] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Load and parse benchmark data from JSON file.

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        with open(self.json_path) as f:
            data = json.load(f)

        self.results = [
            BenchmarkResult.from_json(b) for b in data.get("benchmarks", [])
        ]

    def group_results(self) -> dict[str, list[BenchmarkResult]]:
        """Group benchmark results by test group.

        Returns:
            Dictionary mapping group names to lists of results
        """
        groups: dict[str, list[BenchmarkResult]] = {}
        for result in self.results:
            if result.group not in groups:
                groups[result.group] = []
            groups[result.group].append(result)
        return groups

    def create_group_plot(
        self, group_name: str, results: list[BenchmarkResult]
    ) -> go.Figure:
        """Create a horizontal bar chart for a specific benchmark group.

        This method creates an interactive bar chart showing execution times for
        all tests in a group, color-coded by implementation. Tests are grouped by
        size then sorted by implementation for easy comparison.

        Args:
            group_name: Name of the benchmark group
            results: List of benchmark results in this group

        Returns:
            Plotly Figure object
        """
        # Sort results: group by size (smallest first), then by implementation
        # In Plotly horizontal bar charts, first item appears at bottom
        # So we DON'T reverse to get 100K at top
        results_sorted = sorted(
            results, key=lambda r: (self._size_sort_key(r.size), r.impl)
        )

        # Prepare data for plotting
        test_labels = []
        mean_times = []
        colors = []
        hover_texts = []

        for result in results_sorted:
            # Create readable label: size first for grouping, then implementation
            label = f"{result.size}-{result.impl}"
            test_labels.append(label)
            mean_times.append(result.mean * 1000)  # Convert to milliseconds

            # Get color for this implementation
            colors.append(self.config.get_color(result.impl))

            # Create detailed hover text with statistics
            hover_text = (
                f"<b>{label}</b><br>"
                f"Mean: {result.mean * 1000:.2f} ms<br>"
                f"Median: {result.median * 1000:.2f} ms<br>"
                f"Min: {result.min_time * 1000:.2f} ms<br>"
                f"Max: {result.max_time * 1000:.2f} ms<br>"
                f"StdDev: {result.stddev * 1000:.2f} ms<br>"
                f"OPS: {result.ops:.2f}<br>"
                f"Rounds: {result.rounds}"
            )
            hover_texts.append(hover_text)

        # Create horizontal bar chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=test_labels,  # Horizontal orientation
                x=mean_times,
                orientation="h",
                marker=dict(color=colors),
                hovertext=hover_texts,
                hoverinfo="text",
                width=self.config.bar_height,
            )
        )

        # Calculate dynamic height based on number of tests
        chart_height = max(
            self.config.min_height, len(test_labels) * self.config.height_per_test
        )

        # Update layout with styling
        fig.update_layout(
            title=dict(
                text=f"Benchmark: {group_name}",
                font=dict(
                    size=self.config.title_font_size, family=self.config.font_family
                ),
            ),
            xaxis=dict(
                title=dict(
                    text="Mean Time (ms)",
                    font=dict(
                        family=self.config.font_family, size=self.config.font_size
                    ),
                ),
                gridcolor="lightgray",
                showgrid=True,
                tickfont=dict(
                    family=self.config.font_family, size=self.config.font_size
                ),
            ),
            yaxis=dict(
                title=dict(
                    text="Test Case",
                    font=dict(
                        family=self.config.font_family, size=self.config.font_size
                    ),
                ),
                tickfont=dict(
                    family=self.config.font_family, size=self.config.font_size
                ),
                automargin=True,
            ),
            height=chart_height,
            width=self.config.width,
            margin=dict(
                l=self.config.margin_left,
                r=self.config.margin_right,
                t=self.config.margin_top,
                b=self.config.margin_bottom,
            ),
            font=dict(family=self.config.font_family, size=self.config.font_size),
            hovermode="closest",
            plot_bgcolor="white",
            showlegend=False,
        )

        return fig

    def create_dashboard(self, groups: dict[str, list[BenchmarkResult]]) -> go.Figure:
        """Create a comprehensive dashboard with all benchmark groups.

        This method generates a multi-panel dashboard showing all benchmark groups
        in subplots with two columns: mean time and operations per second.

        Args:
            groups: Dictionary of grouped benchmark results

        Returns:
            Plotly Figure object with subplots
        """
        num_groups = len(groups)
        if num_groups == 0:
            return go.Figure()

        # Create subplots: 2 columns (time and OPS) for each group
        fig = make_subplots(
            rows=num_groups,
            cols=2,
            subplot_titles=[
                title
                for name in sorted(groups.keys())
                for title in [f"{name} - Mean Time", f"{name} - Operations/sec"]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            specs=[[{"type": "bar"}, {"type": "bar"}] for _ in range(num_groups)],
        )

        # Add each group as a row with two columns
        for idx, (_group_name, results) in enumerate(sorted(groups.items()), start=1):
            # Sort and prepare data: smallest sizes first (no reverse)
            # In Plotly horizontal bar charts, first item appears at bottom
            results_sorted = sorted(
                results,
                key=lambda r: (self._size_sort_key(r.size), r.impl),
            )

            test_labels = [f"{r.size}-{r.impl}" for r in results_sorted]
            mean_times = [r.mean * 1000 for r in results_sorted]
            ops_values = [r.ops for r in results_sorted]
            colors = [self.config.get_color(r.impl) for r in results_sorted]

            # Create hover texts for time column
            hover_texts_time = [
                f"<b>{r.size}-{r.impl}</b><br>"
                f"Mean: {r.mean * 1000:.2f} ms<br>"
                f"Median: {r.median * 1000:.2f} ms<br>"
                f"StdDev: {r.stddev * 1000:.2f} ms"
                for r in results_sorted
            ]

            # Create hover texts for OPS column
            hover_texts_ops = [
                f"<b>{r.size}-{r.impl}</b><br>"
                f"OPS: {r.ops:.2f}<br>"
                f"Mean: {r.mean * 1000:.2f} ms"
                for r in results_sorted
            ]

            # Add bar trace to first column (time)
            fig.add_trace(
                go.Bar(
                    y=test_labels,
                    x=mean_times,
                    orientation="h",
                    marker=dict(color=colors),
                    hovertext=hover_texts_time,
                    hoverinfo="text",
                    showlegend=False,
                    width=self.config.bar_height,
                ),
                row=idx,
                col=1,
            )

            # Add bar trace to second column (OPS)
            fig.add_trace(
                go.Bar(
                    y=test_labels,
                    x=ops_values,
                    orientation="h",
                    marker=dict(color=colors),
                    hovertext=hover_texts_ops,
                    hoverinfo="text",
                    showlegend=False,
                    width=self.config.bar_height,
                ),
                row=idx,
                col=2,
            )

            # Update axes for time subplot (first column)
            fig.update_xaxes(
                title_text="Mean Time (ms)",
                gridcolor="lightgray",
                showgrid=True,
                row=idx,
                col=1,
            )
            fig.update_yaxes(title_text="Test Case", row=idx, col=1)

            # Update axes for OPS subplot (second column)
            fig.update_xaxes(
                title_text="Operations/sec",
                gridcolor="lightgray",
                showgrid=True,
                row=idx,
                col=2,
            )
            # Match y-axis labels exactly with first column
            fig.update_yaxes(title_text="", row=idx, col=2, showticklabels=True)

        # Calculate total height
        total_tests = sum(len(results) for results in groups.values())
        dashboard_height = max(800, total_tests * self.config.height_per_test + 200)

        # Update overall layout
        fig.update_layout(
            title=dict(
                text="Benchmark Dashboard - All Groups",
                font=dict(size=20, family=self.config.font_family),
            ),
            height=dashboard_height,
            width=self.config.width,
            font=dict(family=self.config.font_family, size=self.config.font_size),
            hovermode="closest",
            plot_bgcolor="white",
        )

        return fig

    def save_plots(self) -> None:
        """Generate and save all plots to the output directory.

        This method creates:
        1. Individual SVG plots for each benchmark group in plots/ subdirectory
        2. A comprehensive HTML dashboard with all results
        """
        if not self.results:
            print("No benchmark results to plot.")
            return

        groups = self.group_results()

        # Create plots subdirectory for SVG files
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Create individual SVG plots per group
        for group_name, results in groups.items():
            fig = self.create_group_plot(group_name, results)
            output_path = plots_dir / f"{group_name}.svg"
            fig.write_image(str(output_path))
            print(f"✓ Created: {output_path}")

        # Create comprehensive dashboard
        dashboard_fig = self.create_dashboard(groups)
        dashboard_path = self.output_dir / "benchmark_dashboard.html"
        dashboard_fig.write_html(str(dashboard_path))
        print(f"✓ Created: {dashboard_path}")

        # Also create an SVG version of dashboard if there's only one group
        if len(groups) == 1:
            dashboard_svg_path = plots_dir / "benchmark_dashboard.svg"
            dashboard_fig.write_image(str(dashboard_svg_path))
            print(f"✓ Created: {dashboard_svg_path}")

    @staticmethod
    def _size_sort_key(size: str) -> int:
        """Convert size string to sortable integer.

        Args:
            size: Size string like '100K', '1M', '10M'

        Returns:
            Integer representation for sorting
        """
        size = size.upper()
        multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}

        for suffix, multiplier in multipliers.items():
            if suffix in size:
                try:
                    num = float(size.replace(suffix, ""))
                    return int(num * multiplier)
                except ValueError:
                    pass

        # If parsing fails, try to convert directly
        try:
            return int(size)
        except ValueError:
            return 0


def main() -> None:
    """Main entry point for the plotting script.

    Usage:
        python benchmark_plot.py [json_path] [output_dir]

    If no arguments provided, uses default paths:
        - JSON: reports/benchmarks/benchmark.json
        - Output: reports/benchmarks/
    """
    # Parse command line arguments
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = Path("reports/benchmarks/benchmark.json")

    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = Path("reports/benchmarks")

    # Validate input file
    if not json_path.exists():
        print(f"Error: Benchmark JSON file not found: {json_path}")
        print("Please run benchmarks first: just benchmark")
        sys.exit(1)

    # Create plotter and generate visualizations
    print(f"Loading benchmark data from: {json_path}")
    plotter = BenchmarkPlotter(json_path, output_dir)
    plotter.load_data()

    print(f"Found {len(plotter.results)} benchmark results")
    print(f"Generating plots in: {output_dir}")

    plotter.save_plots()
    print("\n✓ All plots generated successfully!")


if __name__ == "__main__":
    main()
