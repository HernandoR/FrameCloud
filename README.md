# Minimal Python Template

A minimal, practical project template for small Python projects and experiments.

## Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv)

## Usage

### Using this Template

1.  **Fork or Copy**: Fork this repository or copy the files to a new directory.
2.  **Rename Directory**: Rename the directory to your desired project name.
3.  **Initialize Project**: Run the following command to initialize the project with `uv`, which will generate a `pyproject.toml` based on your directory name:
    ```bash
    uv init --lib --build-backend hatchling .
    ```
    *Note: If you want to specify a project name different from the directory name, you can pass the `--name` argument (if supported by your uv version) or simply rename the directory first.*

### Installation

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

### Running

- Run the main script or module:
  ```bash
  uv run -m your_package
  ```

  Replace `your_package` with the actual package or script name.

## Testing

- Run tests using `pytest`:
  ```bash
  uv run pytest
  # or
  just test
  ```

### Benchmarking

The project uses `pytest-benchmark` for automated performance benchmarking:

- Run benchmarks:
  ```bash
  just benchmark
  ```

- Run all benchmarks (including slow large-scale tests):
  ```bash
  just benchmark-all
  ```

- Compare with previous results:
  ```bash
  just benchmark-compare
  ```

Benchmark reports are automatically generated in `reports/benchmarks/` including:
- JSON statistics (`benchmark.json`)
- Historical data (organized by commit hash and timestamp)
- SVG histograms for visual analysis

For detailed benchmark usage, see [agc/20260204_01_benchmark_usage.md](agc/20260204_01_benchmark_usage.md)

## Contributing

Contributions are welcome. Open an issue or submit a pull request with a short description of changes.

## License

MIT
