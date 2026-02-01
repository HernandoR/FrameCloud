# FrameCloud Test Suite

## Overview

Complete test suite for the FrameCloud project built with pytest framework. The test suite provides comprehensive coverage of the core `PointCloud` class and `PointCloudIO` I/O operations.

## Test Coverage

- **Total Tests**: 51
- **Overall Coverage**: 94%
- **Core Module Coverage**: 90% (`framecloud/np/core.py`)
- **I/O Module Coverage**: 96% (`framecloud/np/pintcloud_io.py`)

## Test Files

### 1. `tests/test_point_cloud.py`

Tests for the `PointCloud` class with 30 test cases organized in 7 test classes:

#### TestPointCloudInitialization (6 tests)

- Basic point cloud creation
- Point cloud with attributes
- Input validation (shape, dimensions)
- Attribute length validation
- Empty point cloud handling

#### TestPointCloudProperties (3 tests)

- `num_points` property
- `attribute_names` property
- Empty attributes handling

#### TestPointCloudAttributeOperations (8 tests)

- Adding attributes
- Setting attributes (new and existing)
- Removing attributes
- Getting attributes
- Duplicate attribute detection
- Attribute validation

#### TestPointCloudTransformation (5 tests)

- Translation transformation
- Scale transformation
- In-place transformation
- Attribute preservation during transformation
- Invalid matrix detection

#### TestPointCloudCopy (2 tests)

- Copy creates new instance
- Deep copy verification

#### TestPointCloudSampling (4 tests)

- Sampling without replacement
- Sampling with replacement
- Over-sampling detection
- Attribute preservation in sampling

#### TestPointCloudToDict (1 test)

- Dictionary conversion

### 2. `tests/test_point_cloud_io.py`

Tests for the `PointCloudIO` class with 21 test cases organized in 6 test classes:

#### TestPointCloudIOLAS (2 tests)

- LAS file round-trip (load/save)
- LAS with attributes

#### TestPointCloudIOParquet (3 tests)

- Parquet file round-trip
- Parquet with attributes
- Custom position column names

#### TestPointCloudIOBinary (4 tests)

- Binary buffer serialization
- Binary buffer with attributes
- Binary file I/O
- Invalid buffer handling

#### TestPointCloudIONumPy (2 tests)

- NumPy .npy file I/O
- NumPy files with attributes

#### TestPointCloudIONPZ (3 tests)

- NumPy .npz file round-trip
- NPZ with attributes
- Missing attribute error handling

#### TestPointCloudIOGeneric (7 tests)

- File type inference (LAS, Parquet, NPY, NPZ)
- Explicit file type specification
- Unsupported file type error handling
- Generic from_file and to_file methods

## Running the Tests

### Run all tests

```bash
uv run pytest tests/
```

### Run with verbose output

```bash
uv run pytest tests/ -v
```

### Run specific test file

```bash
uv run pytest tests/test_point_cloud.py -v
uv run pytest tests/test_point_cloud_io.py -v
```

### Run specific test class

```bash
uv run pytest tests/test_point_cloud.py::TestPointCloudInitialization -v
```

### Run specific test

```bash
uv run pytest tests/test_point_cloud.py::TestPointCloudInitialization::test_create_basic_point_cloud -v
```

### Generate coverage report

```bash
uv run pytest tests/ --cov=framecloud --cov-report=term-missing
uv run pytest tests/ --cov=framecloud --cov-report=html
```

Coverage reports are generated in:

- Terminal: `--cov-report=term-missing`
- HTML: `reports/coverage/html/index.html`
- XML: `reports/coverage/coverage.xml`

## Test Features

### Input Validation

- Shape validation for point arrays (must be Nx3)
- Attribute length validation
- File format validation

### Data Integrity

- Deep copy verification
- Attribute preservation across operations
- Numerical precision testing (decimal places)

### File Format Support

- LAS/LAZ format (via laspy)
- Parquet format (via polars)
- NumPy .npy format
- NumPy .npz format
- Binary buffers and files

### Error Handling

- Invalid input detection
- Unsupported format handling
- Missing attribute detection
- Over-sampling detection

## Dependencies

The test suite uses:

- `pytest>=9.0.2` - Test framework
- `pytest-cov>=7.0.0` - Coverage reporting
- `numpy` - Numerical operations
- `laspy[lazrs]` - LAS/LAZ I/O
- `polars[pydantic,openpyxl,sqlalchemy,fsspec]` - Parquet I/O
- `loguru` - Logging

## Configuration

Tests are configured in `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v -s -rA --durations=0 --durations-min=1 --cov=framecloud
```

## Notes

- Tests use temporary directories for file I/O operations
- Logging output is captured and displayed during test execution
- Float comparisons use `assert_array_almost_equal` for numerical tolerance
- Tests are isolated and can run in any order
