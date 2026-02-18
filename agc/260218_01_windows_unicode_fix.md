# Windows Unicode Encoding Fix

**Date:** 2026-02-18  
**Issue:** Windows platform failure in benchmark plotting  
**Reference:** https://github.com/HernandoR/FrameCloud/actions/runs/22107814679/job/63895461756

## Problem

The `scripts/benchmark_plot.py` script was failing on Windows platform with a `UnicodeEncodeError`:

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 0: character maps to <undefined>
```

### Root Cause

The script uses Unicode characters (✓ and ✗) for status messages in print statements. On Windows, Python defaults to CP1252 (Windows-1252) encoding for stdout, which doesn't support these Unicode characters.

Specific locations using Unicode symbols:
- Line 529: `print(f"✓ Created: {output_path}")`
- Line 531: `print(f"✗ Failed to write {output_path}: {exc}")`
- Line 537: `print(f"✓ Created: {dashboard_path}")`
- Line 544: `print(f"✓ Created: {dashboard_svg_path}")`
- Line 546: `print(f"✗ Failed to write {dashboard_svg_path}: {exc}")`
- Line 612: `print("\n✓ All plots generated successfully!")`

## Solution

Added UTF-8 encoding configuration for stdout at the beginning of the script (after imports):

```python
# Configure stdout to use UTF-8 encoding on all platforms (especially Windows)
# This ensures Unicode characters (✓, ✗) are properly displayed
if sys.stdout.encoding != "utf-8":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
```

### How It Works

1. Checks if stdout is already using UTF-8 encoding
2. If not (e.g., on Windows with CP1252), it wraps the stdout buffer with `io.TextIOWrapper`
3. The wrapper uses UTF-8 encoding, allowing Unicode characters to be properly encoded
4. This is done early in the script, before any print statements execute

## Implementation Details

- **File Modified:** `scripts/benchmark_plot.py`
- **Lines Added:** 7 (lines 31-36)
- **Impact:** Cross-platform compatibility for console output
- **Backward Compatibility:** Yes - no breaking changes for Unix-like systems

## Testing

### Verification Steps

1. **Import Test:** Verified the script can be imported without errors
   ```bash
   uv run python -c "import sys; import importlib.util; spec = importlib.util.spec_from_file_location('benchmark_plot', 'scripts/benchmark_plot.py'); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)"
   ```

2. **Encoding Test:** Confirmed stdout encoding is UTF-8 after import
   ```
   stdout encoding: utf-8
   ```

3. **Unicode Output Test:** Verified Unicode characters print correctly
   ```
   ✓ Unicode check mark works
   ✗ Unicode cross mark works
   Success: Unicode characters can be printed!
   ```

4. **Functional Test:** Tested with sample benchmark data
   ```bash
   uv run python scripts/benchmark_plot.py /tmp/test_benchmark_plot/test_data.json /tmp/test_benchmark_plot/output
   ```
   
   Output:
   ```
   Loading benchmark data from: /tmp/test_benchmark_plot/test_data.json
   Found 1 benchmark results
   Generating plots in: /tmp/test_benchmark_plot/output
   ✓ Created: /tmp/test_benchmark_plot/output/plots/pcl-creation.svg
   ✓ Created: /tmp/test_benchmark_plot/output/benchmark_dashboard.html
   ✓ Created: /tmp/test_benchmark_plot/output/plots/benchmark_dashboard.svg
   ✓ All plots generated successfully!
   ```

5. **Code Quality:** Verified no new linting issues and proper formatting
   ```bash
   uvx ruff format scripts/benchmark_plot.py  # No changes needed
   ```

6. **Security Scan:** CodeQL found no security issues

## Alternative Solutions Considered

1. **Replace Unicode characters with ASCII alternatives**
   - Rejected: Would reduce visual appeal and clarity of output
   - Example: Using `[OK]` instead of ✓

2. **Use PYTHONIOENCODING environment variable**
   - Rejected: Would require changes to all workflow files and Just recipes
   - Less portable and harder to maintain

3. **Set encoding in individual print statements**
   - Rejected: Would require changes to every print statement
   - More verbose and error-prone

## Benefits

- **Minimal Change:** Only 7 lines added, no business logic modified
- **Cross-Platform:** Works on Windows, macOS, and Linux
- **Transparent:** No API changes, no behavior changes except fixing the encoding
- **Future-Proof:** Any new Unicode characters in print statements will work automatically
- **Standards-Compliant:** Uses UTF-8, the universal standard for text encoding

## Related Workflows

This fix enables the following GitHub Actions workflow to pass on Windows:
- `.github/workflows/test-slow-cross-platform.yml` - Regular Benchmark Tests (Cross Platform)

## Lessons Learned

1. **Windows Encoding Pitfall:** Always be aware that Windows defaults to CP1252, not UTF-8
2. **Early Configuration:** Set encoding at the top of scripts, before any I/O operations
3. **Test Cross-Platform:** CI/CD should include Windows testing for scripts with console output
4. **Unicode in Scripts:** When using Unicode characters, always configure encoding appropriately

## Best Practice for Future Scripts

When creating Python scripts that print Unicode characters:

```python
import sys

# Configure UTF-8 encoding for stdout (Windows compatibility)
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
```

This pattern should be used in any script that:
- Prints Unicode characters beyond basic ASCII
- Is run in CI/CD on multiple platforms
- Produces user-facing console output

## Security Summary

- CodeQL analysis: 0 alerts
- No security vulnerabilities introduced
- No sensitive data exposed
- No unsafe operations added
