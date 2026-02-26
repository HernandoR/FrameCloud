"""Merge multiple pytest-benchmark JSON reports into a single file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_benchmark(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with path.open() as f:
        return json.load(f)


def _entry_key(entry: dict[str, Any]) -> str:
    return entry.get("fullname") or entry.get("name") or ""


def merge_benchmarks(
    primary: dict[str, Any], secondary: dict[str, Any]
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(primary)

    combined_entries: dict[str, dict[str, Any]] = {}
    for entry in primary.get("benchmarks", []):
        combined_entries[_entry_key(entry)] = entry

    for entry in secondary.get("benchmarks", []):
        combined_entries[_entry_key(entry)] = entry

    merged["benchmarks"] = sorted(
        combined_entries.values(),
        key=lambda e: (e.get("group", ""), e.get("name", "")),
    )

    merged.setdefault("machine_info", primary.get("machine_info"))
    merged.setdefault("commit_info", primary.get("commit_info"))
    merged.setdefault("version", primary.get("version"))
    merged.setdefault("datetime", primary.get("datetime"))

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge pytest-benchmark JSON reports.")
    parser.add_argument(
        "primary",
        type=Path,
        help="Path to the primary benchmark JSON (e.g., regular benchmarks).",
    )
    parser.add_argument(
        "secondary",
        type=Path,
        help="Path to the secondary benchmark JSON (e.g., large benchmarks).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for merged report (defaults to overwriting primary).",
    )

    args = parser.parse_args()
    for path in [args.primary, args.secondary]:
        if not path.exists():
            print(f"Error: Benchmark file not found: {path}")
            return

    primary = _load_benchmark(args.primary)
    secondary = _load_benchmark(args.secondary)
    merged = merge_benchmarks(primary, secondary)

    output_path = args.output or args.primary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(merged, f, indent=2)

    print(
        f"Merged {len(primary.get('benchmarks', []))} + "
        f"{len(secondary.get('benchmarks', []))} benchmarks "
        f"into {output_path}"
    )


if __name__ == "__main__":
    main()
