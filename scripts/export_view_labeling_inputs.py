#!/usr/bin/env python3
"""Export a clean JSONL file for external weak view label generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPORT_FIELDS = ("id", "text", "image_path", "overall_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Input dataset JSONL path.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output JSONL path with only labeling input fields.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        missing = [field for field in EXPORT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Row missing required fields {missing}: {row}")

        exported.append({field: row[field] for field in EXPORT_FIELDS})
    return exported


def run(input_jsonl: Path, output_jsonl: Path) -> int:
    rows = _read_jsonl(input_jsonl)
    exported = export_rows(rows)
    _write_jsonl(output_jsonl, exported)
    return len(exported)


def main() -> None:
    args = parse_args()
    count = run(input_jsonl=args.input_jsonl, output_jsonl=args.output_jsonl)
    print(f"Wrote {count} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
