#!/usr/bin/env python3
"""Attach weak per-view labels to an existing JSONL dataset.

External label file format (JSONL):
- One JSON object per line.
- Required field: ``id`` (must match dataset record ``id``).
- Optional fields to merge:
  - ``subject_label``
  - ``event_label``
  - ``scene_label``
  - ``time_label``

Example label JSONL line:
{"id": "weibo_0001", "subject_label": 1, "event_label": 0, "scene_label": null, "time_label": 1}

Merge behavior:
- Existing dataset records are preserved as-is except for supported fields above.
- If a dataset record id appears in the label file, only label fields present in that
  label row are updated on the record.
- Records without a matching label row remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUPPORTED_VIEW_LABEL_FIELDS = (
    "subject_label",
    "event_label",
    "scene_label",
    "time_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Input dataset JSONL path (e.g., train.jsonl).",
    )
    parser.add_argument(
        "--label-jsonl",
        type=Path,
        required=True,
        help="External label JSONL path keyed by id.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output dataset JSONL path with merged view labels.",
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


def _load_label_index(label_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    label_index: dict[str, dict[str, Any]] = {}
    for row in label_rows:
        if "id" not in row:
            raise ValueError(f"Label row missing required 'id' field: {row}")

        record_id = str(row["id"])
        if record_id in label_index:
            raise ValueError(f"Duplicate id in label file: {record_id}")

        label_index[record_id] = {
            field: row[field] for field in SUPPORTED_VIEW_LABEL_FIELDS if field in row
        }
    return label_index


def attach_view_labels(
    dataset_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    label_index = _load_label_index(label_rows)

    merged_rows: list[dict[str, Any]] = []
    updated_count = 0

    for row in dataset_rows:
        merged = dict(row)
        record_id = str(merged.get("id"))
        patch = label_index.get(record_id)
        if patch is not None:
            merged.update(patch)
            updated_count += 1
        merged_rows.append(merged)

    return merged_rows, updated_count


def run(input_jsonl: Path, label_jsonl: Path, output_jsonl: Path) -> tuple[int, int]:
    dataset_rows = _read_jsonl(input_jsonl)
    label_rows = _read_jsonl(label_jsonl)
    merged_rows, updated_count = attach_view_labels(dataset_rows, label_rows)
    _write_jsonl(output_jsonl, merged_rows)
    return len(merged_rows), updated_count


def main() -> None:
    args = parse_args()
    total_count, updated_count = run(
        input_jsonl=args.input_jsonl,
        label_jsonl=args.label_jsonl,
        output_jsonl=args.output_jsonl,
    )
    print(f"Wrote {total_count} records to {args.output_jsonl}")
    print(f"Updated {updated_count} records using labels from {args.label_jsonl}")


if __name__ == "__main__":
    main()
