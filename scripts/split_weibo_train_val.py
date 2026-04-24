#!/usr/bin/env python3
"""Split Weibo train.jsonl into deterministic train/val JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-train",
        type=Path,
        default=Path("data/processed/weibo/train.jsonl"),
        help="Input train JSONL path (default: data/processed/weibo/train.jsonl).",
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        default=Path("data/processed/weibo/train.jsonl"),
        help="Output train JSONL path (default: data/processed/weibo/train.jsonl).",
    )
    parser.add_argument(
        "--output-val",
        type=Path,
        default=Path("data/processed/weibo/val.jsonl"),
        help="Output val JSONL path (default: data/processed/weibo/val.jsonl).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation ratio in [0, 1] (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic split (default: 42).",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(
    records: list[dict[str, object]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"val_ratio must be in [0, 1], got {val_ratio}")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)

    val_size = int(len(records) * val_ratio)
    val_indices = set(indices[:val_size])

    train_records: list[dict[str, object]] = []
    val_records: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if index in val_indices:
            val_records.append(record)
        else:
            train_records.append(record)

    return train_records, val_records


def run(
    input_train: Path,
    output_train: Path,
    output_val: Path,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[int, int]:
    records = _read_jsonl(input_train)
    train_records, val_records = split_records(records, val_ratio=val_ratio, seed=seed)
    _write_jsonl(output_train, train_records)
    _write_jsonl(output_val, val_records)
    return len(train_records), len(val_records)


def main() -> None:
    args = parse_args()
    train_count, val_count = run(
        input_train=args.input_train,
        output_train=args.output_train,
        output_val=args.output_val,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Wrote {train_count} train records to {args.output_train}")
    print(f"Wrote {val_count} val records to {args.output_val}")


if __name__ == "__main__":
    main()
