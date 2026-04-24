#!/usr/bin/env python3
"""Create a deterministic JSONL subset by sampling N records."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

DEFAULT_SEED = 42
DEFAULT_SAMPLE_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of records to sample (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic sampling (default: 42).",
    )
    return parser.parse_args()


def _read_jsonl_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _write_jsonl_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def sample_lines(lines: list[str], *, num_samples: int, seed: int) -> list[str]:
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    if num_samples > len(lines):
        raise ValueError(
            f"num_samples ({num_samples}) cannot exceed number of records ({len(lines)})"
        )

    sampled_indices = set(random.Random(seed).sample(range(len(lines)), k=num_samples))
    return [line for i, line in enumerate(lines) if i in sampled_indices]


def run(input_path: Path, output_path: Path, *, num_samples: int, seed: int) -> int:
    lines = _read_jsonl_lines(input_path)
    sampled_lines = sample_lines(lines, num_samples=num_samples, seed=seed)
    _write_jsonl_lines(output_path, sampled_lines)
    return len(sampled_lines)


def main() -> None:
    args = parse_args()
    count = run(
        input_path=args.input,
        output_path=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
