#!/usr/bin/env python3
"""Convert Weibo raw files into SNF-MV canonical JSONL.

Expected raw layout under ``--raw-root``:
- tweets/train_rumor.txt
- tweets/train_nonrumor.txt
- tweets/test_rumor.txt
- tweets/test_nonrumor.txt
- rumor_images/
- nonrumor_images/

Each tweets file is parsed as repeated 3-line blocks:
1) metadata line (pipe-separated, first field used as record id when available)
2) image URL line (pipe-separated URLs, usually ending with "null")
3) post text line

Filename drives labels/splits:
- *rumor* -> overall_label = 1
- *nonrumor* -> overall_label = 0
- *train* -> split = "train"
- *test* -> split = "test"

Image handling:
- Read image URLs from line 2, pipe-separated.
- Ignore empty values and "null" (case-insensitive).
- Use the first remaining URL only.
- Map basename to:
  - rumor files    -> <raw-root>/rumor_images/<basename>
  - nonrumor files -> <raw-root>/nonrumor_images/<basename>
- If no valid image URL exists, write image_path as null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

EXPECTED_FILES = (
    "train_rumor.txt",
    "train_nonrumor.txt",
    "test_rumor.txt",
    "test_nonrumor.txt",
)

CANONICAL_NULL_FIELDS = {
    "subject_label": None,
    "event_label": None,
    "scene_label": None,
    "time_label": None,
    "subject_event_conflict": None,
    "subject_scene_conflict": None,
    "event_scene_conflict": None,
    "event_time_conflict": None,
    "subject_prior": None,
    "event_prior": None,
    "scene_prior": None,
    "time_prior": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Path containing Weibo raw directories (tweets/, rumor_images/, nonrumor_images/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/weibo"),
        help="Directory for split JSONL outputs (default: data/processed/weibo).",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Optional explicit train JSONL path. Overrides --output-dir/train.jsonl.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=None,
        help="Optional explicit test JSONL path. Overrides --output-dir/test.jsonl.",
    )
    return parser.parse_args()


def _split_from_filename(filename: str) -> str:
    if filename.startswith("train_"):
        return "train"
    if filename.startswith("test_"):
        return "test"
    raise ValueError(f"Cannot infer split from filename: {filename}")


def _overall_label_from_filename(filename: str) -> int:
    if "nonrumor" in filename:
        return 0
    if "rumor" in filename:
        return 1
    raise ValueError(f"Cannot infer label from filename: {filename}")


def _image_dir_from_filename(raw_root: Path, filename: str) -> Path:
    if "nonrumor" in filename:
        return raw_root / "nonrumor_images"
    return raw_root / "rumor_images"


def _basename_from_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    return Path(parsed.path).name


def _parse_image_path(raw_root: Path, filename: str, image_line: str) -> str | None:
    candidates = [part.strip() for part in image_line.split("|")]
    valid_urls = [value for value in candidates if value and value.lower() != "null"]
    if not valid_urls:
        return None

    basename = _basename_from_url(valid_urls[0])
    if not basename:
        return None

    image_path = _image_dir_from_filename(raw_root, filename) / basename
    return str(image_path)


def _record_id_from_metadata(filename: str, block_index: int, metadata_line: str) -> str:
    first_field = metadata_line.split("|", 1)[0].strip()
    if first_field:
        return first_field
    # Fallback keeps ids deterministic if metadata is missing/blank.
    return f"{filename}:{block_index}"


def _build_record(
    raw_root: Path,
    filename: str,
    block_index: int,
    metadata_line: str,
    image_line: str,
    text_line: str,
) -> dict[str, object]:
    record = {
        "id": _record_id_from_metadata(filename, block_index, metadata_line),
        "text": text_line.strip(),
        "image_path": _parse_image_path(raw_root, filename, image_line),
        "overall_label": _overall_label_from_filename(filename),
        "split": _split_from_filename(filename),
    }
    record.update(CANONICAL_NULL_FIELDS)
    return record


def _iter_three_line_blocks(lines: list[str], filename: str) -> Iterator[tuple[int, str, str, str]]:
    total = len(lines)
    full_blocks = total // 3
    remainder = total % 3
    if remainder:
        print(
            f"Warning: {filename} has {remainder} trailing line(s) that do not form a full 3-line sample; ignored."
        )

    for block_index in range(full_blocks):
        start = block_index * 3
        metadata_line = lines[start].rstrip("\n")
        image_line = lines[start + 1].rstrip("\n")
        text_line = lines[start + 2].rstrip("\n")
        yield block_index, metadata_line, image_line, text_line


def iter_records(raw_root: Path) -> Iterator[dict[str, object]]:
    tweets_dir = raw_root / "tweets"

    for filename in EXPECTED_FILES:
        file_path = tweets_dir / filename
        if not file_path.exists():
            print(f"Warning: missing input file {file_path}; skipped.")
            continue

        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        for block_index, metadata_line, image_line, text_line in _iter_three_line_blocks(lines, filename):
            yield _build_record(
                raw_root=raw_root,
                filename=filename,
                block_index=block_index,
                metadata_line=metadata_line,
                image_line=image_line,
                text_line=text_line,
            )


def convert(raw_root: Path, train_output: Path, test_output: Path) -> tuple[int, int]:
    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_count = 0

    with train_output.open("w", encoding="utf-8") as train_f, test_output.open("w", encoding="utf-8") as test_f:
        for record in iter_records(raw_root):
            serialized = json.dumps(record, ensure_ascii=False) + "\n"
            if record["split"] == "train":
                train_f.write(serialized)
                train_count += 1
            else:
                test_f.write(serialized)
                test_count += 1

    return train_count, test_count


def main() -> None:
    args = parse_args()
    train_output = args.train_output or (args.output_dir / "train.jsonl")
    test_output = args.test_output or (args.output_dir / "test.jsonl")

    train_count, test_count = convert(
        raw_root=args.raw_root,
        train_output=train_output,
        test_output=test_output,
    )

    print(f"Wrote {train_count} train records to {train_output}")
    print(f"Wrote {test_count} test records to {test_output}")


if __name__ == "__main__":
    main()
