#!/usr/bin/env python3
"""Convert legacy Weibo raw files into SNF-MV canonical JSONL.

Raw-layout assumptions (based on legacy `src/legacy/mvir/data_process_weibo.py`):
1) Input path contains `tweets/` with files named:
   - train_rumor.txt, train_nonrumor.txt, test_rumor.txt, test_nonrumor.txt
2) Each file has a header row and tab-delimited rows where:
   - column 0 is post id
   - column 1 is post text
   - column 4 is image identifier(s), potentially comma-separated
   - last column is textual label (`fake` or `real`)
3) Image files may exist under `rumor_images/` and `nonrumor_images/` as
   `<image_id>.<ext>`.

If your raw files differ, adapt parsing in `_parse_legacy_line`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

EXPECTED_FILES = (
    "train_rumor.txt",
    "train_nonrumor.txt",
    "test_rumor.txt",
    "test_nonrumor.txt",
)

URL_PATTERN = re.compile(r"https?://\S+")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Path containing legacy Weibo raw files (tweets/, rumor_images/, nonrumor_images/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path.",
    )
    return parser.parse_args()


def _clean_text(text: str) -> str:
    return URL_PATTERN.sub("", text).strip()


def _label_to_int(label: str) -> int:
    lowered = label.strip().lower()
    if lowered == "fake":
        return 1
    if lowered == "real":
        return 0
    raise ValueError(f"Unsupported label: {label}")


def _resolve_image_path(raw_root: Path, image_id_field: str) -> str:
    image_candidates = [img_id.strip() for img_id in image_id_field.split(",") if img_id.strip()]
    if not image_candidates:
        return ""

    search_dirs = [raw_root / "rumor_images", raw_root / "nonrumor_images"]
    for image_id in image_candidates:
        for directory in search_dirs:
            if not directory.exists():
                continue
            # try any extension and also bare filename (some dumps contain extensions already)
            direct = directory / image_id
            if direct.exists():
                return str(direct)
            matches = list(directory.glob(f"{image_id}.*"))
            if matches:
                return str(matches[0])

    # Keep the first id as a placeholder when the image file is not found.
    return image_candidates[0]


def _parse_legacy_line(line: str, raw_root: Path) -> dict[str, object]:
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 5:
        raise ValueError(f"Unexpected line format (<5 tab columns): {line!r}")

    post_id = fields[0].strip()
    text = _clean_text(fields[1])
    image_field = fields[4].strip()
    label = _label_to_int(fields[-1])

    return {
        "id": post_id,
        "text": text,
        "image_path": _resolve_image_path(raw_root, image_field),
        "overall_label": label,
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


def iter_legacy_records(raw_root: Path) -> Iterable[dict[str, object]]:
    tweets_dir = raw_root / "tweets"
    for name in EXPECTED_FILES:
        file_path = tweets_dir / name
        if not file_path.exists():
            # TODO: If a dataset variant omits one split/class file, make this configurable.
            continue

        with file_path.open("r", encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                if line_index == 0:
                    continue
                if not line.strip():
                    continue
                yield _parse_legacy_line(line, raw_root)


def convert(raw_root: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_legacy_records(raw_root):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    num_records = convert(args.raw_root, args.output)
    print(f"Wrote {num_records} records to {args.output}")


if __name__ == "__main__":
    main()
