"""I/O helpers for SNF-MV placeholders."""

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records from disk.

    TODO:
    - Add schema validation and informative errors.
    """

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
