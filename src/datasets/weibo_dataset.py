"""Weibo dataset loader for SNF-MV JSONL structural data.

This module intentionally keeps dependencies minimal so that it can be imported
in early project stages without requiring the training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl

REQUIRED_FIELDS = ("id", "text", "image_path", "overall_label")
OPTIONAL_STRUCTURAL_FIELDS = (
    "subject_label",
    "event_label",
    "scene_label",
    "time_label",
    "subject_event_conflict",
    "subject_scene_conflict",
    "event_scene_conflict",
    "event_time_conflict",
    "subject_prior",
    "event_prior",
    "scene_prior",
    "time_prior",
)


class WeiboStructuralDataset:
    """Dataset backed by canonical SNF-MV Weibo JSONL records.

    Expected record shape (first iteration):
    - required: id, text, image_path, overall_label
    - optional: structural labels/conflicts/priors (filled with ``None`` if absent)

    Image decoding is intentionally left as a future step; callers currently
    receive ``image_path`` as a string.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.samples: list[dict[str, Any]] = [self._normalize_record(r) for r in records]

    @classmethod
    def from_jsonl(cls, jsonl_path: str | Path) -> "WeiboStructuralDataset":
        """Build a dataset from a JSONL file on disk."""
        return cls(read_jsonl(jsonl_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """Minimal collation that groups values by key into lists."""
        if not batch:
            return {}
        keys = batch[0].keys()
        return {k: [sample.get(k) for sample in batch] for k in keys}

    @staticmethod
    def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(f"Record is missing required field(s): {missing_list}. Record: {record}")

        normalized = dict(record)
        for field in OPTIONAL_STRUCTURAL_FIELDS:
            normalized.setdefault(field, None)

        if not isinstance(normalized["overall_label"], int):
            try:
                normalized["overall_label"] = int(normalized["overall_label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Record has non-integer overall_label: {normalized['overall_label']}"
                ) from exc

        return normalized


# Backward-compatible alias for earlier scaffold imports.
WeiboDataset = WeiboStructuralDataset
