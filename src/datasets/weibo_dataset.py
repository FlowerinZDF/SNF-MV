"""Weibo dataset loader for SNF-MV JSONL structural data.

This module intentionally keeps dependencies minimal so that it can be imported
in early project stages without requiring the full training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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

    Expected record shape:
    - required: id, text, image_path, overall_label
    - optional: structural labels/conflicts/priors (filled with ``None`` if absent)

    Image loading is optional. When enabled, ``__getitem__`` returns an extra
    ``image_tensor`` field and a boolean ``image_loaded`` marker.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        load_images: bool = False,
        image_root: str | Path | None = None,
        image_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.samples: list[dict[str, Any]] = [self._normalize_record(r) for r in records]
        self.load_images = load_images
        self.image_root = Path(image_root) if image_root is not None else None
        self.image_transform = image_transform

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str | Path,
        *,
        load_images: bool = False,
        image_root: str | Path | None = None,
        image_transform: Callable[[Any], Any] | None = None,
    ) -> "WeiboStructuralDataset":
        """Build a dataset from a JSONL file on disk."""
        return cls(
            read_jsonl(jsonl_path),
            load_images=load_images,
            image_root=image_root,
            image_transform=image_transform,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.samples[index])
        if not self.load_images:
            return sample

        image_tensor = self._safe_load_image_tensor(sample.get("image_path"))
        sample["image_tensor"] = image_tensor
        sample["image_loaded"] = image_tensor is not None
        return sample

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

    def _resolve_image_path(self, image_path: str | None) -> Path | None:
        if not image_path:
            return None
        path = Path(image_path)
        if path.is_absolute():
            return path
        if self.image_root is not None:
            return self.image_root / path
        return path

    def _safe_load_image_tensor(self, image_path: str | None) -> Any | None:
        resolved = self._resolve_image_path(image_path)
        if resolved is None or not resolved.exists():
            return None

        try:
            from PIL import Image
            import numpy as np
            import torch

            image = Image.open(resolved).convert("RGB")
            if self.image_transform is not None:
                transformed = self.image_transform(image)
                if isinstance(transformed, torch.Tensor):
                    return transformed
                if isinstance(transformed, np.ndarray):
                    if transformed.ndim == 3:
                        transformed = transformed.transpose(2, 0, 1)
                    return torch.from_numpy(transformed).float()
                return transformed

            arr = np.array(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            return tensor
        except Exception:
            # Keep loading robust: downstream can still train text-only.
            return None


# Backward-compatible alias for earlier scaffold imports.
WeiboDataset = WeiboStructuralDataset
