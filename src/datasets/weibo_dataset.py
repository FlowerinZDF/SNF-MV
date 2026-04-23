"""Minimal Weibo dataset placeholder for SNF-MV."""

from typing import Any, Dict, List


class WeiboDataset:
    """A lightweight dataset container for future Weibo experiments.

    TODO:
    - Add JSONL loading and validation.
    - Convert to torch.utils.data.Dataset when training pipeline is added.
    """

    def __init__(self, samples: List[Dict[str, Any]] | None = None) -> None:
        self.samples = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]
