"""Simple smoke script for scaffold import checks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import WeiboDataset
from src.models import FullModel
from src.utils.seed import set_seed


def main() -> None:
    """Run import-safe placeholder checks."""
    set_seed(42)
    ds = WeiboDataset(samples=[{"sample_id": "demo"}])
    model = FullModel()
    print(f"Smoke test OK: dataset_size={len(ds)}, model={model.__class__.__name__}")


if __name__ == "__main__":
    main()
