"""Random seed utilities for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for Python, NumPy, and PyTorch.

    TODO:
    - Add deterministic backend switches when needed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
