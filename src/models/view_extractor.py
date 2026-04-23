"""Multi-view feature extractor placeholder for SNF-MV."""

import torch
from torch import nn


class ViewExtractor(nn.Module):
    """Extracts placeholder view embeddings.

    TODO:
    - Implement semantic/style/propagation/evidence view extraction.
    """

    def __init__(self, input_dim: int = 128, view_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, view_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
