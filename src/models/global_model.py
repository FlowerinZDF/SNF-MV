"""Global model placeholder for SNF-MV."""

import torch
from torch import nn


class GlobalModel(nn.Module):
    """A minimal global branch model.

    TODO:
    - Add backbone encoder.
    - Add configurable pooling and classification head.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
