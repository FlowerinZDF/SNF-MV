"""View-aware model placeholder for SNF-MV."""

import torch
from torch import nn


class ViewModel(nn.Module):
    """A minimal view-specific model.

    TODO:
    - Replace with dedicated encoders per view.
    - Support missing-view handling.
    """

    def __init__(self, input_dim: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(view_features)
