"""End-to-end placeholder assembly for SNF-MV models."""

from __future__ import annotations

import torch
from torch import nn

from .consistency_reasoner import ConsistencyReasoner
from .global_model import GlobalModel
from .view_extractor import VIEW_NAMES, ViewExtractor
from .view_model import ViewModel


class FullModel(nn.Module):
    """A minimal composition of global/view/consistency components.

    NOTE: this remains a scaffold and is intentionally lightweight.
    """

    def __init__(self, input_dim: int = 128, view_dim: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.global_model = GlobalModel(input_dim=input_dim, num_classes=num_classes)
        self.view_extractor = ViewExtractor(input_dim=input_dim, view_dim=view_dim)
        self.view_model = ViewModel(input_dim=input_dim, view_dim=view_dim, num_classes=num_classes)
        self.reasoner = ConsistencyReasoner(view_dim=view_dim)
        self.fusion = nn.Linear(num_classes * 2 + 1, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        global_outputs = self.global_model(features=x)
        global_logits = global_outputs["logits"]

        view_outputs = self.view_model(features=x)
        view_logits = view_outputs["logits"]

        extracted = self.view_extractor(x)
        view_features = extracted["view_features"]
        fused_view_feature = torch.stack([view_features[v] for v in VIEW_NAMES], dim=1).mean(dim=1)
        consistency = self.reasoner(fused_view_feature)

        fused = torch.cat([global_logits, view_logits, consistency], dim=-1)
        final_logits = self.fusion(fused)
        return {
            "global_logits": global_logits,
            "global_probabilities": global_outputs["probabilities"],
            "view_logits": view_logits,
            "consistency": consistency,
            "final_logits": final_logits,
        }
