"""End-to-end placeholder assembly for SNF-MV models."""

import torch
from torch import nn

from .consistency_reasoner import ConsistencyReasoner
from .global_model import GlobalModel
from .view_extractor import ViewExtractor
from .view_model import ViewModel


class FullModel(nn.Module):
    """A minimal composition of global/view/consistency components.

    This is intentionally lightweight and not a full method implementation.

    TODO:
    - Add configurable fusion logic.
    - Add training objectives and multi-task outputs.
    """

    def __init__(self, input_dim: int = 128, view_dim: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.global_model = GlobalModel(input_dim=input_dim, num_classes=num_classes)
        self.view_extractor = ViewExtractor(input_dim=input_dim, view_dim=view_dim)
        self.view_model = ViewModel(input_dim=view_dim, num_classes=num_classes)
        self.reasoner = ConsistencyReasoner(view_dim=view_dim)
        self.fusion = nn.Linear(num_classes * 2 + 1, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        global_logits = self.global_model(x)
        view_feat = self.view_extractor(x)
        view_logits = self.view_model(view_feat)
        consistency = self.reasoner(view_feat)
        fused = torch.cat([global_logits, view_logits, consistency], dim=-1)
        final_logits = self.fusion(fused)
        return {
            "global_logits": global_logits,
            "view_logits": view_logits,
            "consistency": consistency,
            "final_logits": final_logits,
        }
