"""First runnable full SNF-MV model with lightweight consistency reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .consistency_reasoner import ConsistencyReasoner
from .view_model import ViewModel


@dataclass(frozen=True)
class FullModelConfig:
    input_dim: int = 128
    hidden_dim: int = 128
    view_dim: int = 64
    consistency_dim: int = 64
    num_classes: int = 2
    vocab_size: int = 50000
    dropout: float = 0.1


class FullModel(nn.Module):
    """View-aware fake/real model with explicit cross-view consistency reasoning.

    Design goals:
    - keep the current ViewModel pipeline intact
    - add lightweight pairwise reasoning (no heavy GNN)
    - expose debug-friendly outputs for future supervision upgrades
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        *,
        view_dim: int = 64,
        consistency_dim: int = 64,
        vocab_size: int = 50000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = FullModelConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            view_dim=view_dim,
            consistency_dim=consistency_dim,
            num_classes=num_classes,
            vocab_size=vocab_size,
            dropout=dropout,
        )

        self.view_model = ViewModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            view_dim=view_dim,
            vocab_size=vocab_size,
            dropout=dropout,
        )
        self.reasoner = ConsistencyReasoner(
            view_dim=view_dim,
            pair_hidden_dim=view_dim,
            consistency_dim=consistency_dim,
            dropout=dropout,
        )

        # Final decision: combine view-aware logits + consistency signals.
        fusion_dim = num_classes + consistency_dim + 4
        self.final_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor | None = None,
        *,
        texts: Sequence[str] | None = None,
        images: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        view_outputs = self.view_model(features=features, texts=texts, images=images)
        reasoner_outputs = self.reasoner(view_outputs["view_features"])

        fusion_input = torch.cat(
            [
                view_outputs["logits"],
                reasoner_outputs["consistency_representation"],
                reasoner_outputs["pairwise_score_vector"],
            ],
            dim=-1,
        )
        logits = self.final_head(fusion_input)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        return {
            # Main prediction output for training/evaluation.
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            # Keep stage-1 view-aware outputs visible for easy debugging.
            "view_logits": view_outputs["logits"],
            "view_probabilities": view_outputs["probabilities"],
            "per_view_logits": view_outputs["per_view_logits"],
            "per_view_probabilities": view_outputs["per_view_probabilities"],
            "global_logits": view_outputs["global_logits"],
            "global_probabilities": view_outputs["global_probabilities"],
            "view_features": view_outputs["view_features"],
            "shared_representation": view_outputs["shared_representation"],
            # Explicit consistency outputs.
            "pairwise_logits": reasoner_outputs["pairwise_logits"],
            "pairwise_scores": reasoner_outputs["pairwise_scores"],
            "pairwise_score_vector": reasoner_outputs["pairwise_score_vector"],
            "consistency_representation": reasoner_outputs["consistency_representation"],
        }
