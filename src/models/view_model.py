"""First runnable view-aware model stage for SNF-MV.

This module upgrades the global-only baseline into explicit semantic view
prediction without adding consistency reasoning yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .global_model import GlobalModel
from .view_extractor import VIEW_NAMES, ViewExtractor


@dataclass(frozen=True)
class ViewModelConfig:
    input_dim: int = 128
    hidden_dim: int = 128
    view_dim: int = 64
    num_classes: int = 2
    vocab_size: int = 50000
    dropout: float = 0.1


class ViewModel(nn.Module):
    """View-aware fake/real model with explicit semantic decomposition.

    Current training path can focus on overall labels while this model exposes
    per-view outputs so view supervision can be added incrementally.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        *,
        view_dim: int = 64,
        vocab_size: int = 50000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = ViewModelConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            view_dim=view_dim,
            num_classes=num_classes,
            vocab_size=vocab_size,
            dropout=dropout,
        )

        # Reuse baseline text path and overall classifier design.
        self.global_model = GlobalModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            vocab_size=vocab_size,
            dropout=dropout,
        )
        self.view_extractor = ViewExtractor(input_dim=input_dim, view_dim=view_dim, dropout=dropout)

        self.view_classifiers = nn.ModuleDict(
            {view: nn.Linear(view_dim, num_classes) for view in VIEW_NAMES}
        )
        self.overall_head = nn.Sequential(
            nn.Linear(input_dim + len(VIEW_NAMES) * view_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _encode_inputs(
        self,
        features: torch.Tensor | None,
        texts: Sequence[str] | None,
    ) -> torch.Tensor:
        if features is None and texts is None:
            raise ValueError("Either features or texts must be provided.")

        if features is not None:
            if features.ndim != 2:
                raise ValueError(f"Expected [B, D] features, got shape {tuple(features.shape)}")
            return features

        return self.global_model.encode_texts(texts or [])

    def forward(
        self,
        features: torch.Tensor | None = None,
        *,
        texts: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        base_repr = self._encode_inputs(features=features, texts=texts)
        global_outputs = self.global_model(features=base_repr)

        extracted = self.view_extractor(base_repr)
        view_features = extracted["view_features"]

        per_view_logits: dict[str, torch.Tensor] = {}
        per_view_probabilities: dict[str, torch.Tensor] = {}
        for view in VIEW_NAMES:
            logits = self.view_classifiers[view](view_features[view])
            per_view_logits[view] = logits
            per_view_probabilities[view] = torch.softmax(logits, dim=-1)

        fused_view = torch.cat([view_features[view] for view in VIEW_NAMES], dim=-1)
        overall_input = torch.cat([base_repr, fused_view], dim=-1)
        logits = self.overall_head(overall_input)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "per_view_logits": per_view_logits,
            "per_view_probabilities": per_view_probabilities,
            # TODO: add optional per-view supervision loss terms once labels are reliable.
            "global_logits": global_outputs["logits"],
            "global_probabilities": global_outputs["probabilities"],
            "view_features": view_features,
            "stacked_view_features": extracted["stacked_view_features"],
        }
