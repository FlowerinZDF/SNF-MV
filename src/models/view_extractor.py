"""Lightweight semantic view extractor for the first SNF-MV view-aware stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

VIEW_NAMES = ("subject", "event", "scene", "time")


@dataclass(frozen=True)
class ViewExtractorConfig:
    input_dim: int = 128
    view_dim: int = 64
    dropout: float = 0.1


class ViewExtractor(nn.Module):
    """Decompose a shared representation into explicit semantic views.

    Design goals for the first stage:
    - keep architecture lightweight/readable
    - avoid heavy backbone assumptions
    - support both `[B, D]` and `[B, T, D]` inputs

    For each view, we learn:
    - a gate over the shared representation (sigmoid)
    - a small projection into a view-specific representation space

    If token-level input `[B, T, D]` is provided, an attention-like pooling step
    uses learnable view queries to first obtain `[B, D]` per view.
    """

    def __init__(self, input_dim: int = 128, view_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.config = ViewExtractorConfig(input_dim=input_dim, view_dim=view_dim, dropout=dropout)

        self.view_gates = nn.ModuleDict(
            {view: nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid()) for view in VIEW_NAMES}
        )
        self.view_projections = nn.ModuleDict(
            {
                view: nn.Sequential(
                    nn.Linear(input_dim, view_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for view in VIEW_NAMES
            }
        )

        # Used only when inputs are token-level `[B, T, D]`.
        self.view_queries = nn.ParameterDict(
            {view: nn.Parameter(torch.randn(input_dim)) for view in VIEW_NAMES}
        )

    def _pool_if_needed(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim == 2:
            return {view: features for view in VIEW_NAMES}

        if features.ndim != 3:
            raise ValueError(
                f"ViewExtractor expects features with shape [B, D] or [B, T, D], got {tuple(features.shape)}"
            )

        pooled_by_view: dict[str, torch.Tensor] = {}
        # Attention-like pooling with a per-view learnable query.
        for view in VIEW_NAMES:
            query = self.view_queries[view]
            scores = torch.einsum("btd,d->bt", features, query)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.einsum("bt,btd->bd", weights, features)
            pooled_by_view[view] = pooled
        return pooled_by_view

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return explicit per-view representations and stacked tensor output."""
        pooled_inputs = self._pool_if_needed(features)

        view_features: dict[str, torch.Tensor] = {}
        for view in VIEW_NAMES:
            shared = pooled_inputs[view]
            gated = shared * self.view_gates[view](shared)
            view_features[view] = self.view_projections[view](gated)

        stacked = torch.stack([view_features[view] for view in VIEW_NAMES], dim=1)
        return {
            "view_features": view_features,
            "stacked_view_features": stacked,
        }
