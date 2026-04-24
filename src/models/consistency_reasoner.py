"""Lightweight cross-view consistency reasoner for SNF-MV stage-1 reasoning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

PAIR_NAMES = (
    "subject_event",
    "subject_scene",
    "event_scene",
    "event_time",
)

PAIR_TO_VIEWS = {
    "subject_event": ("subject", "event"),
    "subject_scene": ("subject", "scene"),
    "event_scene": ("event", "scene"),
    "event_time": ("event", "time"),
}


@dataclass(frozen=True)
class ConsistencyReasonerConfig:
    view_dim: int = 64
    pair_hidden_dim: int = 64
    consistency_dim: int = 64
    dropout: float = 0.1


class ConsistencyReasoner(nn.Module):
    """Explicit pairwise consistency/conflict scorer across semantic views.

    Pair feature construction is intentionally lightweight:
    - concatenation: [a, b]
    - absolute difference: |a - b|
    - elementwise product: a * b

    Each pair is scored by a small MLP to produce a scalar consistency logit.
    A compact consistency representation is also produced for downstream fusion.
    """

    def __init__(
        self,
        view_dim: int = 64,
        *,
        pair_hidden_dim: int = 64,
        consistency_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = ConsistencyReasonerConfig(
            view_dim=view_dim,
            pair_hidden_dim=pair_hidden_dim,
            consistency_dim=consistency_dim,
            dropout=dropout,
        )

        pair_feature_dim = view_dim * 4

        self.pair_encoders = nn.ModuleDict(
            {
                pair: nn.Sequential(
                    nn.Linear(pair_feature_dim, pair_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for pair in PAIR_NAMES
            }
        )
        self.pair_scorers = nn.ModuleDict(
            {
                pair: nn.Sequential(
                    nn.Linear(pair_hidden_dim, pair_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(pair_hidden_dim, 1),
                )
                for pair in PAIR_NAMES
            }
        )

        self.consistency_projector = nn.Sequential(
            nn.Linear(pair_hidden_dim * len(PAIR_NAMES), consistency_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _build_pair_features(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.cat([first, second, torch.abs(first - second), first * second], dim=-1)

    def forward(self, view_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        pairwise_features: dict[str, torch.Tensor] = {}
        pairwise_logits: dict[str, torch.Tensor] = {}
        pairwise_scores: dict[str, torch.Tensor] = {}

        encoded_pairs: list[torch.Tensor] = []
        score_vector: list[torch.Tensor] = []

        for pair in PAIR_NAMES:
            view_a, view_b = PAIR_TO_VIEWS[pair]
            pair_features = self._build_pair_features(view_features[view_a], view_features[view_b])
            encoded = self.pair_encoders[pair](pair_features)
            logits = self.pair_scorers[pair](encoded)
            scores = torch.sigmoid(logits)

            pairwise_features[pair] = encoded
            pairwise_logits[pair] = logits
            pairwise_scores[pair] = scores

            encoded_pairs.append(encoded)
            score_vector.append(scores)

        stacked_pair_features = torch.cat(encoded_pairs, dim=-1)
        pairwise_score_vector = torch.cat(score_vector, dim=-1)
        consistency_representation = self.consistency_projector(stacked_pair_features)

        return {
            "pairwise_features": pairwise_features,
            "pairwise_logits": pairwise_logits,
            "pairwise_scores": pairwise_scores,
            "pairwise_score_vector": pairwise_score_vector,
            "consistency_representation": consistency_representation,
        }
