"""Lightweight global fake/real baseline model for SNF-MV.

This module is intentionally practical for the first runnable Weibo baseline:
- text-only classifier (global prediction)
- no view-aware outputs
- minimal dependencies (pure PyTorch)

Legacy MViR influence (conceptual reuse):
- keeps the legacy pattern of text encoder -> classifier head -> softmax probs
- keeps two-logit binary setup compatible with CrossEntropyLoss
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import torch
from torch import nn

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class GlobalModelConfig:
    """Configuration for :class:`GlobalModel`."""

    vocab_size: int = 50000
    embed_dim: int = 128
    hidden_dim: int = 128
    num_classes: int = 2
    dropout: float = 0.1


class GlobalModel(nn.Module):
    """Global fake/real classifier.

    The model supports two practical input paths:
    1) ``texts``: list[str] -> hashed bag-of-words embedding path (default for Weibo).
    2) ``features``: precomputed dense tensors (keeps compatibility with scaffold wiring).

    Forward returns a dictionary with at least ``logits`` and ``probabilities``.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        *,
        vocab_size: int = 50000,
        embed_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        embed_size = embed_dim if embed_dim is not None else input_dim
        self.config = GlobalModelConfig(
            vocab_size=vocab_size,
            embed_dim=embed_size,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Text encoder path (EmbeddingBag keeps preprocessing simple and fast).
        self.text_embedding = nn.EmbeddingBag(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embed_dim,
            mode="mean",
        )

        # Shared classifier head for both text and dense-feature paths.
        self.classifier = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_classes),
        )

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode a batch of texts into dense features via hashed token ids."""
        if not texts:
            raise ValueError("texts must contain at least one sample.")

        token_ids: list[int] = []
        offsets: list[int] = [0]

        for text in texts:
            tokens = _TOKEN_PATTERN.findall((text or "").lower())
            if not tokens:
                tokens = ["<empty>"]

            hashed = [hash(token) % self.config.vocab_size for token in tokens]
            token_ids.extend(hashed)
            offsets.append(offsets[-1] + len(hashed))

        device = self.text_embedding.weight.device
        ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        return self.text_embedding(ids_tensor, offsets_tensor)

    def forward(
        self,
        features: torch.Tensor | None = None,
        *,
        texts: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run forward pass and return logits/probabilities/predictions.

        Args:
            features: Optional dense feature tensor of shape ``[B, D]``.
            texts: Optional raw text batch used when ``features`` is absent.
        """
        if features is None and texts is None:
            raise ValueError("Either features or texts must be provided.")

        if features is not None:
            if features.ndim != 2:
                raise ValueError(f"Expected 2D features tensor, got shape {tuple(features.shape)}")
            encoded = features
        else:
            encoded = self.encode_texts(texts or [])

        logits = self.classifier(encoded)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }
