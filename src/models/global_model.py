"""Lightweight global fake/real baseline model for SNF-MV.

This module keeps the first runnable baseline practical while now supporting a
real multimodal path (text + image) without heavy dependencies.
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
    """Global fake/real classifier with lightweight multimodal fusion.

    The model supports three practical input paths:
    1) ``features``: precomputed shared tensor of shape ``[B, D]``.
    2) ``texts``: list[str], encoded via hashed token embedding.
    3) ``images``: image tensor/list, encoded via compact CNN branch.

    When ``features`` is not provided, text and image representations are fused
    via concatenation + projection into a shared representation.
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

        # Compact image encoder path (lightweight CNN).
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.config.embed_dim),
        )

        # Explicit multimodal fusion into a shared representation.
        self.fusion = nn.Sequential(
            nn.Linear(self.config.embed_dim * 2, self.config.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

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

    def encode_images(self, images: torch.Tensor | Sequence[torch.Tensor | None]) -> torch.Tensor:
        """Encode image tensors into dense features.

        Accepts either a batched tensor ``[B, C, H, W]`` or a sequence where
        missing images can be represented by ``None``.
        """
        if isinstance(images, torch.Tensor):
            image_batch = images
        else:
            tensors: list[torch.Tensor] = []
            for item in images:
                if item is None:
                    tensors.append(torch.zeros(3, 32, 32, dtype=torch.float32))
                    continue
                if item.ndim != 3:
                    raise ValueError(f"Expected [C, H, W] image tensor, got shape {tuple(item.shape)}")
                tensors.append(item.float())
            if not tensors:
                raise ValueError("images must contain at least one sample.")
            image_batch = torch.stack(tensors, dim=0)

        if image_batch.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] image tensor, got shape {tuple(image_batch.shape)}")

        image_batch = image_batch.float()
        if image_batch.shape[1] == 1:
            image_batch = image_batch.repeat(1, 3, 1, 1)
        elif image_batch.shape[1] != 3:
            raise ValueError(f"Expected channel size 1 or 3, got {image_batch.shape[1]}")

        if float(image_batch.max()) > 1.0:
            image_batch = image_batch / 255.0

        device = self.text_embedding.weight.device
        image_batch = image_batch.to(device)
        return self.image_encoder(image_batch)

    def encode_multimodal(
        self,
        *,
        texts: Sequence[str] | None = None,
        images: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Build shared representation from available text/image inputs."""
        batch_size: int | None = None

        if texts is not None:
            text_repr = self.encode_texts(texts)
            batch_size = text_repr.shape[0]
        else:
            text_repr = None

        if images is not None:
            image_repr = self.encode_images(images)
            if batch_size is not None and image_repr.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch between text ({batch_size}) and image ({image_repr.shape[0]}) inputs."
                )
            batch_size = image_repr.shape[0]
        else:
            image_repr = None

        if batch_size is None:
            raise ValueError("At least one modality (texts or images) must be provided.")

        device = self.text_embedding.weight.device
        if text_repr is None:
            text_repr = torch.zeros(batch_size, self.config.embed_dim, device=device)
        if image_repr is None:
            image_repr = torch.zeros(batch_size, self.config.embed_dim, device=device)

        fused = torch.cat([text_repr, image_repr], dim=-1)
        return self.fusion(fused)

    def forward(
        self,
        features: torch.Tensor | None = None,
        *,
        texts: Sequence[str] | None = None,
        images: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run forward pass and return logits/probabilities/predictions.

        Args:
            features: Optional precomputed shared tensor of shape ``[B, D]``.
            texts: Optional raw text batch.
            images: Optional image tensor or sequence.
        """
        if features is None and texts is None and images is None:
            raise ValueError("Provide features or at least one modality input (texts/images).")

        if features is not None:
            if features.ndim != 2:
                raise ValueError(f"Expected 2D features tensor, got shape {tuple(features.shape)}")
            shared_representation = features
        else:
            shared_representation = self.encode_multimodal(texts=texts, images=images)

        logits = self.classifier(shared_representation)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "shared_representation": shared_representation,
        }
