"""Shared multimodal input handling for SNF-MV trainers."""

from __future__ import annotations

import inspect
from typing import Any

import torch

_IMAGE_KEYS = ("image_tensors", "image_tensor", "images", "image")


def _pick_image_values(batch: dict[str, Any], preferred_key: str | None = None) -> Any:
    if preferred_key:
        return batch.get(preferred_key)
    for key in _IMAGE_KEYS:
        if key in batch:
            return batch[key]
    return None


def _to_batched_image_tensors(values: Any, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Convert collated image values into a batch tensor + validity mask.

    Returns ``(None, None)`` when no image tensor is available.
    """
    if values is None:
        return None, None

    if isinstance(values, torch.Tensor):
        if values.ndim < 2:
            return None, None
        mask = torch.ones(values.shape[0], dtype=torch.bool, device=device)
        return values.to(device), mask

    if not isinstance(values, list) or not values:
        return None, None

    first_tensor = next((v for v in values if isinstance(v, torch.Tensor)), None)
    if first_tensor is None:
        return None, None

    template = torch.zeros_like(first_tensor)
    stacked: list[torch.Tensor] = []
    mask_values: list[bool] = []

    for value in values:
        if isinstance(value, torch.Tensor):
            stacked.append(value.to(device))
            mask_values.append(True)
        else:
            stacked.append(template.clone().to(device))
            mask_values.append(False)

    image_tensors = torch.stack(stacked, dim=0)
    image_mask = torch.tensor(mask_values, dtype=torch.bool, device=device)
    return image_tensors, image_mask


def build_model_inputs(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device,
    image_key: str | None = None,
    enable_images: bool = True,
) -> dict[str, Any]:
    """Build model kwargs and keep only parameters supported by the model."""
    inputs: dict[str, Any] = {"texts": [str(text or "") for text in batch["text"]]}

    if enable_images:
        raw_images = _pick_image_values(batch, preferred_key=image_key)
        image_tensors, image_mask = _to_batched_image_tensors(raw_images, device)
        if image_tensors is not None:
            inputs["image_tensors"] = image_tensors
            inputs["image_mask"] = image_mask

    signature = inspect.signature(model.forward)
    accepted = set(signature.parameters.keys())
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return inputs

    return {k: v for k, v in inputs.items() if k in accepted}
