"""Tests for optional weak per-view supervision in view-stage trainer."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.trainers.train_view import _compute_aux_view_loss, _prepare_optional_view_targets


class TestViewAuxSupervision(unittest.TestCase):
    def test_prepare_optional_view_targets_handles_missing_and_null(self) -> None:
        batch = {
            "overall_label": [1, 0, 1],
            "subject_label": [1, None, "0"],
            "event_label": [None, "bad", 1],
            # scene_label intentionally missing from batch.
            "time_label": [0, None, None],
        }

        targets = _prepare_optional_view_targets(batch, device=torch.device("cpu"))

        subject_labels, subject_mask = targets["subject"]
        self.assertEqual(subject_labels.tolist(), [1, 0, 0])
        self.assertEqual(subject_mask.tolist(), [True, False, True])

        event_labels, event_mask = targets["event"]
        self.assertEqual(event_labels.tolist(), [0, 0, 1])
        self.assertEqual(event_mask.tolist(), [False, False, True])

        scene_labels, scene_mask = targets["scene"]
        self.assertEqual(scene_labels.tolist(), [0, 0, 0])
        self.assertEqual(scene_mask.tolist(), [False, False, False])

        time_labels, time_mask = targets["time"]
        self.assertEqual(time_labels.tolist(), [0, 0, 0])
        self.assertEqual(time_mask.tolist(), [True, False, False])

    def test_compute_aux_view_loss_uses_only_available_labels(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        outputs = {
            "logits": logits,
            "per_view_logits": {
                "subject": logits.clone(),
                "event": logits.clone(),
                "scene": logits.clone(),
                "time": logits.clone(),
            },
        }
        targets = {
            "subject": (
                torch.tensor([0, 1], dtype=torch.long),
                torch.tensor([True, False], dtype=torch.bool),
            ),
            "event": (
                torch.tensor([1, 1], dtype=torch.long),
                torch.tensor([False, True], dtype=torch.bool),
            ),
            "scene": (
                torch.tensor([0, 0], dtype=torch.long),
                torch.tensor([False, False], dtype=torch.bool),
            ),
            "time": (
                torch.tensor([0, 0], dtype=torch.long),
                torch.tensor([False, False], dtype=torch.bool),
            ),
        }
        criterion = nn.CrossEntropyLoss()

        aux_loss = _compute_aux_view_loss(outputs, targets, criterion=criterion)
        expected = criterion(torch.tensor([[2.0, 0.0]]), torch.tensor([0]))
        self.assertAlmostEqual(float(aux_loss.item()), float(expected.item()), places=6)

    def test_compute_aux_view_loss_is_zero_without_any_view_labels(self) -> None:
        outputs = {
            "logits": torch.randn(3, 2),
            "per_view_logits": {k: torch.randn(3, 2) for k in ("subject", "event", "scene", "time")},
        }
        targets = {
            k: (
                torch.zeros(3, dtype=torch.long),
                torch.zeros(3, dtype=torch.bool),
            )
            for k in ("subject", "event", "scene", "time")
        }
        criterion = nn.CrossEntropyLoss()

        aux_loss = _compute_aux_view_loss(outputs, targets, criterion=criterion)
        self.assertEqual(float(aux_loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
