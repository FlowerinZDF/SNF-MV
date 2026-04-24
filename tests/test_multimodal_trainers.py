"""Trainer integration tests for optional multimodal image tensor path."""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.datasets.weibo_dataset import WeiboStructuralDataset
from src.trainers import train_full, train_global, train_view


class _DummyMMModel(nn.Module):
    """Small model that accepts optional image tensors for integration checks."""

    calls: list[dict[str, object]] = []

    def __init__(self, input_dim: int = 8, num_classes: int = 2, **_: object) -> None:
        super().__init__()
        self.config = SimpleNamespace(input_dim=input_dim, num_classes=num_classes)
        self.head = nn.Linear(1, num_classes)

    def forward(
        self,
        *,
        texts: list[str],
        image_tensors: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self.__class__.calls.append(
            {
                "image_tensors": image_tensors,
                "image_mask": image_mask,
                "batch_size": len(texts),
            }
        )

        text_score = torch.tensor([len(text) for text in texts], dtype=torch.float32).unsqueeze(-1)
        if image_tensors is not None:
            image_score = image_tensors.float().view(image_tensors.shape[0], -1).mean(dim=1, keepdim=True)
            text_score = text_score + image_score

        logits = self.head(text_score)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return {"logits": logits, "probabilities": probs, "predictions": preds}


class TestMultimodalTrainerIntegration(unittest.TestCase):
    def _run_trainer(
        self,
        trainer_module,
        model_attr: str,
        *,
        include_images: bool,
        include_missing_image: bool,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"
            records = [
                {
                    "id": "1",
                    "text": "sample fake news",
                    "image_path": "a.jpg",
                    "overall_label": 1,
                    "image_tensor": torch.tensor([1.0, 2.0, 3.0]) if include_images else None,
                },
                {
                    "id": "2",
                    "text": "sample real news",
                    "image_path": "b.jpg",
                    "overall_label": 0,
                    "image_tensor": (
                        None if include_missing_image else torch.tensor([0.1, 0.2, 0.3])
                    )
                    if include_images
                    else None,
                },
                {
                    "id": "3",
                    "text": "another fake sample",
                    "image_path": "c.jpg",
                    "overall_label": 1,
                    "image_tensor": torch.tensor([0.3, 0.4, 0.5]) if include_images else None,
                },
                {
                    "id": "4",
                    "text": "another real sample",
                    "image_path": "d.jpg",
                    "overall_label": 0,
                    "image_tensor": torch.tensor([0.6, 0.7, 0.8]) if include_images else None,
                },
            ]
            dataset = WeiboStructuralDataset(records)

            def _fake_from_jsonl(_path: str, **_: object) -> WeiboStructuralDataset:
                return dataset

            _DummyMMModel.calls = []
            args = Namespace(
                train_jsonl="unused_train.jsonl",
                eval_jsonl="unused_eval.jsonl",
                output_dir=str(out_dir),
                epochs=1,
                batch_size=2,
                lr=1e-3,
                weight_decay=1e-4,
                embed_dim=8,
                hidden_dim=8,
                view_dim=4,
                consistency_dim=4,
                vocab_size=100,
                dropout=0.0,
                eval_ratio=0.5,
                seed=7,
                device="cpu",
                enable_images=True,
                image_key=None,
            )

            with patch.object(trainer_module, model_attr, _DummyMMModel), patch.object(
                trainer_module.WeiboStructuralDataset,
                "from_jsonl",
                side_effect=_fake_from_jsonl,
            ):
                outputs = trainer_module.run_training(args)

            self.assertTrue(Path(outputs["checkpoint"]).exists())
            self.assertTrue(Path(outputs["metrics_json"]).exists())
            return outputs, list(_DummyMMModel.calls)

    def test_multimodal_path_across_all_trainers(self) -> None:
        trainer_specs = [
            (train_global, "GlobalModel"),
            (train_view, "ViewModel"),
            (train_full, "FullModel"),
        ]

        for trainer_module, model_attr in trainer_specs:
            with self.subTest(trainer=trainer_module.__name__):
                _, calls = self._run_trainer(
                    trainer_module,
                    model_attr,
                    include_images=True,
                    include_missing_image=True,
                )
                self.assertGreater(len(calls), 0)
                calls_with_images = [c for c in calls if c["image_tensors"] is not None]
                self.assertGreater(len(calls_with_images), 0)

                first = calls_with_images[0]
                image_tensors = first["image_tensors"]
                image_mask = first["image_mask"]
                assert isinstance(image_tensors, torch.Tensor)
                assert isinstance(image_mask, torch.Tensor)
                self.assertEqual(image_tensors.shape[0], first["batch_size"])
                self.assertTrue(torch.any(~image_mask).item())

    def test_missing_image_fallback_across_all_trainers(self) -> None:
        trainer_specs = [
            (train_global, "GlobalModel"),
            (train_view, "ViewModel"),
            (train_full, "FullModel"),
        ]

        for trainer_module, model_attr in trainer_specs:
            with self.subTest(trainer=trainer_module.__name__):
                outputs, calls = self._run_trainer(
                    trainer_module,
                    model_attr,
                    include_images=False,
                    include_missing_image=False,
                )
                self.assertIn("best_metrics", outputs)
                self.assertGreater(len(calls), 0)
                self.assertTrue(all(c["image_tensors"] is None for c in calls))


if __name__ == "__main__":
    unittest.main()
