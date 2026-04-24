"""Smoke tests for lightweight multimodal upgrade."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.datasets.weibo_dataset import WeiboStructuralDataset
from src.models.full_model import FullModel
from src.models.global_model import GlobalModel
from src.models.view_model import ViewModel


class TestMultimodalUpgrade(unittest.TestCase):
    def test_global_forward_with_text_and_image(self) -> None:
        model = GlobalModel(input_dim=32, hidden_dim=16, num_classes=2, vocab_size=2000)
        texts = ["rumor text", "verified text"]
        images = torch.rand(2, 3, 32, 32)

        outputs = model(texts=texts, images=images)

        self.assertEqual(tuple(outputs["logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["shared_representation"].shape), (2, 32))

    def test_view_and_full_forward_with_image_only(self) -> None:
        images = torch.rand(2, 3, 32, 32)

        view_model = ViewModel(input_dim=32, hidden_dim=16, view_dim=8, num_classes=2, vocab_size=2000)
        view_outputs = view_model(images=images)
        self.assertEqual(tuple(view_outputs["logits"].shape), (2, 2))

        full_model = FullModel(
            input_dim=32,
            hidden_dim=16,
            view_dim=8,
            consistency_dim=8,
            num_classes=2,
            vocab_size=2000,
        )
        full_outputs = full_model(images=images)
        self.assertEqual(tuple(full_outputs["logits"].shape), (2, 2))

    def test_dataset_optional_image_loading_graceful_failure(self) -> None:
        records = [
            {"id": "1", "text": "hello", "image_path": "missing.jpg", "overall_label": 1},
        ]
        ds = WeiboStructuralDataset(records, load_images=True)
        sample = ds[0]

        self.assertIn("image_tensor", sample)
        self.assertIsNone(sample["image_tensor"])
        self.assertFalse(sample["image_loaded"])

    def test_dataset_optional_image_loading_success(self) -> None:
        try:
            pil = __import__("PIL.Image", fromlist=["Image"])
        except ModuleNotFoundError:
            self.skipTest("Pillow is not installed in this environment.")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_path = root / "ok.png"
            image = pil.new("RGB", (16, 16), color=(12, 45, 78))
            image.save(img_path)

            records = [{"id": "1", "text": "hello", "image_path": str(img_path), "overall_label": 1}]
            ds = WeiboStructuralDataset(records, load_images=True)
            sample = ds[0]

            self.assertTrue(sample["image_loaded"])
            self.assertIsInstance(sample["image_tensor"], torch.Tensor)
            self.assertEqual(tuple(sample["image_tensor"].shape), (3, 16, 16))


if __name__ == "__main__":
    unittest.main()
