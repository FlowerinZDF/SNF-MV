"""Smoke test for first runnable global Weibo baseline."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from src.trainers.train_global import run_training


class TestGlobalBaseline(unittest.TestCase):
    def test_run_training_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"

            train_records = [
                {"id": "1", "text": "fake rumor spread quickly", "image_path": "a.jpg", "overall_label": 1},
                {"id": "2", "text": "official report confirms truth", "image_path": "b.jpg", "overall_label": 0},
                {"id": "3", "text": "fabricated story with no source", "image_path": "c.jpg", "overall_label": 1},
                {"id": "4", "text": "verified announcement by agency", "image_path": "d.jpg", "overall_label": 0},
            ]
            eval_records = [
                {"id": "5", "text": "rumor account posted fake claim", "image_path": "e.jpg", "overall_label": 1},
                {"id": "6", "text": "trusted media released facts", "image_path": "f.jpg", "overall_label": 0},
            ]

            train_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_records) + "\n", encoding="utf-8")
            eval_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in eval_records) + "\n", encoding="utf-8")

            args = Namespace(
                train_jsonl=str(train_path),
                eval_jsonl=str(eval_path),
                output_dir=str(root / "out"),
                epochs=1,
                batch_size=2,
                lr=1e-3,
                weight_decay=1e-4,
                embed_dim=32,
                hidden_dim=32,
                vocab_size=5000,
                dropout=0.1,
                eval_ratio=0.2,
                seed=123,
                device="cpu",
            )

            outputs = run_training(args)
            self.assertIn("best_metrics", outputs)
            self.assertTrue(Path(outputs["checkpoint"]).exists())
            self.assertTrue(Path(outputs["metrics_json"]).exists())


if __name__ == "__main__":
    unittest.main()
