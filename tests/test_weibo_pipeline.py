"""Tests for Weibo conversion script and JSONL dataset loader."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_weibo_to_jsonl import convert
from src.datasets.weibo_dataset import WeiboStructuralDataset


class TestWeiboPipeline(unittest.TestCase):
    """Validate first-iteration Weibo conversion and loading."""

    def test_convert_and_load_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tweets_dir = root / "tweets"
            tweets_dir.mkdir(parents=True)

            # Build one minimal legacy file with a header + one sample row.
            train_rumor = tweets_dir / "train_rumor.txt"
            train_rumor.write_text(
                "id\ttext\tunused2\tunused3\timages\tlabel\n"
                "123\thello http://example.com world\tu2\tu3\timg123\tfake\n",
                encoding="utf-8",
            )

            # Place image in expected legacy directory so converter can resolve path.
            rumor_images = root / "rumor_images"
            rumor_images.mkdir()
            (rumor_images / "img123.jpg").write_bytes(b"not-an-image-but-exists")

            output_jsonl = root / "out" / "weibo.jsonl"
            written = convert(root, output_jsonl)
            self.assertEqual(written, 1)

            lines = output_jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["id"], "123")
            self.assertEqual(record["overall_label"], 1)
            self.assertEqual(record["text"], "hello  world")

            ds = WeiboStructuralDataset.from_jsonl(output_jsonl)
            self.assertEqual(len(ds), 1)
            sample = ds[0]
            self.assertIn("subject_label", sample)
            self.assertIsNone(sample["subject_label"])

            batch = WeiboStructuralDataset.collate_fn([sample])
            self.assertEqual(batch["id"], ["123"])


if __name__ == "__main__":
    unittest.main()
