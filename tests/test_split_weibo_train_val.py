"""Tests for deterministic Weibo train/val split script."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.split_weibo_train_val import run
from src.utils.io import read_jsonl


class TestSplitWeiboTrainVal(unittest.TestCase):
    def test_split_is_deterministic_and_preserves_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_train = root / "train.jsonl"
            output_train = root / "train_out.jsonl"
            output_val = root / "val_out.jsonl"

            records = [
                {"id": f"id-{i}", "text": f"text-{i}", "extra": i, "overall_label": i % 2}
                for i in range(10)
            ]
            input_train.write_text(
                "".join(f"{json.dumps(record)}\n" for record in records),
                encoding="utf-8",
            )

            train_count_1, val_count_1 = run(
                input_train=input_train,
                output_train=output_train,
                output_val=output_val,
                val_ratio=0.2,
                seed=7,
            )
            first_train = read_jsonl(output_train)
            first_val = read_jsonl(output_val)

            train_count_2, val_count_2 = run(
                input_train=input_train,
                output_train=output_train,
                output_val=output_val,
                val_ratio=0.2,
                seed=7,
            )
            second_train = read_jsonl(output_train)
            second_val = read_jsonl(output_val)

            self.assertEqual((train_count_1, val_count_1), (8, 2))
            self.assertEqual((train_count_2, val_count_2), (8, 2))
            self.assertEqual(first_train, second_train)
            self.assertEqual(first_val, second_val)

            all_output_ids = {record["id"] for record in first_train + first_val}
            input_ids = {record["id"] for record in records}
            self.assertEqual(all_output_ids, input_ids)

            sample = first_train[0]
            self.assertIn("extra", sample)


if __name__ == "__main__":
    unittest.main()
