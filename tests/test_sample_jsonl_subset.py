"""Tests for deterministic JSONL subset sampling script."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.sample_jsonl_subset import run


class TestSampleJsonlSubset(unittest.TestCase):
    def test_sampling_is_deterministic_and_preserves_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "train.jsonl"
            output_path = root / "subset.jsonl"

            input_lines = [
                '{"id": 1, "text": "a", "label": 0}',
                '{"text": "b", "label": 1, "id": 2}',
                '{"id": 3, "label": 0, "text": "c"}',
                '{"id": 4, "text": "d", "label": 1}',
                '{"id": 5, "text": "e", "label": 0}',
            ]
            input_path.write_text("\n".join(input_lines) + "\n", encoding="utf-8")

            count_1 = run(input_path=input_path, output_path=output_path, num_samples=3, seed=11)
            output_lines_1 = output_path.read_text(encoding="utf-8").splitlines()

            count_2 = run(input_path=input_path, output_path=output_path, num_samples=3, seed=11)
            output_lines_2 = output_path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(count_1, 3)
            self.assertEqual(count_2, 3)
            self.assertEqual(output_lines_1, output_lines_2)

            for line in output_lines_1:
                self.assertIn(line, input_lines)


if __name__ == "__main__":
    unittest.main()
