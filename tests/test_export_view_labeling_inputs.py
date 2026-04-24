"""Tests for exporting clean view-labeling inputs from JSONL datasets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_view_labeling_inputs import export_rows, run


class TestExportViewLabelingInputs(unittest.TestCase):
    def test_run_exports_only_expected_fields_and_preserves_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"

            rows = [
                {
                    "id": "weibo_1",
                    "text": "中文测试🙂",
                    "image_path": "images/1.jpg",
                    "overall_label": 1,
                    "subject_label": None,
                    "unused": "drop",
                },
                {
                    "id": "weibo_2",
                    "text": "second row",
                    "image_path": "images/2.jpg",
                    "overall_label": 0,
                    "event_label": 1,
                },
            ]
            input_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            count = run(input_jsonl=input_path, output_jsonl=output_path)
            exported = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(count, 2)
            self.assertEqual(
                exported,
                [
                    {
                        "id": "weibo_1",
                        "text": "中文测试🙂",
                        "image_path": "images/1.jpg",
                        "overall_label": 1,
                    },
                    {
                        "id": "weibo_2",
                        "text": "second row",
                        "image_path": "images/2.jpg",
                        "overall_label": 0,
                    },
                ],
            )

    def test_export_rows_raises_when_required_field_missing(self) -> None:
        rows = [{"id": "x1", "text": "t", "image_path": "a.jpg"}]

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            export_rows(rows)


if __name__ == "__main__":
    unittest.main()
