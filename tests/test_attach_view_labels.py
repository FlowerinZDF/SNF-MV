"""Tests for attaching weak per-view labels to JSONL datasets."""

from __future__ import annotations

import unittest

from scripts.attach_view_labels import attach_view_labels


class TestAttachViewLabels(unittest.TestCase):
    def test_attach_view_labels_updates_supported_fields_and_preserves_others(self) -> None:
        dataset_rows = [
            {
                "id": "a1",
                "text": "sample 1",
                "overall_label": 1,
                "subject_label": None,
                "event_label": None,
                "scene_label": None,
                "time_label": None,
                "extra_field": "keep",
            },
            {
                "id": "a2",
                "text": "sample 2",
                "overall_label": 0,
                "subject_label": None,
                "event_label": None,
                "scene_label": None,
                "time_label": None,
            },
        ]
        label_rows = [
            {
                "id": "a1",
                "subject_label": 1,
                "event_label": 0,
                "scene_label": 1,
                "time_label": None,
                "ignored_column": 123,
            }
        ]

        merged_rows, updated_count = attach_view_labels(dataset_rows, label_rows)

        self.assertEqual(updated_count, 1)
        self.assertEqual(merged_rows[0]["subject_label"], 1)
        self.assertEqual(merged_rows[0]["event_label"], 0)
        self.assertEqual(merged_rows[0]["scene_label"], 1)
        self.assertIsNone(merged_rows[0]["time_label"])
        self.assertEqual(merged_rows[0]["extra_field"], "keep")
        self.assertNotIn("ignored_column", merged_rows[0])

        # Record without labels remains unchanged.
        self.assertIsNone(merged_rows[1]["subject_label"])
        self.assertIsNone(merged_rows[1]["event_label"])
        self.assertIsNone(merged_rows[1]["scene_label"])
        self.assertIsNone(merged_rows[1]["time_label"])

    def test_attach_view_labels_rejects_duplicate_label_ids(self) -> None:
        dataset_rows = [{"id": "a1", "subject_label": None}]
        label_rows = [
            {"id": "a1", "subject_label": 1},
            {"id": "a1", "subject_label": 0},
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate id"):
            attach_view_labels(dataset_rows, label_rows)


if __name__ == "__main__":
    unittest.main()
