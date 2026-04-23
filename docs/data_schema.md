# SNF-MV Canonical Data Schema (JSONL)

This document defines the canonical JSONL sample format for early SNF-MV experiments.
Each line in a dataset file must be one JSON object.

## Current required fields (Weibo v0 pipeline)

- `id` (string): sample identifier from source dataset.
- `text` (string): cleaned post text.
- `image_path` (string): resolved image path (or unresolved image id placeholder if file is missing).
- `overall_label` (integer): binary label (`0` real, `1` fake).

## Supported optional structural fields (can be `null` in v0)

- `subject_label`
- `event_label`
- `scene_label`
- `time_label`
- `subject_event_conflict`
- `subject_scene_conflict`
- `event_scene_conflict`
- `event_time_conflict`
- `subject_prior`
- `event_prior`
- `scene_prior`
- `time_prior`

## Example JSONL line

```json
{
  "id": "123456",
  "text": "示例微博正文",
  "image_path": ".../rumor_images/abc123.jpg",
  "overall_label": 1,
  "subject_label": null,
  "event_label": null,
  "scene_label": null,
  "time_label": null,
  "subject_event_conflict": null,
  "subject_scene_conflict": null,
  "event_scene_conflict": null,
  "event_time_conflict": null,
  "subject_prior": null,
  "event_prior": null,
  "scene_prior": null,
  "time_prior": null
}
```

## Notes

- Keep optional structural fields present as empty values (`null`) for stable parsing.
- Future schema revisions may additionally include nested metadata blocks for provenance and views.
