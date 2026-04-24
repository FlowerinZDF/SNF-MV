# SNF-MV

SNF-MV (**S**tructural **N**ews **F**orensics with **M**ulti-**V**iew decomposition and reasoning) is a lightweight research codebase scaffold.

## Current Scope

This repository currently provides only the **initial project structure** and placeholder modules.

- First target dataset: **Weibo**.
- Legacy code is preserved under `src/legacy/`.
- No full training pipeline or backbone implementations are included yet.

## Repository Layout

- `configs/`: configuration placeholders for data/model/train
- `data/`: raw, processed, and split data directories
- `docs/`: data schema and phased development plan
- `scripts/`: utility scripts (e.g., smoke tests)
- `src/`: research code modules and legacy stubs
- `tests/`: lightweight tests for import safety
- `outputs/`: logs, checkpoints, predictions, and figures

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
python -m unittest tests/test_smoke_imports.py

# Optional: split processed Weibo train into train/val
python scripts/split_weibo_train_val.py --val-ratio 0.1 --seed 42
```

## Weak Per-View Label Attachment

Use `scripts/attach_view_labels.py` to merge weak per-view labels into an existing dataset JSONL (for example `train.jsonl`).

External label file format (JSONL):
- one object per line
- required key: `id`
- supported optional keys: `subject_label`, `event_label`, `scene_label`, `time_label`

Example label row:

```json
{"id": "weibo_0001", "subject_label": 1, "event_label": 0, "scene_label": null, "time_label": 1}
```

Example command:

```bash
python scripts/attach_view_labels.py \
  --input-jsonl data/processed/weibo/train.jsonl \
  --label-jsonl data/processed/weibo/train_view_labels.jsonl \
  --output-jsonl data/processed/weibo/train.with_view_labels.jsonl
```

## Notes

- This scaffold is intentionally minimal for incremental experimentation.
- TODOs in source files mark planned future implementation points.
