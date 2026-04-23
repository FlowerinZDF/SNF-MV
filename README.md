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
```

## Notes

- This scaffold is intentionally minimal for incremental experimentation.
- TODOs in source files mark planned future implementation points.
