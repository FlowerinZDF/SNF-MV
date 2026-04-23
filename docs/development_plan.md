# SNF-MV Development Plan

## Phase 1: Bootstrap

- Initialize repository scaffold and minimal module placeholders.
- Establish canonical data schema and basic import smoke tests.

## Phase 2: Weibo Data Conversion

- Build Weibo raw-to-canonical JSONL conversion pipeline.
- Add split generation (`train/val/test`) and lightweight validation checks.
- Current conversion assumptions follow legacy MViR layout:
  - `raw_root/tweets/{train,test}_{rumor,nonrumor}.txt`
  - tab-delimited rows where id/text/images/label are parsed from columns `0/1/4/-1`
  - images searched under `raw_root/rumor_images` and `raw_root/nonrumor_images`

## Phase 3: Global Baseline

- Implement a minimal global encoder/classifier baseline.
- Add reproducible training/evaluation scripts with seed control.

## Phase 4: View-Aware Model

- Introduce separate view extraction modules.
- Fuse multi-view representations for baseline improvements.

## Phase 5: Consistency Reasoning

- Add cross-view consistency reasoning components.
- Study conflict alignment between view predictions and evidence.

## Phase 6: TIFS-Style Evaluation

- Reproduce TIFS-style evaluation protocol for fair comparison.
- Report performance, ablations, and robustness analyses.
