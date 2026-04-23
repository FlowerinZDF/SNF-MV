# SNF-MV Canonical Data Schema (JSONL)

This document defines the canonical JSONL sample format for SNF-MV experiments.
Each line in a dataset file must be one JSON object.

## Top-level fields

- `sample_id` (string): globally unique sample identifier.
- `dataset` (string): dataset source, e.g., `"weibo"`.
- `split` (string): one of `"train"`, `"val"`, `"test"`.
- `label` (integer): binary label (`0` for real, `1` for fake).
- `language` (string): language tag, e.g., `"zh"`.
- `timestamp` (string, optional): ISO-8601 timestamp if available.
- `content` (object): text and metadata payload.
- `views` (object): multi-view features (raw or precomputed placeholders).
- `provenance` (object, optional): crawl/source traceability fields.

## `content` object

- `title` (string)
- `body` (string)
- `tokens` (array of string, optional)
- `entities` (array of string, optional)
- `hashtags` (array of string, optional)

## `views` object

- `semantic` (object, optional): semantic-view input fields.
- `style` (object, optional): writing-style view fields.
- `propagation` (object, optional): social diffusion view fields.
- `evidence` (object, optional): retrieved evidence view fields.

## Example JSONL line

```json
{
  "sample_id": "weibo_000001",
  "dataset": "weibo",
  "split": "train",
  "label": 1,
  "language": "zh",
  "timestamp": "2024-04-16T08:15:00Z",
  "content": {
    "title": "示例标题",
    "body": "示例正文内容",
    "tokens": ["示例", "正文"],
    "entities": ["实体A"],
    "hashtags": ["#示例#"]
  },
  "views": {
    "semantic": {"text": "示例正文内容"},
    "style": {"punct_ratio": 0.12},
    "propagation": {"repost_count": 18},
    "evidence": {"urls": []}
  },
  "provenance": {
    "source_platform": "weibo",
    "source_url": "https://weibo.com/..."
  }
}
```

## Notes

- Keep optional fields present as empty values where feasible for stable parsing.
- Additional view-specific fields can be appended without breaking compatibility.
