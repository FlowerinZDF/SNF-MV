"""Evaluation metric helpers for SNF-MV."""

from __future__ import annotations


def accuracy(num_correct: int, num_total: int) -> float:
    """Compute simple accuracy with zero-safe behavior."""
    if num_total <= 0:
        return 0.0
    return float(num_correct) / float(num_total)


def binary_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    """Compute core binary metrics without external dependencies.

    Label convention:
    - ``1``: fake
    - ``0``: real
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")

    total = len(y_true)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision_fake": 0.0,
            "recall_fake": 0.0,
            "f1_fake": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "num_samples": 0,
        }

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": (tp + tn) / total,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num_samples": total,
    }
