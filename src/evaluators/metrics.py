"""Evaluation metric helpers for SNF-MV placeholders."""


def accuracy(num_correct: int, num_total: int) -> float:
    """Compute simple accuracy with zero-safe behavior.

    TODO:
    - Add precision/recall/F1 and robustness metrics.
    """

    if num_total <= 0:
        return 0.0
    return float(num_correct) / float(num_total)
