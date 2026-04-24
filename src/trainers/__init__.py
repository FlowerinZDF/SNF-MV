"""Trainer package for SNF-MV."""

from .train_global import run_training as run_global_training
from .train_view import run_training as run_view_training

__all__ = ["run_global_training", "run_view_training"]
