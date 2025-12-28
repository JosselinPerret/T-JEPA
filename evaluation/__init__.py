"""Evaluation module for Tabular-JEPA."""

from .linear_probe import (
    LinearProbe,
    LinearProbeEvaluator,
    evaluate_linear_probe,
)

__all__ = [
    'LinearProbe',
    'LinearProbeEvaluator',
    'evaluate_linear_probe',
]
