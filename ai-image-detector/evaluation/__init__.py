"""Evaluation module for AI image detection."""

from .evaluate import compute_per_generator_metrics
from .robustness_eval import (
    evaluate_robustness,
    print_robustness_report,
    compute_robustness_degradation,
    RobustnessPerturbation
)

__all__ = [
    'compute_per_generator_metrics',
    'evaluate_robustness',
    'print_robustness_report',
    'compute_robustness_degradation',
    'RobustnessPerturbation'
]
