"""
Evaluation Package

This package provides core model evaluation functionality,
including metrics calculation and performance evaluation.
"""

from .metrics import calculate_metrics
from .evaluator import YellowbrickEvaluator

__all__ = [
    # Main evaluator class
    'YellowbrickEvaluator',
    
    # Metrics functions
    'calculate_metrics'
]
