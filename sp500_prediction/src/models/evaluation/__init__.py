"""
Model Evaluation Package

This package provides a comprehensive set of tools for model evaluation,
including metrics calculation, analysis, visualization, and reporting.

The main interface is the ModelEvaluator class, which coordinates all evaluation
components. Individual components can also be used directly if needed.
"""

from .metrics_calculator import calculate_performance_metrics
from .analysis import (
    analyze_predictions_by_ticker,
    analyze_predictions_by_time
)
from .visualization import (
    create_performance_visualizations,
    create_feature_importance_plot,
    create_time_series_performance_plot
)
from .reporting import (
    generate_performance_report,
    convert_to_serializable
)
from .evaluator import ModelEvaluator

__all__ = [
    # Main interface
    'ModelEvaluator',
    
    # Individual components
    'calculate_performance_metrics',
    'analyze_predictions_by_ticker',
    'analyze_predictions_by_time',
    'create_performance_visualizations',
    'create_feature_importance_plot',
    'create_time_series_performance_plot',
    'generate_performance_report',
    'convert_to_serializable'
]
