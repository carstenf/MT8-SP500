
"""
Explanation module for S&P500 Prediction Project.

This module provides tools for model explainability and transparency.
"""

from .model_explainer import (
    ModelExplainer,
    generate_shap_values,
    create_shap_summary_plot,
    create_partial_dependence_plots,
    create_ice_plots,
    create_force_plot,
    analyze_shap_interactions,
    create_explanation_for_stock,
    analyze_correct_vs_incorrect,
    generate_explanation_report
)

__all__ = [
    'ModelExplainer',
    'generate_shap_values',
    'create_shap_summary_plot',
    'create_partial_dependence_plots',
    'create_ice_plots',
    'create_force_plot',
    'analyze_shap_interactions',
    'create_explanation_for_stock',
    'analyze_correct_vs_incorrect',
    'generate_explanation_report'
]