"""
Model Explanation Package

This package provides comprehensive tools for explaining model predictions,
including SHAP values, partial dependence plots, feature interactions,
and case studies for specific stock predictions.

The main interface is the ModelExplainer class, which coordinates all
explanation components.
"""

from .model_explainer import ModelExplainer
from .components.shap_explainer import generate_shap_values, create_shap_summary_plot
from .components.dependence_plots import create_partial_dependence_plots, create_ice_plots
from .components.interaction_analyzer import analyze_shap_interactions
from .components.error_analyzer import analyze_correct_vs_incorrect
from .components.case_studies import create_explanation_for_stock, create_case_study_report

__all__ = [
    # Main interface
    'ModelExplainer',
    
    # SHAP components
    'generate_shap_values',
    'create_shap_summary_plot',
    
    # Dependence plots
    'create_partial_dependence_plots',
    'create_ice_plots',
    
    # Interaction analysis
    'analyze_shap_interactions',
    
    # Error analysis
    'analyze_correct_vs_incorrect',
    
    # Case studies
    'create_explanation_for_stock',
    'create_case_study_report'
]
