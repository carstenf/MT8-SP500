"""
Model Explanation Components Package

This package contains the individual components used by the ModelExplainer:

- shap_explainer: SHAP value calculation and visualization
- dependence_plots: Partial dependence and ICE plots
- interaction_analyzer: Feature interaction analysis
- error_analyzer: Analysis of model errors
- case_studies: Stock-specific explanations and case studies
"""

from .shap_explainer import generate_shap_values, create_shap_summary_plot, create_force_plot
from .dependence_plots import create_partial_dependence_plots, create_ice_plots
from .interaction_analyzer import analyze_shap_interactions, create_interaction_report
from .error_analyzer import analyze_correct_vs_incorrect, create_error_analysis_report, create_example_explanations
from .case_studies import create_explanation_for_stock, create_case_study_report, analyze_common_factors

__all__ = [
    # SHAP components
    'generate_shap_values',
    'create_shap_summary_plot',
    'create_force_plot',
    
    # Dependence plots
    'create_partial_dependence_plots',
    'create_ice_plots',
    
    # Interaction analysis
    'analyze_shap_interactions',
    'create_interaction_report',
    
    # Error analysis
    'analyze_correct_vs_incorrect',
    'create_error_analysis_report',
    'create_example_explanations',
    
    # Case studies
    'create_explanation_for_stock',
    'create_case_study_report',
    'analyze_common_factors'
]
