"""
Report Generator Module for S&P500 Prediction Project

This module provides a high-level interface to the report generation functionality.
It uses the modular implementation from the reporting package.
"""

import logging
from typing import Dict, Optional

from .reporting import ReportGenerator

# Set up logging
logger = logging.getLogger(__name__)

def generate_model_report(
    model_name: str,
    metrics: Dict,
    data_info: Dict,
    output_file: Optional[str] = None,
    config: Optional[Dict] = None,
    **kwargs
) -> str:
    """
    Generate a comprehensive PDF report for model performance.
    
    Parameters:
    -----------
    model_name : str
        Name of the model being reported on
    metrics : Dict
        Dictionary with model performance metrics
    data_info : Dict
        Dictionary with data preparation information
    output_file : str, optional
        Path to save the PDF report
    config : Dict, optional
        Configuration dictionary with options for report generation
    **kwargs : Dict
        Additional arguments to pass to ReportGenerator.generate_performance_report
        
    Returns:
    --------
    str
        Path to the generated PDF report
    """
    # Initialize report generator
    report_gen = ReportGenerator(config)
    
    # Generate report
    return report_gen.generate_performance_report(
        model_name=model_name,
        metrics=metrics,
        data_info=data_info,
        output_file=output_file,
        **kwargs
    )

def generate_report_from_evaluation(
    model_name: str,
    eval_results: Dict,
    data_info: Dict,
    output_file: Optional[str] = None,
    config: Optional[Dict] = None,
    project_info: Optional[Dict] = None
) -> str:
    """
    Generate a report from model evaluation results.
    
    Parameters:
    -----------
    model_name : str
        Name of the model being reported on
    eval_results : Dict
        Dictionary with evaluation results from ModelEvaluator
    data_info : Dict
        Dictionary with data preparation information
    output_file : str, optional
        Path to save the PDF report
    config : Dict, optional
        Configuration dictionary with options for report generation
    project_info : Dict, optional
        Dictionary with project information for the title page
        
    Returns:
    --------
    str
        Path to the generated PDF report
    """
    # Initialize report generator
    report_gen = ReportGenerator(config)
    
    # Generate report
    return report_gen.generate_report_from_evaluation(
        model_name=model_name,
        eval_results=eval_results,
        data_info=data_info,
        output_file=output_file,
        project_info=project_info
    )
