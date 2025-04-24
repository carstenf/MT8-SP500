"""
Reporting Package

This package provides tools for generating reports from model evaluation results,
including PDF report generation and performance summaries.
"""

from .report_generator import generate_report, generate_pdf_report
from .report_utils import format_metrics, convert_numpy_types, load_image_safely
from .pages import (
    create_title_page,
    create_metrics_page,
    create_charts_page,
    create_time_analysis_page
)

__all__ = [
    # Main report generation
    'generate_report',
    'generate_pdf_report',
    
    # Utility functions
    'format_metrics',
    'convert_numpy_types',
    'load_image_safely',
    
    # Page generation
    'create_title_page',
    'create_metrics_page',
    'create_charts_page',
    'create_time_analysis_page'
]
