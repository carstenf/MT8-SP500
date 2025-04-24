"""
Visualization Package

This package provides tools for creating visualizations and plots,
using both Yellowbrick and custom matplotlib implementations.
"""

from .plotters import (
    create_time_series_plot,
    create_correlation_matrix_plot,
    save_plot,
    get_yellowbrick_visualizer
)

__all__ = [
    # Direct plotting functions
    'create_time_series_plot',
    'create_correlation_matrix_plot',
    'save_plot',
    
    # Yellowbrick integration
    'get_yellowbrick_visualizer'
]
