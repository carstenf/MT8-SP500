"""
Visualization Utilities Module

This module provides a bridge between Yellowbrick visualizers and custom matplotlib plots.
It avoids duplicating functionality already provided by Yellowbrick while enabling
custom visualizations not available in Yellowbrick.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any

# Import Yellowbrick components
from yellowbrick.classifier import ConfusionMatrix, ROCAUC, PrecisionRecallCurve
from yellowbrick.model_selection import LearningCurve
from yellowbrick.features import FeatureImportances

# Set up logging
logger = logging.getLogger(__name__)

# ----------------------
# Non-Yellowbrick Plots
# ----------------------

def create_time_series_plot(periods: List[str],
                          metrics: Dict[str, List[float]],
                          ax: plt.Axes = None,
                          metric_names: List[str] = None) -> plt.Axes:
    """
    Create a time series plot of performance metrics.
    
    Parameters:
    -----------
    periods : List[str]
        Time period labels
    metrics : Dict[str, List[float]]
        Dictionary with metric names and values
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    metric_names : List[str], optional
        Names of metrics to plot
        
    Returns:
    --------
    plt.Axes
        Plot axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    if not metric_names:
        metric_names = list(metrics.keys())
    
    # Convert string periods to datetime if possible
    try:
        x = pd.to_datetime(periods)
    except:
        x = range(len(periods))
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45)
    
    # Plot each metric
    for metric in metric_names:
        if metric in metrics:
            values = metrics[metric]
            if len(values) == len(x):
                ax.plot(x, values, 'o-', label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def create_correlation_matrix_plot(corr_matrix: pd.DataFrame,
                                 ax: plt.Axes = None,
                                 cmap: str = 'RdYlBu_r',
                                 show_values: bool = True) -> plt.Axes:
    """
    Create a correlation matrix heatmap.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    cmap : str, optional
        Colormap name
    show_values : bool, optional
        Whether to show values in cells
        
    Returns:
    --------
    plt.Axes
        Plot axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
              mask=mask,
              cmap=cmap,
              annot=show_values,
              fmt='.2f',
              square=True,
              cbar_kws={'label': 'Correlation Coefficient'},
              ax=ax)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    return ax

# ----------------------
# Utility Functions
# ----------------------

def save_plot(fig: plt.Figure, 
            filename: str, 
            output_dir: str = 'results/plots',
            dpi: int = 150,
            bbox_inches: str = 'tight') -> str:
    """
    Save a plot to file.
    
    Parameters:
    -----------
    fig : plt.Figure
        Figure to save
    filename : str
        Output filename
    output_dir : str, optional
        Output directory
    dpi : int, optional
        DPI for saving
    bbox_inches : str, optional
        Bbox inches parameter for saving
        
    Returns:
    --------
    str
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure filename has .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    
    # Save the figure
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, bbox_inches=bbox_inches, dpi=dpi)
    plt.close(fig)
    
    return output_path

# ----------------------
# Yellowbrick Integration
# ----------------------

def get_yellowbrick_visualizer(viz_type: str, model: Any, **kwargs) -> Any:
    """
    Get appropriate Yellowbrick visualizer for a given type.
    
    Parameters:
    -----------
    viz_type : str
        Type of visualization ('confusion_matrix', 'roc_curve', 'pr_curve', 
                              'feature_importance', 'learning_curve')
    model : Any
        Model to visualize
    **kwargs : dict
        Additional parameters for visualizer
        
    Returns:
    --------
    Any
        Yellowbrick visualizer instance
    """
    viz_map = {
        'confusion_matrix': ConfusionMatrix,
        'roc_curve': ROCAUC,
        'pr_curve': PrecisionRecallCurve,
        'feature_importance': FeatureImportances,
        'learning_curve': LearningCurve
    }
    
    if viz_type not in viz_map:
        logger.error(f"Unknown visualizer type: {viz_type}")
        return None
    
    visualizer_class = viz_map[viz_type]
    return visualizer_class(model, **kwargs)
