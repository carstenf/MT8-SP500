"""
Visualization Utilities Module

This module provides utility functions for creating visualizations in the PDF report.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

def create_confusion_matrix_plot(confusion_matrix: List[List[int]], ax: plt.Axes) -> None:
    """
    Create a confusion matrix heatmap.
    
    Parameters:
    -----------
    confusion_matrix : List[List[int]]
        2x2 confusion matrix values
    ax : plt.Axes
        Matplotlib axes to plot on
    """
    cm = np.array(confusion_matrix)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

def create_roc_curve_plot(fpr: List[float], 
                         tpr: List[float], 
                         roc_auc: float,
                         ax: plt.Axes,
                         linewidth: int = 2) -> None:
    """
    Create a ROC curve plot.
    
    Parameters:
    -----------
    fpr : List[float]
        False positive rates
    tpr : List[float]
        True positive rates
    roc_auc : float
        ROC AUC score
    ax : plt.Axes
        Matplotlib axes to plot on
    linewidth : int, optional
        Line width for the plot
    """
    ax.plot(fpr, tpr, lw=linewidth)
    ax.plot([0, 1], [0, 1], 'k--', lw=linewidth)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve (AUC = {roc_auc:.3f})')

def create_pr_curve_plot(precision: List[float], 
                        recall: List[float], 
                        pr_auc: float,
                        ax: plt.Axes,
                        linewidth: int = 2) -> None:
    """
    Create a Precision-Recall curve plot.
    
    Parameters:
    -----------
    precision : List[float]
        Precision values
    recall : List[float]
        Recall values
    pr_auc : float
        PR AUC score
    ax : plt.Axes
        Matplotlib axes to plot on
    linewidth : int, optional
        Line width for the plot
    """
    ax.plot(recall, precision, lw=linewidth)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve (AUC = {pr_auc:.3f})')

def create_feature_importance_plot(importance_df: pd.DataFrame,
                                 ax: plt.Axes,
                                 n_features: int = 15) -> None:
    """
    Create a feature importance bar plot.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with feature importance information
    ax : plt.Axes
        Matplotlib axes to plot on
    n_features : int, optional
        Number of top features to display
    """
    top_features = importance_df.head(n_features)
    y_pos = range(len(top_features))
    
    ax.barh(y_pos, top_features['importance'], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()  # Features with highest importance at the top
    ax.set_xlabel('Importance')

def create_time_series_plot(time_series: Dict,
                           ax: plt.Axes,
                           metrics: List[str] = None) -> None:
    """
    Create a time series plot of performance metrics.
    
    Parameters:
    -----------
    time_series : Dict
        Dictionary with time series data
    ax : plt.Axes
        Matplotlib axes to plot on
    metrics : List[str], optional
        List of metrics to plot
    """
    if not metrics:
        metrics = ['balanced_accuracy', 'precision', 'recall', 'f1']
    
    if 'periods' in time_series:
        periods = time_series['periods']
        
        # Convert string periods to datetime if possible
        try:
            x = pd.to_datetime(periods)
        except:
            x = range(len(periods))
            ax.set_xticks(x)
            ax.set_xticklabels(periods, rotation=45)
        
        # Plot each metric
        for metric in metrics:
            if metric in time_series:
                values = time_series[metric]
                if len(values) == len(x):
                    ax.plot(x, values, 'o-', label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

def create_learning_curve_plot(train_sizes: List[float],
                             train_mean: List[float],
                             test_mean: List[float],
                             train_std: List[float] = None,
                             test_std: List[float] = None,
                             ax: plt.Axes = None) -> None:
    """
    Create a learning curve plot.
    
    Parameters:
    -----------
    train_sizes : List[float]
        Training set sizes
    train_mean : List[float]
        Mean training scores
    test_mean : List[float]
        Mean validation scores
    train_std : List[float], optional
        Standard deviation of training scores
    test_std : List[float], optional
        Standard deviation of validation scores
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    """
    if ax is None:
        _, ax = plt.subplots()
    
    # Plot learning curves
    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    
    # Add shaded regions for standard deviation if provided
    if train_std is not None and test_std is not None:
        ax.fill_between(train_sizes, 
                       [max(t - s, 0) for t, s in zip(train_mean, train_std)],
                       [min(t + s, 1) for t, s in zip(train_mean, train_std)], 
                       alpha=0.1, color='r')
        ax.fill_between(train_sizes, 
                       [max(t - s, 0) for t, s in zip(test_mean, test_std)],
                       [min(t + s, 1) for t, s in zip(test_mean, test_std)], 
                       alpha=0.1, color='g')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Balanced Accuracy Score')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

def load_image_safely(image_path: str) -> Optional[np.ndarray]:
    """
    Safely load an image file.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
        
    Returns:
    --------
    Optional[np.ndarray]
        Image array if successful, None otherwise
    """
    try:
        if os.path.exists(image_path):
            return plt.imread(image_path)
        else:
            logger.warning(f"Image file not found: {image_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def create_table_from_data(data: List[List[str]], 
                          col_labels: List[str],
                          ax: plt.Axes,
                          fontsize: int = 10,
                          scale: Tuple[float, float] = (1, 1.5)) -> None:
    """
    Create a table in a matplotlib axes.
    
    Parameters:
    -----------
    data : List[List[str]]
        Table data as list of rows
    col_labels : List[str]
        Column labels
    ax : plt.Axes
        Matplotlib axes to create table in
    fontsize : int, optional
        Font size for table text
    scale : Tuple[float, float], optional
        Scale factors for table width and height
    """
    ax.axis('off')
    table = ax.table(cellText=data,
                    colLabels=col_labels,
                    loc='center',
                    cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(*scale)
