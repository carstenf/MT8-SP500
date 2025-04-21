"""
Visualization Module

This module handles the creation of performance visualization plots.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)

def create_performance_visualizations(
    metrics: Dict,
    output_dir: str = 'results/plots',
    viz_config: Optional[Dict] = None
) -> Dict:
    """
    Create and save model performance visualizations.
    
    Parameters:
    -----------
    metrics : Dict
        Dictionary with performance metrics containing:
        - confusion_matrix: List[List[int]], confusion matrix values
        - roc_curve: Dict with 'fpr', 'tpr', 'thresholds' (optional)
        - roc_auc: float, ROC AUC score (optional)
        - pr_curve: Dict with 'precision', 'recall', 'thresholds' (optional)
        - pr_auc: float, PR AUC score (optional)
        - class_report: Dict, classification report by class
    output_dir : str, optional
        Directory to save the plots
    viz_config : Dict, optional
        Visualization configuration settings
        
    Returns:
    --------
    Dict
        Dictionary with paths to the saved plots
    """
    logger.info("Creating performance visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = {}
    
    # Get plot settings from config
    if viz_config is None:
        viz_config = {}
    
    # Plot confusion matrix if available
    if 'confusion_matrix' in metrics:
        try:
            figsize = viz_config.get('plot_figsize', {}).get('confusion_matrix', (8, 6))
            plt.figure(figsize=figsize)
            
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save the plot
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            
            plot_paths['confusion_matrix'] = cm_path
        except Exception as e:
            logger.error(f"Failed to create confusion matrix plot: {str(e)}")
            if plt.get_fignums():
                plt.close()
    
    # Plot ROC curve if available
    if 'roc_curve' in metrics and metrics['roc_curve'] is not None:
        try:
            figsize = viz_config.get('plot_figsize', {}).get('default', (8, 6))
            linewidth = viz_config.get('plot_style', {}).get('linewidth', 2)
            
            plt.figure(figsize=figsize)
            fpr = metrics['roc_curve']['fpr']
            tpr = metrics['roc_curve']['tpr']
            plt.plot(fpr, tpr, lw=linewidth)
            plt.plot([0, 1], [0, 1], 'k--', lw=linewidth)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (AUC = {metrics.get("roc_auc", 0):.3f})')
            
            # Save the plot
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            plt.savefig(roc_path)
            plt.close()
            
            plot_paths['roc_curve'] = roc_path
        except Exception as e:
            logger.error(f"Failed to create ROC curve plot: {str(e)}")
            if plt.get_fignums():
                plt.close()
    
    # Plot Precision-Recall curve if available
    if 'pr_curve' in metrics and metrics['pr_curve'] is not None:
        try:
            figsize = viz_config.get('plot_figsize', {}).get('default', (8, 6))
            plt.figure(figsize=figsize)
            precision = metrics['pr_curve']['precision']
            recall = metrics['pr_curve']['recall']
            plt.plot(recall, precision, lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve (AUC = {metrics.get("pr_auc", 0):.3f})')
            
            # Save the plot
            pr_path = os.path.join(output_dir, 'pr_curve.png')
            plt.savefig(pr_path)
            plt.close()
            
            plot_paths['pr_curve'] = pr_path
        except Exception as e:
            logger.error(f"Failed to create PR curve plot: {str(e)}")
            if plt.get_fignums():
                plt.close()
    
    # Plot class metrics comparison if available
    if 'class_report' in metrics:
        try:
            plt.figure(figsize=(10, 6))
            
            # Extract metrics from class report
            class_metrics = metrics['class_report']
            if '1' in class_metrics and '0' in class_metrics:
                positive_metrics = class_metrics['1']
                negative_metrics = class_metrics['0']
                
                # Create metric comparison
                metric_names = ['precision', 'recall', 'f1-score']
                pos_values = [positive_metrics[m] for m in metric_names]
                neg_values = [negative_metrics[m] for m in metric_names]
                
                x = np.arange(len(metric_names))
                width = 0.35
                
                plt.bar(x - width/2, pos_values, width, label='Positive Class (1)')
                plt.bar(x + width/2, neg_values, width, label='Negative Class (0)')
                
                plt.xlabel('Metric')
                plt.ylabel('Score')
                plt.title('Performance Metrics by Class')
                plt.xticks(x, metric_names)
                plt.ylim(0, 1)
                plt.legend()
                
                # Save the plot
                metrics_path = os.path.join(output_dir, 'class_metrics.png')
                plt.savefig(metrics_path)
                plt.close()
                
                plot_paths['class_metrics'] = metrics_path
        except Exception as e:
            logger.error(f"Failed to create class metrics plot: {str(e)}")
            if plt.get_fignums():
                plt.close()
    
    logger.info(f"Created {len(plot_paths)} visualization plots in {output_dir}")
    
    return plot_paths

def create_feature_importance_plot(
    feature_importance: Dict,
    output_dir: str = 'results/plots',
    n_features: int = 20,
    viz_config: Optional[Dict] = None
) -> str:
    """
    Create and save feature importance visualization.
    
    Parameters:
    -----------
    feature_importance : Dict
        Dictionary with feature importance information
    output_dir : str, optional
        Directory to save the plot
    n_features : int, optional
        Number of top features to include in the plot
    viz_config : Dict, optional
        Visualization configuration settings
        
    Returns:
    --------
    str
        Path to the saved plot file
    """
    logger.info(f"Creating feature importance plot for top {n_features} features...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if feature importance data is available
    if 'importance_df' not in feature_importance:
        logger.error("Feature importance data not available")
        return None
    
    try:
        # Get importance DataFrame
        importance_df = feature_importance['importance_df']
        
        # Limit to top N features
        plot_df = importance_df.head(n_features).copy()
        
        # Get plot settings from config
        if viz_config is None:
            viz_config = {}
        figsize = viz_config.get('plot_figsize', {}).get('feature_importance', (12, 8))
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Create horizontal bar chart
        plt.barh(range(len(plot_df)), plot_df['importance'], align='center')
        plt.yticks(range(len(plot_df)), plot_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {n_features} Feature Importances ({feature_importance.get("method", "unknown")} method)')
        
        # Add values to the bars
        for i, importance in enumerate(plot_df['importance']):
            plt.text(importance, i, f'{importance:.4f}', va='center')
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
        
        return plot_path
        
    except Exception as e:
        logger.error(f"Failed to create feature importance plot: {str(e)}")
        if plt.get_fignums():
            plt.close()
        return None

def create_time_series_performance_plot(
    time_analysis: Dict,
    output_dir: str = 'results/plots',
    metric: str = 'balanced_accuracy',
    viz_config: Optional[Dict] = None
) -> str:
    """
    Create and save time series performance plot.
    
    Parameters:
    -----------
    time_analysis : Dict
        Dictionary with time-based performance analysis
    output_dir : str, optional
        Directory to save the plot
    metric : str, optional
        Metric to plot over time
    viz_config : Dict, optional
        Visualization configuration settings
        
    Returns:
    --------
    str
        Path to the saved plot file
    """
    logger.info(f"Creating time series plot for {metric}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if time analysis data is available
    if 'time_series' not in time_analysis:
        logger.error("Time series data not available")
        return None
    
    try:
        # Extract data
        time_series = time_analysis['time_series']
        
        if 'periods' not in time_series or metric not in time_series:
            logger.error(f"Required time series data for {metric} not available")
            return None
        
        # Get plot settings from config
        if viz_config is None:
            viz_config = {}
        figsize = viz_config.get('plot_figsize', {}).get('default', (12, 6))
        
        # Create plot
        plt.figure(figsize=figsize)
        
        periods = time_series['periods']
        values = time_series[metric]
        
        # Plot time series
        plt.plot(range(len(periods)), values, 'o-', linewidth=2)
        plt.xticks(range(len(periods)), periods, rotation=45)
        
        # Add trend line if enough data points
        if len(periods) >= 3:
            x = np.arange(len(periods))
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), '--', linewidth=1, color='red')
            
            # Add trend direction annotation
            trend = time_analysis.get('trend_direction', {}).get(metric, 'unknown')
            plt.annotate(f"Trend: {trend}", xy=(0.7, 0.05), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
        
        plt.xlabel('Time Period')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Over Time')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, f'timeseries_{metric}.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Time series plot saved to {plot_path}")
        
        return plot_path
        
    except Exception as e:
        logger.error(f"Failed to create time series plot: {str(e)}")
        if plt.get_fignums():
            plt.close()
        return None
