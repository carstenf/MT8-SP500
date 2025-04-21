"""
Metrics Calculator Module

This module handles the calculation of various performance metrics for model evaluation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    auc
)

# Set up logging
logger = logging.getLogger(__name__)

def calculate_performance_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    Calculate various performance metrics for a classification model.
    
    Parameters:
    -----------
    y_true : pd.Series
        True class labels
    y_pred : pd.Series
        Predicted class labels
    y_prob : pd.Series, optional
        Predicted probabilities for the positive class
    config : Dict, optional
        Configuration dictionary with settings:
        - metrics.zero_division: int, value to use for zero division (default: 0)
        - metrics.balanced_accuracy_threshold: float, threshold for accuracy (default: 0.55)
        
    Returns:
    --------
    Dict
        Dictionary of performance metrics including:
        - accuracy: float, overall accuracy
        - balanced_accuracy: float, balanced accuracy score
        - precision: float, precision score
        - recall: float, recall score
        - f1: float, F1 score
        - confusion_matrix: List[List[int]], confusion matrix values
        - meets_accuracy_threshold: bool, whether balanced accuracy meets threshold
        - metrics_version: str, version of metrics calculation
        - config_used: Dict, configuration settings used
        - class_report: Dict, detailed classification report
        - roc_auc: float, optional ROC AUC score if probabilities provided
        - roc_curve: Dict, optional ROC curve data if probabilities provided
        - pr_curve: Dict, optional PR curve data if probabilities provided
        - pr_auc: float, optional PR AUC score if probabilities provided
    """
    logger.info("Calculating performance metrics...")
    
    # Check if we have more than one class
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    is_single_class = len(unique_classes) == 1

    # Get zero_division value from config
    if config is None:
        zero_div = 0
        threshold = 0.55
    elif "metrics" in config:
        zero_div = config["metrics"].get("zero_division", 0)
        threshold = config["metrics"].get("balanced_accuracy_threshold", 0.55)
    else:
        zero_div = config.get("zero_division", 0)
        threshold = config.get("balanced_accuracy_threshold", 0.55)

    # Basic classification metrics with handling for edge cases
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=zero_div, average='weighted'),
        'recall': recall_score(y_true, y_pred, zero_division=zero_div, average='weighted'),
        'f1': f1_score(y_true, y_pred, zero_division=zero_div, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'is_single_class': is_single_class,
        'meets_accuracy_threshold': False  # Initialize to False
    }
    
    # Add class report with zero division handling
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=zero_div)
    metrics['class_report'] = class_report
    
    # Add probability-based metrics if probabilities are provided and we have multiple classes
    if y_prob is not None and not is_single_class:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['roc_auc'] = None
        
        try:
            # Calculate ROC curve with error handling
            fpr, tpr, thresholds = roc_curve(y_true, y_prob, labels=[0, 1])
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        except Exception:
            metrics['roc_curve'] = None
            
        try:
            # Calculate Precision-Recall curve with error handling
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
            metrics['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds_pr.tolist() if len(thresholds_pr) > 0 else None
            }
            metrics['pr_auc'] = auc(recall, precision)
        except Exception:
            metrics['pr_curve'] = None
            metrics['pr_auc'] = None
    
    # Check if balanced accuracy meets threshold
    metrics['meets_accuracy_threshold'] = metrics['balanced_accuracy'] >= threshold
    
    # Add configuration used
    metrics['config_used'] = {
        'zero_division': zero_div,
        'balanced_accuracy_threshold': threshold
    }
    
    # Add version information
    metrics['metrics_version'] = '1.0.0'
    
    logger.info(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}, "
                f"F1 score: {metrics['f1']:.4f}")
    
    return metrics
