"""
Report Generation Utilities Module

This module provides utility functions for report generation and file handling.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

def collect_visualization_paths(viz_dir: str, model_name: str) -> Dict[str, str]:
    """
    Collect paths to visualization files for a model.
    
    Parameters:
    -----------
    viz_dir : str
        Directory containing visualization files
    model_name : str
        Name of the model to collect visualizations for
        
    Returns:
    --------
    Dict[str, str]
        Dictionary with visualization paths
    """
    viz_paths = {}
    
    # Common visualization file patterns
    patterns = {
        'confusion_matrix': ['confusion_matrix.png', f'{model_name}_confusion_matrix.png'],
        'roc_curve': ['roc_curve.png', f'{model_name}_roc_curve.png'],
        'pr_curve': ['pr_curve.png', f'{model_name}_pr_curve.png'],
        'feature_importance': ['feature_importance.png', f'{model_name}_feature_importance.png'],
        'bias_variance': ['learning_curve.png', f'{model_name}_learning_curve.png'],
        'time_series': ['timeseries_balanced_accuracy.png', f'{model_name}_time_series.png'],
        'shap_summary': ['shap_summary_bar.png', f'{model_name}_shap_summary.png'],
        'shap_dependence': ['shap_dependence_*.png'],  # Wildcard pattern
        'error_analysis': ['error_analysis_feature_importance.png', f'{model_name}_error_analysis.png']
    }
    
    # Check for files matching each pattern
    for key, filenames in patterns.items():
        for filename in filenames:
            # Check if pattern contains wildcard
            if '*' in filename:
                import glob
                # Search for matching files
                matches = glob.glob(os.path.join(viz_dir, filename))
                if matches:
                    viz_paths[key] = matches[0]  # Use the first match
                    break
            else:
                # Check for exact filename
                path = os.path.join(viz_dir, filename)
                if os.path.exists(path):
                    viz_paths[key] = path
                    break
    
    return viz_paths

def format_metrics_for_display(metrics: Dict) -> Dict:
    """
    Format metrics dictionary for display in report.
    
    Parameters:
    -----------
    metrics : Dict
        Dictionary with raw metrics
        
    Returns:
    --------
    Dict
        Dictionary with formatted metrics
    """
    formatted = {}
    
    # Format numeric values
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if key in ['balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = f"{value:,}"
        else:
            formatted[key] = value
    
    return formatted

def generate_success_criteria(metrics: Dict, 
                            time_analysis: Optional[Dict] = None,
                            feature_importance: Optional[Dict] = None) -> Dict:
    """
    Generate success criteria assessment.
    
    Parameters:
    -----------
    metrics : Dict
        Dictionary with model performance metrics
    time_analysis : Dict, optional
        Dictionary with time-based performance analysis
    feature_importance : Dict, optional
        Dictionary with feature importance information
        
    Returns:
    --------
    Dict
        Dictionary with success criteria assessment
    """
    criteria = {
        'balanced_accuracy_above_55': metrics.get('balanced_accuracy', 0) > 0.55,
        'stable_performance': False,
        'model_explanations_available': False,
        'pipeline_reproducible': True  # Assuming the pipeline is reproducible by design
    }
    
    # Check time analysis criteria
    if time_analysis and 'trend_direction' in time_analysis:
        bal_acc_trend = time_analysis['trend_direction'].get('balanced_accuracy', '')
        criteria['stable_performance'] = bal_acc_trend != 'decreasing'
    
    # Check feature importance criteria
    if feature_importance and 'importance_df' in feature_importance:
        criteria['model_explanations_available'] = True
    
    return criteria

def generate_recommendations(metrics: Dict,
                           success_criteria: Dict,
                           bias_variance: Optional[Dict] = None) -> List[str]:
    """
    Generate recommendations based on model performance.
    
    Parameters:
    -----------
    metrics : Dict
        Dictionary with model performance metrics
    success_criteria : Dict
        Dictionary with success criteria assessment
    bias_variance : Dict, optional
        Dictionary with bias-variance analysis
        
    Returns:
    --------
    List[str]
        List of recommendations
    """
    recommendations = []
    
    # Get balanced accuracy
    bal_acc = metrics.get('balanced_accuracy', 0)
    
    # Check bias-variance diagnosis
    if bias_variance and 'diagnosis' in bias_variance:
        diagnosis = bias_variance['diagnosis']
        if diagnosis == 'high_bias':
            recommendations.extend([
                "Use a more complex model",
                "Add more features",
                "Reduce regularization strength"
            ])
        elif diagnosis == 'high_variance':
            recommendations.extend([
                "Use more training data",
                "Apply stronger regularization",
                "Reduce number of features"
            ])
    
    # Add general recommendations based on performance
    if bal_acc >= 0.60:
        recommendations.extend([
            "Deploy model for production use",
            "Monitor performance over time",
            "Consider retraining quarterly with fresh data"
        ])
    elif bal_acc >= 0.55:
        recommendations.extend([
            "Deploy with careful monitoring",
            "Use predictions as one of multiple signals",
            "Implement stringent thresholds for actions"
        ])
    else:
        recommendations.extend([
            "Review feature engineering approach",
            "Try additional model architectures",
            "Consider using more training data"
        ])
    
    return recommendations

def save_report_metadata(output_dir: str,
                        report_info: Dict,
                        filename: str = 'report_metadata.json') -> None:
    """
    Save report generation metadata.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save metadata
    report_info : Dict
        Dictionary with report information
    filename : str, optional
        Name of metadata file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Add generation timestamp
        report_info['generated_at'] = datetime.now().isoformat()
        
        # Save metadata
        metadata_path = os.path.join(output_dir, filename)
        with open(metadata_path, 'w') as f:
            json.dump(report_info, f, indent=4)
            
        logger.info(f"Report metadata saved to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error saving report metadata: {str(e)}")

def validate_report_inputs(metrics: Dict,
                         data_info: Dict,
                         required_metrics: List[str] = None,
                         required_data_info: List[str] = None) -> bool:
    """
    Validate required inputs for report generation.
    
    Parameters:
    -----------
    metrics : Dict
        Dictionary with model performance metrics
    data_info : Dict
        Dictionary with data preparation information
    required_metrics : List[str], optional
        List of required metric keys
    required_data_info : List[str], optional
        List of required data info keys
        
    Returns:
    --------
    bool
        True if all required inputs are present
    """
    if required_metrics is None:
        required_metrics = ['balanced_accuracy', 'precision', 'recall', 'f1']
    
    if required_data_info is None:
        required_data_info = ['start_date', 'end_date', 'n_tickers', 'n_samples']
    
    # Check metrics
    for key in required_metrics:
        if key not in metrics:
            logger.error(f"Missing required metric: {key}")
            return False
    
    # Check data info
    for key in required_data_info:
        if key not in data_info:
            logger.error(f"Missing required data info: {key}")
            return False
    
    return True
