"""
Utility functions for model explainability.

This module contains utility functions for generating SHAP values and explanation reports.
"""

import pandas as pd
import numpy as np
import logging
import shap
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import json

# For tree-based models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Set up logging
logger = logging.getLogger(__name__)

def generate_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_type: str = 'auto',
    sample_size: int = None,
    random_state: int = 42
) -> Dict:
    """
    Generate SHAP values for a trained model.
    """
    logger.info("Generating SHAP values for model explanations...")
    
    # If sample_size is provided and less than number of rows, take a sample
    if sample_size is not None and sample_size < len(X):
        logger.info(f"Sampling {sample_size} instances for SHAP calculation")
        X_sample = X.sample(sample_size, random_state=random_state)
    else:
        X_sample = X
    
    # Auto-detect model type if not specified
    if model_type == 'auto':
        if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier, lgb.LGBMClassifier)):
            model_type = 'tree'
        elif hasattr(model, 'coef_'):  # Linear models
            model_type = 'linear'
        elif hasattr(model, 'predict_proba') and not hasattr(model, 'feature_importances_'):
            model_type = 'deep'
        else:
            logger.warning("Could not automatically detect model type, defaulting to 'tree'")
            model_type = 'tree'
    
    try:
        # Create appropriate explainer based on model type
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            background = shap.kmeans(X_sample, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, use positive class
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Extract expected value
        if hasattr(explainer, 'expected_value'):
            if isinstance(explainer.expected_value, list):
                expected_value = explainer.expected_value[1]
            else:
                expected_value = explainer.expected_value
        else:
            expected_value = None
            
        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'expected_value': expected_value,
            'X_sample': X_sample,
            'model_type': model_type
        }
        
    except Exception as e:
        logger.error(f"Error generating SHAP values: {str(e)}")
        return {}

def generate_explanation_report(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series = None,
    output_dir: str = 'results/explanation',
    n_top_features: int = 20,
    sample_size: int = 100,
    model_type: str = 'auto',
    show_plots: bool = False
) -> Dict:
    """
    Generate a comprehensive explanation report for a trained model.
    """
    logger.info("Generating comprehensive model explanation report...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize report
    report = {
        'model_type': model_type,
        'n_features': len(X.columns),
        'n_samples': len(X),
        'sample_size_used': min(sample_size, len(X)),
        'report_paths': {},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # Calculate SHAP values
        shap_result = generate_shap_values(model, X, model_type, sample_size)
        if not shap_result:
            return report
        
        # Store SHAP values and create visualizations
        shap_values = shap_result['shap_values']
        X_sample = shap_result['X_sample']
        
        # Create feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        report['feature_importance'] = feature_importance.to_dict('records')
        
        # Save report as JSON
        report_path = os.path.join(output_dir, 'explanation_report.json')
        with open(report_path, 'w') as f:
            # Helper to convert all numpy/pandas types to native Python types
            def convert_to_native(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(v) for v in obj]
                return obj
            
            json.dump(convert_to_native(report), f, indent=2)
            
        report['report_paths']['main_report'] = report_path
        logger.info(f"Explanation report saved to {report_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating explanation report: {str(e)}")
        return report
