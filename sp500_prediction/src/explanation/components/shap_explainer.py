"""
SHAP Explainer Module

This module handles SHAP value calculation and visualization for model explanations.
"""

import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Any
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
    
    Parameters:
    -----------
    model : Any
        Trained model for which to generate SHAP values
    X : pd.DataFrame
        Feature matrix (can be training or test data)
    model_type : str, optional
        Type of model ('tree', 'linear', 'deep', or 'auto' for auto-detection)
    sample_size : int, optional
        Number of samples to use for SHAP calculation (for large datasets)
    random_state : int, optional
        Random state for reproducibility when sampling
        
    Returns:
    --------
    Dict
        Dictionary with SHAP values and explanation objects
    """
    logger.info("Generating SHAP values for model explanations...")
    
    # Sample data if needed
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
        # Create appropriate explainer
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_sample)
        elif model_type == 'deep':
            background = shap.kmeans(X_sample, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return {}
            
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification output
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Use positive class
        
        # Get expected value
        if hasattr(explainer, 'expected_value'):
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
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
        import traceback
        traceback.print_exc()
        return {}

def create_shap_summary_plot(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_dir: str = 'results/explanation',
    plot_type: str = 'bar',
    max_display: int = 20,
    class_names: List[str] = None,
    title: str = None,
    show_plots: bool = False
) -> Dict:
    """
    Create SHAP summary visualizations.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values from generate_shap_values function
    X_sample : pd.DataFrame
        Feature matrix used to calculate SHAP values
    output_dir : str, optional
        Directory to save the plots
    plot_type : str, optional
        Type of summary plot ('bar', 'beeswarm', 'violin', or 'compact_dot')
    max_display : int, optional
        Maximum number of features to display
    class_names : List[str], optional
        Names of classes for binary classification models
    title : str, optional
        Title for the plot
    show_plots : bool, optional
        Whether to display plots
        
    Returns:
    --------
    Dict
        Dictionary with paths to saved plots
    """
    logger.info(f"Creating SHAP summary {plot_type} plot...")
    
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    
    try:
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X_sample, plot_type='bar', 
                            max_display=max_display, show=False, 
                            class_names=class_names)
        elif plot_type == 'beeswarm':
            shap.summary_plot(shap_values, X_sample, plot_type=None, 
                            max_display=max_display, show=False, 
                            class_names=class_names)
        elif plot_type == 'violin':
            shap.summary_plot(shap_values, X_sample, plot_type='violin', 
                            max_display=max_display, show=False, 
                            class_names=class_names)
        elif plot_type == 'compact_dot':
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        else:
            logger.warning(f"Unsupported plot type: {plot_type}, defaulting to 'bar'")
            shap.summary_plot(shap_values, X_sample, plot_type='bar', 
                            max_display=max_display, show=False, 
                            class_names=class_names)
            
        if title:
            plt.title(title, fontsize=14)
            
        summary_path = os.path.join(output_dir, f'shap_summary_{plot_type}.png')
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        plt.close()
        
        plot_paths['summary_plot'] = summary_path
        
        # Create dependence plots for top features
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        dependence_paths = []
        for feature in feature_importance.head(5)['feature']:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, X_sample, 
                               interaction_index=None, show=False)
            
            dependence_path = os.path.join(output_dir, f'shap_dependence_{feature}.png')
            plt.tight_layout()
            plt.savefig(dependence_path, dpi=300)
            
            if show_plots:
                plt.show()
            plt.close()
            
            dependence_paths.append(dependence_path)
            
        plot_paths['dependence_plots'] = dependence_paths
        
        return plot_paths
        
    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {str(e)}")
        if plt.get_fignums():
            plt.close()
        return {}

def create_force_plot(
    explainer: Any,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    instance_idx: int = 0,
    output_dir: str = 'results/explanation',
    plot_format: str = 'html',
    show_plots: bool = False
) -> str:
    """
    Create SHAP force plot for a single prediction.
    
    Parameters:
    -----------
    explainer : Any
        SHAP explainer object
    shap_values : np.ndarray
        SHAP values from generate_shap_values function
    X_sample : pd.DataFrame
        Feature matrix used to calculate SHAP values
    instance_idx : int, optional
        Index of the instance to explain
    output_dir : str, optional
        Directory to save the plots
    plot_format : str, optional
        Format to save the plot ('html' or 'png')
    show_plots : bool, optional
        Whether to display plots
        
    Returns:
    --------
    str
        Path to the saved force plot
    """
    logger.info(f"Creating SHAP force plot for instance {instance_idx}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get expected value
        if hasattr(explainer, 'expected_value'):
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            expected_value = 0.5
        
        # Get instance data
        instance = X_sample.iloc[instance_idx:instance_idx+1]
        instance_shap = shap_values[instance_idx:instance_idx+1]
            
        if plot_format == 'html':
            force_plot = shap.force_plot(expected_value, instance_shap, instance, 
                                       matplotlib=False)
            plot_path = os.path.join(output_dir, f'force_plot_{instance_idx}.html')
            shap.save_html(plot_path, force_plot)
        else:
            plt.figure(figsize=(20, 3))
            shap.force_plot(expected_value, instance_shap, instance, 
                          matplotlib=True, show=False)
            plot_path = os.path.join(output_dir, f'force_plot_{instance_idx}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            plt.close()
        
        return plot_path
        
    except Exception as e:
        logger.error(f"Error creating force plot: {str(e)}")
        if plt.get_fignums():
            plt.close()
        return ""
