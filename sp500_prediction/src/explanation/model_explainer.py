"""
Model Explainer Module for S&P500 Prediction Project

This module handles model explainability through SHAP values, PDPs, and ICE plots
to provide transparent explanations of stock direction predictions.
"""

import pandas as pd
import numpy as np
import logging
import shap
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Union, Any, Tuple
import pickle
from datetime import datetime
import seaborn as sns

# For tree-based models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        Dictionary with SHAP values and explanation objects:
        - shap_values: np.ndarray, SHAP values for each prediction
        - explainer: shap.Explainer, SHAP explainer object
        - expected_value: float, base/expected value for the model
        - X_sample: pd.DataFrame, sampled data used for SHAP calculation
        - model_type: str, detected or specified model type
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
            # Assume deep learning or complex model
            model_type = 'deep'
        else:
            logger.warning("Could not automatically detect model type, defaulting to 'tree'")
            model_type = 'tree'
    
    logger.info(f"Creating SHAP explainer for model type: {model_type}")
    
    # Create appropriate explainer based on model type
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_sample)
        elif model_type == 'deep':
            # For black-box models, use KernelExplainer
            # This is more computationally expensive
            background = shap.kmeans(X_sample, 100)  # Use kmeans for more efficient background
            explainer = shap.KernelExplainer(model.predict_proba, background)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return {}
            
        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # For classifiers with binary output, sometimes shap_values is a list with [neg_class, pos_class]
        # We want the positive class (index 1) for binary classification
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # For binary classification, use positive class
        
        # Extract expected value
        if hasattr(explainer, 'expected_value'):
            if isinstance(explainer.expected_value, list):
                expected_value = explainer.expected_value[1]  # Positive class for binary
            else:
                expected_value = explainer.expected_value
        else:
            expected_value = None
            
        logger.info("SHAP values generated successfully")
        
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
    Create SHAP summary visualizations to explain feature importance.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values generated from generate_shap_values function
    X_sample : pd.DataFrame
        Feature matrix that was used to calculate SHAP values
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
        Whether to display plots (interactive sessions only)
        
    Returns:
    --------
    Dict
        Dictionary with paths to saved plots:
        - summary_plot: path to the summary plot
        - dependence_plots: paths to dependence plots for top features
    """
    logger.info(f"Creating SHAP summary {plot_type} plot...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = {}
    
    # Set matplotlib style for prettier plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    try:
        # Create the summary plot based on the requested type
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=max_display, 
                             show=False, class_names=class_names)
        elif plot_type == 'beeswarm':
            shap.summary_plot(shap_values, X_sample, plot_type=None, max_display=max_display, 
                             show=False, class_names=class_names)
        elif plot_type == 'violin':
            shap.summary_plot(shap_values, X_sample, plot_type='violin', max_display=max_display, 
                             show=False, class_names=class_names)
        elif plot_type == 'compact_dot':
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        else:
            logger.warning(f"Unsupported plot type: {plot_type}, defaulting to 'bar'")
            shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=max_display, 
                             show=False, class_names=class_names)
            
        # Add title if provided
        if title:
            plt.title(title, fontsize=14)
            
        # Save the plot
        summary_path = os.path.join(output_dir, f'shap_summary_{plot_type}.png')
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['summary_plot'] = summary_path
        
        logger.info(f"SHAP summary plot saved to {summary_path}")
        
        # Create individual dependence plots for top features
        # Determine the top features based on mean absolute SHAP values
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Get top N features (or all if less than max_display)
        top_n_features = min(max_display, len(feature_importance))
        top_features = feature_importance.head(top_n_features)['feature'].tolist()
        
        # Create dependence plots for top features
        dependence_paths = []
        
        for feature in top_features[:5]:  # Limit to top 5 to avoid too many plots
            plt.figure(figsize=(10, 6))
            
            # Create dependence plot
            shap.dependence_plot(
                feature, shap_values, X_sample, 
                interaction_index=None,  # Auto-identify strongest interaction
                show=False
            )
            
            # Save the plot
            dependence_path = os.path.join(output_dir, f'shap_dependence_{feature}.png')
            plt.tight_layout()
            plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            dependence_paths.append(dependence_path)
            logger.info(f"Dependence plot for {feature} saved to {dependence_path}")
        
        plot_paths['dependence_plots'] = dependence_paths
        
        if show_plots:
            logger.info("Displaying plots in interactive session...")
            # Re-create the summary plot for display
            plt.figure(figsize=(10, 8))
            if plot_type == 'bar':
                shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=max_display, 
                                show=True, class_names=class_names)
            elif plot_type == 'beeswarm':
                shap.summary_plot(shap_values, X_sample, plot_type=None, max_display=max_display, 
                                show=True, class_names=class_names)
            else:
                shap.summary_plot(shap_values, X_sample, plot_type=plot_type, max_display=max_display, 
                                show=True, class_names=class_names)
        
        return plot_paths
        
    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def create_partial_dependence_plots(
    model: Any,
    X: pd.DataFrame,
    features: List[str] = None,
    n_top_features: int = 10,
    output_dir: str = 'results/explanation',
    feature_importances: Dict = None,
    grid_resolution: int = 50,
    n_jobs: int = -1,
    show_plots: bool = False
) -> Dict:
    """
    Create Partial Dependence Plots (PDPs) to show the relationship between 
    features and the predicted outcome, while accounting for the average effect
    of all other features.
    
    Parameters:
    -----------
    model : Any
        Trained model for which to create PDPs
    X : pd.DataFrame
        Feature matrix (typically validation or test data)
    features : List[str], optional
        Specific features to compute PDPs for. If None, use top n_top_features
    n_top_features : int, optional
        Number of top features to include if features is None
    output_dir : str, optional
        Directory to save the plots
    feature_importances : Dict, optional
        Dictionary with feature importance information, used to select top features
    grid_resolution : int, optional
        Resolution of the grid for PDP calculation
    n_jobs : int, optional
        Number of jobs for parallel computation
    show_plots : bool, optional
        Whether to display plots (interactive sessions only)
        
    Returns:
    --------
    Dict
        Dictionary with paths to saved plots:
        - individual_plots: list of paths to individual PDP plots
        - pdp_values: computed PDP values for each feature
    """
    logger.info("Creating partial dependence plots...")
    
    from sklearn.inspection import partial_dependence
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If features not specified, use top features from feature_importances
    if features is None:
        if feature_importances is not None and 'importance_df' in feature_importances:
            top_features = feature_importances['importance_df'].head(n_top_features)['feature'].tolist()
        else:
            # If no feature importances provided, try to get from model
            if hasattr(model, 'feature_importances_'):
                # Sort features by importance
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[::-1][:n_top_features]
                top_features = [X.columns[i] for i in top_indices]
            else:
                # If we can't get importances, just take first n_top_features
                logger.warning("No feature importances available, using first n_top_features")
                top_features = X.columns[:n_top_features].tolist()
    else:
        # Use specified features
        top_features = features
        
    logger.info(f"Calculating PDPs for features: {top_features}")
    
    # Calculate PDPs for selected features
    try:
        result = {}
        pdp_values = {}
        individual_plots = []
        
        for feature in top_features:
            # Skip if feature not in X
            if feature not in X.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
                
            # Calculate partial dependence
            pd_results = partial_dependence(
                model, X, [feature], 
                grid_resolution=grid_resolution,
                kind='average'
            )
            
            # Extract values
            pd_feature_values = pd_results["values"][0]
            pd_feature_predictions = pd_results["average"][0]
            
            # Store values for return
            pdp_values[feature] = {
                'feature_values': pd_feature_values.tolist(),
                'average_predictions': pd_feature_predictions.tolist()
            }
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(pd_feature_values, pd_feature_predictions, 'b-', linewidth=2)
            plt.xlabel(feature)
            plt.ylabel('Partial Dependence')
            plt.title(f'Partial Dependence of Prediction on {feature}')
            plt.grid(True, alpha=0.3)
            
            # Add rug plot (data distribution)
            plt.plot(X[feature].values, np.zeros_like(X[feature].values) - 0.02, '|k', ms=15, alpha=0.2)
            
            # Save the plot
            plot_path = os.path.join(output_dir, f'pdp_{feature}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            individual_plots.append(plot_path)
            logger.info(f"PDP for {feature} saved to {plot_path}")
        
        # Store results
        result['individual_plots'] = individual_plots
        result['pdp_values'] = pdp_values
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating partial dependence plots: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def create_ice_plots(
    model: Any,
    X: pd.DataFrame,
    features: List[str] = None,
    n_top_features: int = 5,
    n_samples: int = 20,
    output_dir: str = 'results/explanation',
    feature_importances: Dict = None,
    grid_resolution: int = 50,
    random_state: int = 42,
    show_plots: bool = False
) -> Dict:
    """
    Create Individual Conditional Expectation (ICE) plots to show the relationship
    between features and predicted outcome for individual instances.
    
    Parameters:
    -----------
    model : Any
        Trained model for which to create ICE plots
    X : pd.DataFrame
        Feature matrix (typically validation or test data)
    features : List[str], optional
        Specific features to compute ICE for. If None, use top n_top_features
    n_top_features : int, optional
        Number of top features to include if features is None
    n_samples : int, optional
        Number of instances to sample for ICE plots
    output_dir : str, optional
        Directory to save the plots
    feature_importances : Dict, optional
        Dictionary with feature importance information, used to select top features
    grid_resolution : int, optional
        Resolution of the grid for ICE calculation
    random_state : int, optional
        Random state for reproducibility when sampling
    show_plots : bool, optional
        Whether to display plots (interactive sessions only)
        
    Returns:
    --------
    Dict
        Dictionary with paths to saved plots:
        - ice_plots: list of paths to ICE plots
    """
    logger.info("Creating individual conditional expectation (ICE) plots...")
    
    from sklearn.inspection import partial_dependence
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample instances for ICE plots to avoid clutter
    if n_samples < len(X):
        X_sampled = X.sample(n_samples, random_state=random_state)
    else:
        X_sampled = X
    
    # If features not specified, use top features from feature_importances
    if features is None:
        if feature_importances is not None and 'importance_df' in feature_importances:
            top_features = feature_importances['importance_df'].head(n_top_features)['feature'].tolist()
        else:
            # If no feature importances provided, try to get from model
            if hasattr(model, 'feature_importances_'):
                # Sort features by importance
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[::-1][:n_top_features]
                top_features = [X.columns[i] for i in top_indices]
            else:
                # If we can't get importances, just take first n_top_features
                logger.warning("No feature importances available, using first n_top_features")
                top_features = X.columns[:n_top_features].tolist()
    else:
        # Use specified features
        top_features = features[:n_top_features]  # Limit to n_top_features
        
    logger.info(f"Calculating ICE for features: {top_features}")
    
    # Calculate ICE for selected features
    try:
        ice_plots = []
        
        for feature in top_features:
            # Skip if feature not in X
            if feature not in X.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
                
            # Calculate individual and average conditional expectation
            pd_results = partial_dependence(
                model, X_sampled, [feature], 
                grid_resolution=grid_resolution,
                kind='both'  # Calculate both individual and average
            )
            
            # Extract values
            feature_values = pd_results["values"][0]
            individual_predictions = pd_results["individual"][0]
            average_predictions = pd_results["average"][0]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot individual ICE lines
            for i in range(individual_predictions.shape[0]):
                plt.plot(feature_values, individual_predictions[i], 'C0-', alpha=0.1)
            
            # Plot average (PDP)
            plt.plot(feature_values, average_predictions, 'r-', linewidth=2, label='PDP (average)')
            
            plt.xlabel(feature)
            plt.ylabel('Prediction')
            plt.title(f'Individual Conditional Expectation Plots for {feature}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add rug plot (data distribution)
            plt.plot(X_sampled[feature].values, np.zeros_like(X_sampled[feature].values) - 0.02, '|k', ms=15, alpha=0.2)
            
            # Save the plot
            plot_path = os.path.join(output_dir, f'ice_{feature}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            ice_plots.append(plot_path)
            logger.info(f"ICE plot for {feature} saved to {plot_path}")
        
        return {
            'ice_plots': ice_plots
        }
        
    except Exception as e:
        logger.error(f"Error creating ICE plots: {str(e)}")
        import traceback
        traceback.print_exc()
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
    Create SHAP force plot for a single prediction to visualize each
    feature's contribution to the prediction.
    
    Parameters:
    -----------
    explainer : Any
        SHAP explainer object
    shap_values : np.ndarray
        SHAP values generated from generate_shap_values function
    X_sample : pd.DataFrame
        Feature matrix that was used to calculate SHAP values
    instance_idx : int, optional
        Index of the instance to explain
    output_dir : str, optional
        Directory to save the plots
    plot_format : str, optional
        Format to save the plot ('html' or 'png')
    show_plots : bool, optional
        Whether to display plots (interactive sessions only)
        
    Returns:
    --------
    str
        Path to the saved force plot
    """
    logger.info(f"Creating SHAP force plot for instance {instance_idx}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get expected value (baseline)
        if hasattr(explainer, 'expected_value'):
            if isinstance(explainer.expected_value, list):
                expected_value = explainer.expected_value[1]  # Positive class for binary
            else:
                expected_value = explainer.expected_value
        else:
            logger.warning("No expected_value found in explainer, using 0.5 as default")
            expected_value = 0.5
        
        # Get the instance data
        if isinstance(instance_idx, int):
            if instance_idx >= len(X_sample):
                logger.warning(f"Instance index {instance_idx} out of bounds, using first instance")
                instance_idx = 0
                
            instance = X_sample.iloc[instance_idx]
            instance_shap = shap_values[instance_idx]
        else:
            logger.warning(f"Invalid instance index, using first instance")
            instance = X_sample.iloc[0]
            instance_shap = shap_values[0]
            
        if plot_format == 'html':
            # Create and save HTML version (preferred for interactivity)
            force_plot = shap.force_plot(expected_value, instance_shap, instance, 
                                        matplotlib=False)
            
            # Save as HTML
            plot_path = os.path.join(output_dir, f'force_plot_{instance_idx}.html')
            shap.save_html(plot_path, force_plot)
            
            logger.info(f"Force plot saved as HTML to {plot_path}")
            
        else:  # PNG format
            # Create matplotlib version
            plt.figure(figsize=(20, 3))
            shap.force_plot(expected_value, instance_shap, instance, 
                           matplotlib=True, show=False)
            
            # Save as PNG
            plot_path = os.path.join(output_dir, f'force_plot_{instance_idx}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Force plot saved as PNG to {plot_path}")
        
        if show_plots:
            # For interactive sessions
            plt.figure(figsize=(20, 3))
            shap.force_plot(expected_value, instance_shap, instance, 
                           matplotlib=True, show=True)
        
        return plot_path
        
    except Exception as e:
        logger.error(f"Error creating force plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


def analyze_shap_interactions(
    model: Any,
    X: pd.DataFrame,
    n_top_features: int = 10,
    sample_size: int = 100,
    output_dir: str = 'results/explanation',
    feature_importances: Dict = None,
    random_state: int = 42,
    show_plots: bool = False
) -> Dict:
    """
    Analyze feature interactions using SHAP interaction values.
    
    Parameters:
    -----------
    model : Any
        Trained model for which to analyze interactions
    X : pd.DataFrame
        Feature matrix (typically validation or test data)
    n_top_features : int, optional
        Number of top features to include in the analysis
    sample_size : int, optional
        Number of instances to sample for interaction calculation
    output_dir : str, optional
        Directory to save the plots
    feature_importances : Dict, optional
        Dictionary with feature importance information
    random_state : int, optional
        Random state for reproducibility when sampling
    show_plots : bool, optional
        Whether to display plots (interactive sessions only)
        
    Returns:
    --------
    Dict
        Dictionary with interaction analysis results and plot paths:
        - interaction_values: SHAP interaction values
        - interaction_plots: paths to saved interaction plots
        - interaction_summary: summary of most important interactions
    """
    logger.info("Analyzing feature interactions using SHAP...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample instances to speed up calculation
    if sample_size < len(X):
        X_sample = X.sample(sample_size, random_state=random_state)
    else:
        X_sample = X
    
    # Identify top features if feature_importances provided
    if feature_importances is not None and 'importance_df' in feature_importances:
        top_features = feature_importances['importance_df'].head(n_top_features)['feature'].tolist()
    else:
        top_features = None
    
    try:
        # Create TreeExplainer for interaction values
        # Currently, only TreeExplainer supports interaction values
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP interaction values (this can be computationally intensive)
        logger.info("Calculating SHAP interaction values (this may take a while)...")
        
        if top_features is not None:
            # Filter to top features to reduce computation
            X_filtered = X_sample[top_features]
            shap_interaction_values = explainer.shap_interaction_values(X_filtered)
        else:
            shap_interaction_values = explainer.shap_interaction_values(X_sample)
        
        # SHAP interaction values array has shape (n_samples, n_features, n_features)
        # For classification with 2 classes, it returns a list with 2 arrays
        if isinstance(shap_interaction_values, list):
            # Use positive class (index 1) for binary classification
            shap_interaction_values = shap_interaction_values[1]
        
        # Get feature names
        if top_features is not None:
            feature_names = top_features
        else:
            feature_names = X_sample.columns.tolist()
        
        # Calculate average absolute interaction values
        abs_interaction_values = np.abs(shap_interaction_values).mean(axis=0)
        
        # Create interaction summary
        interaction_summary = []
        
        # Convert to DataFrame for easier manipulation
        interaction_df = pd.DataFrame(abs_interaction_values, 
                                     index=feature_names, 
                                     columns=feature_names)
        
        # Get top interactions excluding self-interactions
        np.fill_diagonal(abs_interaction_values, 0)  # Zero out diagonal
        
        # Find top interaction pairs
        n_pairs = min(10, len(feature_names) * (len(feature_names) - 1) // 2)  # Top 10 or fewer
        flat_indices = np.argsort(abs_interaction_values.flatten())[::-1][:n_pairs]
        
        # Convert flat indices to 2D indices
        for flat_idx in flat_indices:
            i, j = np.unravel_index(flat_idx, abs_interaction_values.shape)
            if i < j:  # Avoid duplicates (interaction i,j = j,i)
                interaction_summary.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'interaction_strength': abs_interaction_values[i, j]
                })
        
        # Create interaction plots
        interaction_plots = []
        
        # Plot top interactions
        for interaction in interaction_summary[:5]:  # Limit to top 5
            feature1 = interaction['feature1']
            feature2 = interaction['feature2']
            
            plt.figure(figsize=(10, 8))
            
            # Calculate SHAP values for the specific subset
            feature_shap_values = explainer.shap_values(X_sample[feature_names])
            if isinstance(feature_shap_values, list):
                feature_shap_values = feature_shap_values[1]  # For binary classification, use positive class
            
            # Plot dependence with color showing interaction
            shap.dependence_plot(
                feature1,
                feature_shap_values,
                X_sample[feature_names],
                interaction_index=feature2,
                show=False
            )
            
            plt.title(f'Interaction between {feature1} and {feature2}')
            
            # Save the plot
            plot_path = os.path.join(output_dir, f'interaction_{feature1}_{feature2}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            interaction_plots.append(plot_path)
            logger.info(f"Interaction plot for {feature1} and {feature2} saved to {plot_path}")
        
        # Save interaction matrix heatmap
        plt.figure(figsize=(12, 10))
        mask = np.zeros_like(abs_interaction_values, dtype=bool)
        mask[np.triu_indices_from(mask)] = True  # Mask upper triangle
        
        sns.heatmap(abs_interaction_values, mask=mask, 
                   xticklabels=feature_names, yticklabels=feature_names,
                   cmap='viridis')
        
        plt.title('SHAP Interaction Values')
        
        # Save the heatmap
        heatmap_path = os.path.join(output_dir, 'interaction_heatmap.png')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        interaction_plots.append(heatmap_path)
        logger.info(f"Interaction heatmap saved to {heatmap_path}")
        
        return {
            'interaction_values': shap_interaction_values,
            'interaction_plots': interaction_plots,
            'interaction_summary': interaction_summary,
            'interaction_matrix': abs_interaction_values.tolist(),
            'feature_names': feature_names
        }
        
    except Exception as e:
        logger.error(f"Error analyzing SHAP interactions: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def create_explanation_for_stock(
    model: Any,
    X: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
    output_dir: str = 'results/explanation/case_studies',
    shap_values: np.ndarray = None,
    explainer: Any = None,
    show_plots: bool = False
) -> Dict:
    """
    Create detailed explanations for a specific stock's prediction.
    
    Parameters:
    -----------
    model : Any
        Trained model to explain
    X : pd.DataFrame
        Feature matrix with multi-index (ticker, date)
    ticker : str
        Stock ticker to explain
    date : pd.Timestamp
        Date for the prediction
    output_dir : str, optional
        Directory to save the explanation
    shap_values : np.ndarray, optional
        Pre-computed SHAP values (if available)
    explainer : Any, optional
        Pre-computed SHAP explainer (if available)
    show_plots : bool, optional
        Whether to display plots (interactive sessions only)
        
    Returns:
    --------
    Dict
        Dictionary with explanation results:
        - prediction: prediction for the stock
        - explanation_plots: paths to explanation plots
        - feature_contributions: feature contributions to the prediction
    """
    logger.info(f"Creating explanation for {ticker} on {date}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Find the specific instance for this ticker and date
        if isinstance(X.index, pd.MultiIndex):
            try:
                instance = X.loc[(ticker, date)]
                instance_df = pd.DataFrame([instance])
                instance_idx = None  # We're using direct selection
            except KeyError:
                logger.error(f"No data found for {ticker} on {date}")
                return {}
        else:
            logger.error("Input data must have multi-index (ticker, date)")
            return {}
        
        # Make prediction for this instance
        prediction = model.predict_proba(instance_df)[0, 1]
        prediction_class = 1 if prediction >= 0.5 else 0
        prediction_label = "UP" if prediction_class == 1 else "DOWN"
        
        # Calculate SHAP values if not provided
        if shap_values is None or explainer is None:
            shap_result = generate_shap_values(model, instance_df)
            if not shap_result:
                logger.error("Failed to generate SHAP values")
                return {
                    'prediction': prediction,
                    'prediction_class': prediction_class,
                    'prediction_label': prediction_label
                }
            local_shap_values = shap_result['shap_values']
            local_explainer = shap_result['explainer']
        else:
            # Find the index of this instance in the original data
            if instance_idx is None:
                # Try to find the index in the original data
                if isinstance(X.index, pd.MultiIndex):
                    try:
                        instance_idx = list(X.index).index((ticker, date))
                        local_shap_values = np.array([shap_values[instance_idx]])
                    except ValueError:
                        logger.error(f"Instance ({ticker}, {date}) not found in original data")
                        return {}
                else:
                    logger.error("Cannot find instance in provided SHAP values")
                    return {}
            else:
                local_shap_values = np.array([shap_values[instance_idx]])
            local_explainer = explainer
        
        # Create waterfall plot for the instance
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(local_shap_values[0], max_display=10, show=False)
        
        # Save waterfall plot
        waterfall_path = os.path.join(output_dir, f'{ticker}_{date.strftime("%Y-%m-%d")}_waterfall.png')
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        # Create force plot
        force_plot_path = create_force_plot(
            local_explainer,
            local_shap_values,
            instance_df,
            0,
            output_dir,
            plot_format='html',
            show_plots=show_plots
        )
        
        # Calculate feature contributions
        feature_contributions = []
        for i, feature in enumerate(instance_df.columns):
            feature_contributions.append({
                'feature': feature,
                'value': instance[feature],
                'contribution': local_shap_values[0, i],
                'abs_contribution': abs(local_shap_values[0, i])
            })
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        # Determine driving factors
        top_positive = [fc for fc in feature_contributions if fc['contribution'] > 0][:3]
        top_negative = [fc for fc in feature_contributions if fc['contribution'] < 0][:3]
        
        # Create explanation summary
        explanation = {
            'ticker': ticker,
            'date': date.strftime('%Y-%m-%d'),
            'prediction': float(prediction),
            'prediction_class': int(prediction_class),
            'prediction_label': prediction_label,
            'explanation_plots': {
                'waterfall': waterfall_path,
                'force': force_plot_path
            },
            'feature_contributions': feature_contributions,
            'top_factors': {
                'positive': top_positive,
                'negative': top_negative
            }
        }
        
        logger.info(f"Created explanation for {ticker} on {date}, prediction: {prediction_label} ({prediction:.4f})")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error creating explanation for {ticker} on {date}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def analyze_correct_vs_incorrect(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str = 'results/explanation/error_analysis',
    n_samples: int = 10,
    random_state: int = 42,
    show_plots: bool = False
) -> Dict:
    """
    Analyze differences between correct and incorrect predictions.
    
    Parameters:
    -----------
    model : Any
        Trained model to analyze
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        True target values
    output_dir : str, optional
        Directory to save the analysis
    n_samples : int, optional
        Number of samples to analyze in each category
    random_state : int, optional
        Random state for reproducibility when sampling
    show_plots : bool, optional
        Whether to display plots (interactive sessions only)
        
    Returns:
    --------
    Dict
        Dictionary with analysis results:
        - correct_cases: explanations for correct predictions
        - incorrect_cases: explanations for incorrect predictions
        - feature_importance_diff: differences in feature importance
        - summary_plot: path to summary comparison plot
    """
    logger.info("Analyzing correct vs. incorrect predictions...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Identify correct and incorrect predictions
        correct_mask = (y_pred == y)
        incorrect_mask = ~correct_mask
        
        # Get indices for correct and incorrect predictions
        correct_indices = np.where(correct_mask)[0]
        incorrect_indices = np.where(incorrect_mask)[0]
        
        # Sample if we have more than n_samples
        if len(correct_indices) > n_samples:
            correct_samples = np.random.RandomState(random_state).choice(
                correct_indices, n_samples, replace=False)
        else:
            correct_samples = correct_indices
            
        if len(incorrect_indices) > n_samples:
            incorrect_samples = np.random.RandomState(random_state).choice(
                incorrect_indices, n_samples, replace=False)
        else:
            incorrect_samples = incorrect_indices
            
        # Create X and y subsets
        X_correct = X.iloc[correct_samples]
        y_correct = y.iloc[correct_samples]
        y_pred_correct = y_pred[correct_samples]
        
        X_incorrect = X.iloc[incorrect_samples]
        y_incorrect = y.iloc[incorrect_samples]
        y_pred_incorrect = y_pred[incorrect_samples]
        
        # Calculate SHAP values for both groups
        shap_correct = generate_shap_values(model, X_correct)
        shap_incorrect = generate_shap_values(model, X_incorrect)
        
        # Check if SHAP calculation was successful
        if not shap_correct or not shap_incorrect:
            logger.error("Failed to calculate SHAP values for analysis")
            return {}
            
        # Create summary plots for both groups
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        shap.summary_plot(shap_correct['shap_values'], X_correct, plot_type='bar', 
                         show=False, max_display=10)
        plt.title('Feature Importance for Correct Predictions')
        
        plt.subplot(2, 1, 2)
        shap.summary_plot(shap_incorrect['shap_values'], X_incorrect, plot_type='bar', 
                         show=False, max_display=10)
        plt.title('Feature Importance for Incorrect Predictions')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = os.path.join(output_dir, 'correct_vs_incorrect_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        # Calculate average absolute SHAP values for each feature
        correct_importance = pd.DataFrame({
            'feature': X_correct.columns,
            'importance': np.abs(shap_correct['shap_values']).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        incorrect_importance = pd.DataFrame({
            'feature': X_incorrect.columns,
            'importance': np.abs(shap_incorrect['shap_values']).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Merge to find differences
        importance_diff = pd.merge(
            correct_importance, incorrect_importance,
            on='feature', suffixes=('_correct', '_incorrect')
        )
        
        # Calculate difference and ratio
        importance_diff['diff'] = importance_diff['importance_incorrect'] - importance_diff['importance_correct']
        importance_diff['ratio'] = importance_diff['importance_incorrect'] / importance_diff['importance_correct']
        
        # Sort by absolute difference
        importance_diff = importance_diff.sort_values('diff', key=abs, ascending=False)
        
        # Create error analysis plot
        plt.figure(figsize=(12, 8))
        features = importance_diff.head(10)['feature'].tolist()
        correct_values = importance_diff.head(10)['importance_correct'].tolist()
        incorrect_values = importance_diff.head(10)['importance_incorrect'].tolist()
        
        x = np.arange(len(features))
        width = 0.35
        
        plt.bar(x - width/2, correct_values, width, label='Correct Predictions')
        plt.bar(x + width/2, incorrect_values, width, label='Incorrect Predictions')
        
        plt.xlabel('Features')
        plt.ylabel('Mean |SHAP Value|')
        plt.title('Feature Importance: Correct vs. Incorrect Predictions')
        plt.xticks(x, features, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        
        # Save error analysis plot
        error_analysis_path = os.path.join(output_dir, 'error_analysis_feature_importance.png')
        plt.savefig(error_analysis_path, dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Create individual explanations for a few examples
        correct_examples = []
        incorrect_examples = []
        
        # Process a few examples from each category
        for i in range(min(5, len(X_correct))):
            instance = X_correct.iloc[i:i+1]
            instance_shap = shap_correct['shap_values'][i:i+1]
            
            # Create force plot
            force_path = create_force_plot(
                shap_correct['explainer'],
                instance_shap,
                instance,
                i,
                os.path.join(output_dir, 'correct'),
                plot_format='html',
                show_plots=False
            )
            
            # Determine driving factors
            feature_contributions = []
            for j, feature in enumerate(instance.columns):
                feature_contributions.append({
                    'feature': feature,
                    'value': instance.iloc[0, j],
                    'contribution': instance_shap[0, j],
                    'abs_contribution': abs(instance_shap[0, j])
                })
                
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            correct_examples.append({
                'index': correct_samples[i],
                'true_value': int(y_correct.iloc[i]),
                'predicted_value': int(y_pred_correct[i]),
                'force_plot': force_path,
                'top_features': feature_contributions[:5]
            })
            
        for i in range(min(5, len(X_incorrect))):
            instance = X_incorrect.iloc[i:i+1]
            instance_shap = shap_incorrect['shap_values'][i:i+1]
            
            # Create force plot
            force_path = create_force_plot(
                shap_incorrect['explainer'],
                instance_shap,
                instance,
                i,
                os.path.join(output_dir, 'incorrect'),
                plot_format='html',
                show_plots=False
            )
            
            # Determine driving factors
            feature_contributions = []
            for j, feature in enumerate(instance.columns):
                feature_contributions.append({
                    'feature': feature,
                    'value': instance.iloc[0, j],
                    'contribution': instance_shap[0, j],
                    'abs_contribution': abs(instance_shap[0, j])
                })
                
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            incorrect_examples.append({
                'index': incorrect_samples[i],
                'true_value': int(y_incorrect.iloc[i]),
                'predicted_value': int(y_pred_incorrect[i]),
                'force_plot': force_path,
                'top_features': feature_contributions[:5]
            })
            
        return {
            'correct_cases': correct_examples,
            'incorrect_cases': incorrect_examples,
            'feature_importance_diff': importance_diff.to_dict('records'),
            'summary_plots': {
                'comparison': comparison_path,
                'error_analysis': error_analysis_path
            },
            'sample_counts': {
                'total': len(X),
                'correct': int(correct_mask.sum()),
                'incorrect': int(incorrect_mask.sum()),
                'accuracy': float(correct_mask.mean())
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing correct vs incorrect predictions: {str(e)}")
        import traceback
        traceback.print_exc()
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
    
    Parameters:
    -----------
    model : Any
        Trained model to explain
    X : pd.DataFrame
        Feature matrix
    y : pd.Series, optional
        True target values (for error analysis)
    output_dir : str, optional
        Directory to save the report
    n_top_features : int, optional
        Number of top features to include in explanations
    sample_size : int, optional
        Number of samples to use for calculations
    model_type : str, optional
        Type of model ('tree', 'linear', 'deep', or 'auto')
    show_plots : bool, optional
        Whether to display plots (interactive sessions only)
        
    Returns:
    --------
    Dict
        Dictionary with comprehensive explanation results:
        - shap_summary: SHAP summary information
        - pdp_analysis: PDP analysis results
        - ice_analysis: ICE plot information
        - interaction_analysis: feature interactions
        - error_analysis: correct vs. incorrect analysis (if y provided)
        - report_paths: paths to generated explanation files
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
    
    # Calculate SHAP values
    logger.info("Step 1: Calculating SHAP values...")
    shap_result = generate_shap_values(
        model, X, model_type=model_type, 
        sample_size=sample_size
    )
    
    if not shap_result:
        logger.error("Failed to generate SHAP values, cannot continue with explanation")
        return report
    
    # Store SHAP values for later use
    shap_values = shap_result['shap_values']
    explainer = shap_result['explainer']
    X_sample = shap_result['X_sample']
    
    # Create SHAP summary plots
    logger.info("Step 2: Creating SHAP summary visualizations...")
    
    # Bar plot summary
    bar_plots = create_shap_summary_plot(
        shap_values, X_sample, output_dir, 'bar', 
        n_top_features, show_plots=show_plots
    )
    
    # Beeswarm plot
    beeswarm_plots = create_shap_summary_plot(
        shap_values, X_sample, output_dir, 'beeswarm',
        n_top_features, show_plots=show_plots
    )
    
    # Store SHAP information in report
    report['shap_summary'] = {
        'feature_importance': pd.DataFrame({
            'feature': X_sample.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False).to_dict('records'),
        'plots': {
            'bar': bar_plots.get('summary_plot'),
            'beeswarm': beeswarm_plots.get('summary_plot'),
            'dependence': bar_plots.get('dependence_plots', [])
        },
        'expected_value': float(shap_result['expected_value']) if shap_result['expected_value'] is not None else None
    }
    
    # Create PDP plots
    logger.info("Step 3: Creating partial dependence plots...")
    pdp_results = create_partial_dependence_plots(
        model, X_sample, n_top_features=n_top_features,
        output_dir=output_dir, show_plots=show_plots
    )
    
    report['pdp_analysis'] = {
        'pdp_plots': pdp_results.get('individual_plots', []),
        'pdp_values': pdp_results.get('pdp_values', {})
    }
    
    # Create ICE plots
    logger.info("Step 4: Creating ICE plots...")
    ice_results = create_ice_plots(
        model, X_sample, n_top_features=min(5, n_top_features),
        output_dir=output_dir, show_plots=show_plots
    )
    
    report['ice_analysis'] = {
        'ice_plots': ice_results.get('ice_plots', [])
    }
    
    # Analyze feature interactions
    if model_type == 'tree' or (model_type == 'auto' and 
                                isinstance(model, (RandomForestClassifier, xgb.XGBClassifier, lgb.LGBMClassifier))):
        logger.info("Step 5: Analyzing feature interactions...")
        interaction_results = analyze_shap_interactions(
            model, X_sample, n_top_features=min(10, n_top_features),
            sample_size=min(50, sample_size),  # Use a smaller sample for interactions
            output_dir=output_dir, show_plots=show_plots
        )
        
        report['interaction_analysis'] = {
            'interaction_plots': interaction_results.get('interaction_plots', []),
            'top_interactions': interaction_results.get('interaction_summary', [])
        }
    else:
        logger.info("Skipping interaction analysis for non-tree models")
        report['interaction_analysis'] = None
    
    # Error analysis if y is provided
    if y is not None:
        logger.info("Step 6: Performing error analysis...")
        error_analysis = analyze_correct_vs_incorrect(
            model, X, y, 
            output_dir=os.path.join(output_dir, 'error_analysis'),
            n_samples=min(20, sample_size),
            show_plots=show_plots
        )
        
        report['error_analysis'] = {
            'sample_counts': error_analysis.get('sample_counts', {}),
            'feature_importance_diff': error_analysis.get('feature_importance_diff', []),
            'plots': error_analysis.get('summary_plots', {})
        }
    else:
        logger.info("Skipping error analysis (no target values provided)")
        report['error_analysis'] = None
    
    # Create case studies for specific instances
    logger.info("Step 7: Creating example case studies...")
    
    # Select a few examples for detailed explanation
    n_examples = min(5, len(X_sample))
    example_indices = np.random.choice(len(X_sample), n_examples, replace=False)
    
    case_studies = []
    
    for idx in example_indices:
        instance = X_sample.iloc[idx:idx+1]
        instance_shap = shap_values[idx:idx+1]
        
        # Create force plot
        force_path = create_force_plot(
            explainer,
            instance_shap,
            instance,
            idx,
            os.path.join(output_dir, 'case_studies'),
            plot_format='html',
            show_plots=False
        )
        
        # Determine driving factors
        feature_contributions = []
        for j, feature in enumerate(instance.columns):
            feature_contributions.append({
                'feature': feature,
                'value': float(instance.iloc[0, j]),
                'contribution': float(instance_shap[0, j]),
                'abs_contribution': float(abs(instance_shap[0, j]))
            })
            
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        prediction = model.predict_proba(instance)[0, 1]
        
        case_studies.append({
            'instance_idx': int(idx),
            'prediction': float(prediction),
            'force_plot': force_path,
            'top_contributions': feature_contributions[:5]
        })
    
    report['case_studies'] = case_studies
    
    # Save the report as JSON
    try:
        # Convert non-JSON serializable elements
        report_json = {k: report[k] for k in report}
        
        # Handle non-serializable types
        for key in report_json:
            if key == 'shap_summary' and 'expected_value' in report_json[key]:
                if isinstance(report_json[key]['expected_value'], np.ndarray):
                    report_json[key]['expected_value'] = report_json[key]['expected_value'].tolist()
                elif isinstance(report_json[key]['expected_value'], np.number):
                    report_json[key]['expected_value'] = float(report_json[key]['expected_value'])
        
        # Save report
        report_path = os.path.join(output_dir, 'explanation_report.json')
        with open(report_path, 'w') as f:
            import json
            json.dump(report_json, f, indent=2)
            
        report['report_paths']['main_report'] = report_path
        logger.info(f"Explanation report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error saving explanation report: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return report


class ModelExplainer:
    """
    Class to handle model explainability for S&P500 prediction.
    
    This class provides methods to:
    - Generate SHAP values and visualizations
    - Create Partial Dependence Plots (PDPs)
    - Create Individual Conditional Expectation (ICE) plots
    - Analyze feature interactions
    - Create case studies for specific stock predictions
    - Compare correct and incorrect predictions
    - Generate comprehensive explanation reports
    
    The explainer works with different model types and provides detailed
    insights into model behavior for better understanding and trust.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the ModelExplainer with optional configuration.
        
        Parameters:
        -----------
        config : Dict, optional
            Configuration dictionary with options for model explanation
        """
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'results/explanation')
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
    def explain_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series = None,
        model_type: str = 'auto',
        n_top_features: int = 20,
        sample_size: int = 100,
        show_plots: bool = False
    ) -> Dict:
        """
        Generate a comprehensive explanation for a trained model.
        
        Parameters:
        -----------
        model : Any
            Trained model to explain
        X : pd.DataFrame
            Feature matrix
        y : pd.Series, optional
            True target values (for error analysis)
        model_type : str, optional
            Type of model ('tree', 'linear', 'deep', or 'auto')
        n_top_features : int, optional
            Number of top features to include in explanations
        sample_size : int, optional
            Number of samples to use for calculations
        show_plots : bool, optional
            Whether to display plots (interactive sessions only)
            
        Returns:
        --------
        Dict
            Dictionary with explanation results
        """
        # Update n_top_features and sample_size from config if provided
        n_top_features = self.config.get('n_top_features', n_top_features)
        sample_size = self.config.get('sample_size', sample_size)
        
        # Generate the explanation report
        report = generate_explanation_report(
            model, X, y, self.output_dir, 
            n_top_features, sample_size, 
            model_type, show_plots
        )
        
        # Store SHAP explainer and values if available
        if 'shap_summary' in report:
            # The explainer and SHAP values are not stored in the report
            # for JSON serialization, so we need to regenerate them
            shap_result = generate_shap_values(
                model, X, model_type=model_type, 
                sample_size=sample_size
            )
            if shap_result:
                self.explainer = shap_result['explainer']
                self.shap_values = shap_result['shap_values']
                self.feature_names = X.columns.tolist()
        
        return report
    
    def explain_stock_prediction(
        self,
        model: Any,
        X: pd.DataFrame,
        ticker: str,
        date: pd.Timestamp,
        show_plots: bool = False
    ) -> Dict:
        """
        Create detailed explanation for a specific stock's prediction.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X : pd.DataFrame
            Feature matrix with multi-index (ticker, date)
        ticker : str
            Stock ticker to explain
        date : pd.Timestamp
            Date for the prediction
        show_plots : bool, optional
            Whether to display plots (interactive sessions only)
            
        Returns:
        --------
        Dict
            Dictionary with explanation for the stock
        """
        # Create directory for stock-specific explanations
        stock_dir = os.path.join(self.output_dir, 'stocks', ticker)
        
        return create_explanation_for_stock(
            model, X, ticker, date,
            output_dir=stock_dir,
            shap_values=self.shap_values,
            explainer=self.explainer,
            show_plots=show_plots
        )
    
    def analyze_model_errors(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_samples: int = 10,
        show_plots: bool = False
    ) -> Dict:
        """
        Analyze differences between correct and incorrect predictions.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            True target values
        n_samples : int, optional
            Number of samples to analyze in each category
        show_plots : bool, optional
            Whether to display plots (interactive sessions only)
            
        Returns:
        --------
        Dict
            Dictionary with error analysis results
        """
        # Create directory for error analysis
        error_dir = os.path.join(self.output_dir, 'error_analysis')
        
        return analyze_correct_vs_incorrect(
            model, X, y,
            output_dir=error_dir,
            n_samples=n_samples,
            show_plots=show_plots
        )
    
    def save_explainer(self, file_path: str) -> bool:
        """
        Save the SHAP explainer to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the explainer
            
        Returns:
        --------
        bool
            True if save succeeded, False otherwise
        """
        if self.explainer is None:
            logger.error("No explainer to save. Run explain_model() first.")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the explainer
            with open(file_path, 'wb') as f:
                pickle.dump(self.explainer, f)
            
            logger.info(f"Saved explainer to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving explainer to {file_path}: {str(e)}")
            return False
    
    def load_explainer(self, file_path: str) -> bool:
        """
        Load the SHAP explainer from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the saved explainer
            
        Returns:
        --------
        bool
            True if load succeeded, False otherwise
        """
        try:
            # Load the explainer
            with open(file_path, 'rb') as f:
                self.explainer = pickle.load(f)
            
            logger.info(f"Loaded explainer from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading explainer from {file_path}: {str(e)}")
            return False            #
