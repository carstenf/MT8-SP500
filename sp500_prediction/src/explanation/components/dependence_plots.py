"""
Dependence Plots Module

This module handles creation of Partial Dependence Plots (PDPs) and
Individual Conditional Expectation (ICE) plots.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import os
from sklearn.inspection import partial_dependence

# Set up logging
logger = logging.getLogger(__name__)

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
    Create Partial Dependence Plots (PDPs).
    
    Parameters:
    -----------
    model : Any
        Trained model for which to create PDPs
    X : pd.DataFrame
        Feature matrix
    features : List[str], optional
        Specific features to compute PDPs for
    n_top_features : int, optional
        Number of top features to include if features is None
    output_dir : str, optional
        Directory to save the plots
    feature_importances : Dict, optional
        Dictionary with feature importance information
    grid_resolution : int, optional
        Resolution of the grid for PDP calculation
    n_jobs : int, optional
        Number of jobs for parallel computation
    show_plots : bool, optional
        Whether to display plots
        
    Returns:
    --------
    Dict
        Dictionary with paths to saved plots and PDP values
    """
    logger.info("Creating partial dependence plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Select features to analyze
    if features is None:
        if feature_importances is not None and 'importance_df' in feature_importances:
            features = feature_importances['importance_df'].head(n_top_features)['feature'].tolist()
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:n_top_features]
            features = [X.columns[i] for i in top_indices]
        else:
            features = X.columns[:n_top_features].tolist()
    
    try:
        result = {'individual_plots': [], 'pdp_values': {}}
        
        for feature in features:
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
            pd_feature_values = pd_results.grid_values[0]
            pd_feature_predictions = pd_results.average.squeeze()
            
            # Store values
            result['pdp_values'][feature] = {
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
            
            # Add rug plot
            plt.plot(X[feature].values, np.zeros_like(X[feature].values) - 0.02, 
                    '|k', ms=15, alpha=0.2)
            
            # Save plot
            plot_path = os.path.join(output_dir, f'pdp_{feature}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            
            if show_plots:
                plt.show()
            plt.close()
            
            result['individual_plots'].append(plot_path)
            logger.info(f"PDP for {feature} saved to {plot_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating partial dependence plots: {str(e)}")
        if plt.get_fignums():
            plt.close()
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
    Create Individual Conditional Expectation (ICE) plots.
    
    Parameters:
    -----------
    model : Any
        Trained model for which to create ICE plots
    X : pd.DataFrame
        Feature matrix
    features : List[str], optional
        Specific features to compute ICE for
    n_top_features : int, optional
        Number of top features to include if features is None
    n_samples : int, optional
        Number of instances to sample for ICE plots
    output_dir : str, optional
        Directory to save the plots
    feature_importances : Dict, optional
        Dictionary with feature importance information
    grid_resolution : int, optional
        Resolution of the grid for ICE calculation
    random_state : int, optional
        Random state for reproducibility
    show_plots : bool, optional
        Whether to display plots
        
    Returns:
    --------
    Dict
        Dictionary with paths to ICE plots
    """
    logger.info("Creating individual conditional expectation (ICE) plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample instances
    if n_samples < len(X):
        X_sampled = X.sample(n_samples, random_state=random_state)
    else:
        X_sampled = X
    
    # Select features to analyze
    if features is None:
        if feature_importances is not None and 'importance_df' in feature_importances:
            features = feature_importances['importance_df'].head(n_top_features)['feature'].tolist()
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:n_top_features]
            features = [X.columns[i] for i in top_indices]
        else:
            features = X.columns[:n_top_features].tolist()
    
    try:
        ice_plots = []
        
        for feature in features:
            if feature not in X.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
            
            # Calculate ICE and PDP
            pd_results = partial_dependence(
                model, X_sampled, [feature], 
                grid_resolution=grid_resolution,
                kind='both'
            )
            
            # Extract values
            feature_values = pd_results.grid_values[0]
            individual_predictions = pd_results.individual.squeeze()
            average_predictions = pd_results.average.squeeze()
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot individual ICE lines
            for i in range(len(individual_predictions)):
                plt.plot(feature_values, individual_predictions[i], 'C0-', alpha=0.1)
            
            # Plot average (PDP)
            plt.plot(feature_values, average_predictions, 'r-', 
                    linewidth=2, label='PDP (average)')
            
            plt.xlabel(feature)
            plt.ylabel('Prediction')
            plt.title(f'Individual Conditional Expectation Plots for {feature}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add rug plot
            plt.plot(X_sampled[feature].values, 
                    np.zeros_like(X_sampled[feature].values) - 0.02, 
                    '|k', ms=15, alpha=0.2)
            
            # Save plot
            plot_path = os.path.join(output_dir, f'ice_{feature}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            
            if show_plots:
                plt.show()
            plt.close()
            
            ice_plots.append(plot_path)
            logger.info(f"ICE plot for {feature} saved to {plot_path}")
        
        return {'ice_plots': ice_plots}
        
    except Exception as e:
        logger.error(f"Error creating ICE plots: {str(e)}")
        if plt.get_fignums():
            plt.close()
        return {}
