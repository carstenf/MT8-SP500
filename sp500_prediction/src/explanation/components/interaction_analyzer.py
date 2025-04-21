"""
Interaction Analyzer Module

This module handles analysis of feature interactions using SHAP interaction values.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from typing import Dict, List, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)

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
        Feature matrix
    n_top_features : int, optional
        Number of top features to include in the analysis
    sample_size : int, optional
        Number of instances to sample for interaction calculation
    output_dir : str, optional
        Directory to save the plots
    feature_importances : Dict, optional
        Dictionary with feature importance information
    random_state : int, optional
        Random state for reproducibility
    show_plots : bool, optional
        Whether to display plots
        
    Returns:
    --------
    Dict
        Dictionary with interaction analysis results
    """
    logger.info("Analyzing feature interactions using SHAP...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample data if needed
    if sample_size < len(X):
        X_sample = X.sample(sample_size, random_state=random_state)
    else:
        X_sample = X
    
    # Identify top features
    if feature_importances is not None and 'importance_df' in feature_importances:
        top_features = feature_importances['importance_df'].head(n_top_features)['feature'].tolist()
    else:
        top_features = None
    
    try:
        # Create TreeExplainer for interaction values
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP interaction values
        logger.info("Calculating SHAP interaction values...")
        
        if top_features is not None:
            X_filtered = X_sample[top_features]
            shap_interaction_values = explainer.shap_interaction_values(X_filtered)
        else:
            shap_interaction_values = explainer.shap_interaction_values(X_sample)
        
        # Handle binary classification output
        if isinstance(shap_interaction_values, list):
            shap_interaction_values = shap_interaction_values[1]
        
        # Get feature names
        feature_names = top_features if top_features is not None else X_sample.columns.tolist()
        
        # Calculate average absolute interaction values
        abs_interaction_values = np.abs(shap_interaction_values).mean(axis=0)
        
        # Create interaction summary
        interaction_summary = []
        
        # Convert to DataFrame
        interaction_df = pd.DataFrame(abs_interaction_values, 
                                    index=feature_names, 
                                    columns=feature_names)
        
        # Zero out diagonal for finding top interactions
        np.fill_diagonal(abs_interaction_values, 0)
        
        # Find top interaction pairs
        n_pairs = min(10, len(feature_names) * (len(feature_names) - 1) // 2)
        flat_indices = np.argsort(abs_interaction_values.flatten())[::-1][:n_pairs]
        
        # Convert flat indices to 2D indices
        for flat_idx in flat_indices:
            i, j = np.unravel_index(flat_idx, abs_interaction_values.shape)
            if i < j:  # Avoid duplicates
                interaction_summary.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'interaction_strength': float(abs_interaction_values[i, j])
                })
        
        # Create interaction plots
        interaction_plots = []
        
        # Plot top interactions
        for interaction in interaction_summary[:5]:
            feature1 = interaction['feature1']
            feature2 = interaction['feature2']
            
            plt.figure(figsize=(10, 8))
            
            # Calculate SHAP values
            feature_shap_values = explainer.shap_values(X_sample[feature_names])
            if isinstance(feature_shap_values, list):
                feature_shap_values = feature_shap_values[1]
            
            # Plot dependence with interaction
            shap.dependence_plot(
                feature1,
                feature_shap_values,
                X_sample[feature_names],
                interaction_index=feature2,
                show=False
            )
            
            plt.title(f'Interaction between {feature1} and {feature2}')
            
            # Save plot
            plot_path = os.path.join(output_dir, f'interaction_{feature1}_{feature2}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            
            if show_plots:
                plt.show()
            plt.close()
            
            interaction_plots.append(plot_path)
            logger.info(f"Interaction plot saved to {plot_path}")
        
        # Create interaction matrix heatmap
        plt.figure(figsize=(12, 10))
        mask = np.zeros_like(abs_interaction_values, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        sns.heatmap(abs_interaction_values, mask=mask, 
                   xticklabels=feature_names, yticklabels=feature_names,
                   cmap='viridis')
        
        plt.title('SHAP Interaction Values')
        
        # Save heatmap
        heatmap_path = os.path.join(output_dir, 'interaction_heatmap.png')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        
        if show_plots:
            plt.show()
        plt.close()
        
        interaction_plots.append(heatmap_path)
        
        return {
            'interaction_values': shap_interaction_values,
            'interaction_plots': interaction_plots,
            'interaction_summary': interaction_summary,
            'interaction_matrix': abs_interaction_values.tolist(),
            'feature_names': feature_names
        }
        
    except Exception as e:
        logger.error(f"Error analyzing SHAP interactions: {str(e)}")
        if plt.get_fignums():
            plt.close()
        return {}

def create_interaction_report(
    interaction_results: Dict,
    output_dir: str = 'results/explanation'
) -> Dict:
    """
    Create a summary report of feature interactions.
    
    Parameters:
    -----------
    interaction_results : Dict
        Results from analyze_shap_interactions
    output_dir : str, optional
        Directory to save the report
        
    Returns:
    --------
    Dict
        Dictionary with interaction report information
    """
    if not interaction_results:
        return {}
    
    try:
        # Extract key information
        summary = interaction_results.get('interaction_summary', [])
        feature_names = interaction_results.get('feature_names', [])
        
        # Create summary statistics
        n_features = len(feature_names)
        n_interactions = len(summary)
        
        # Calculate average interaction strength
        interaction_strengths = [item['interaction_strength'] for item in summary]
        avg_interaction = np.mean(interaction_strengths) if interaction_strengths else 0
        
        # Identify strongest interactions
        top_interactions = summary[:3] if len(summary) >= 3 else summary
        
        # Create report
        report = {
            'statistics': {
                'n_features_analyzed': n_features,
                'n_interactions_found': n_interactions,
                'average_interaction_strength': float(avg_interaction)
            },
            'top_interactions': top_interactions,
            'visualization_paths': interaction_results.get('interaction_plots', []),
            'recommendations': []
        }
        
        # Generate recommendations
        if avg_interaction > 0.1:  # Arbitrary threshold
            report['recommendations'].append(
                "Strong feature interactions detected. Consider creating interaction features."
            )
        if n_interactions > 10:
            report['recommendations'].append(
                "Many significant interactions found. Consider feature selection or dimensionality reduction."
            )
        
        # Save report
        report_path = os.path.join(output_dir, 'interaction_report.json')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
        
    except Exception as e:
        logger.error(f"Error creating interaction report: {str(e)}")
        return {}
