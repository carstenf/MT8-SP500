"""
Error Analyzer Module

This module handles analysis of model errors and comparison of correct vs. incorrect predictions.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
from typing import Dict, List, Optional, Any

from .shap_explainer import generate_shap_values, create_force_plot

# Set up logging
logger = logging.getLogger(__name__)

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
        Random state for reproducibility
    show_plots : bool, optional
        Whether to display plots
        
    Returns:
    --------
    Dict
        Dictionary with analysis results
    """
    logger.info("Analyzing correct vs. incorrect predictions...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Identify correct and incorrect predictions
        correct_mask = (y_pred == y)
        incorrect_mask = ~correct_mask
        
        # Get indices for each category
        correct_indices = np.where(correct_mask)[0]
        incorrect_indices = np.where(incorrect_mask)[0]
        
        # Sample instances
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
            
        # Create subsets
        X_correct = X.iloc[correct_samples]
        y_correct = y.iloc[correct_samples]
        y_pred_correct = y_pred[correct_samples]
        
        X_incorrect = X.iloc[incorrect_samples]
        y_incorrect = y.iloc[incorrect_samples]
        y_pred_incorrect = y_pred[incorrect_samples]
        
        # Calculate SHAP values for both groups
        shap_correct = generate_shap_values(model, X_correct)
        shap_incorrect = generate_shap_values(model, X_incorrect)
        
        if not shap_correct or not shap_incorrect:
            logger.error("Failed to calculate SHAP values for analysis")
            return {}
            
        # Create summary plots
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
        plt.close()
            
        # Calculate feature importance differences
        correct_importance = pd.DataFrame({
            'feature': X_correct.columns,
            'importance': np.abs(shap_correct['shap_values']).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        incorrect_importance = pd.DataFrame({
            'feature': X_incorrect.columns,
            'importance': np.abs(shap_incorrect['shap_values']).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Find differences
        importance_diff = pd.merge(
            correct_importance, incorrect_importance,
            on='feature', suffixes=('_correct', '_incorrect')
        )
        
        importance_diff['diff'] = importance_diff['importance_incorrect'] - importance_diff['importance_correct']
        importance_diff['ratio'] = importance_diff['importance_incorrect'] / importance_diff['importance_correct']
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
        plt.close()
        
        # Create example explanations
        correct_examples = create_example_explanations(
            model, X_correct, y_correct, y_pred_correct,
            shap_correct, os.path.join(output_dir, 'correct'),
            show_plots=show_plots
        )
        
        incorrect_examples = create_example_explanations(
            model, X_incorrect, y_incorrect, y_pred_incorrect,
            shap_incorrect, os.path.join(output_dir, 'incorrect'),
            show_plots=show_plots
        )
        
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
        if plt.get_fignums():
            plt.close()
        return {}

def create_example_explanations(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    shap_result: Dict,
    output_dir: str,
    n_examples: int = 5,
    show_plots: bool = False
) -> List[Dict]:
    """
    Create detailed explanations for example predictions.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : pd.DataFrame
        Feature matrix
    y_true : pd.Series
        True target values
    y_pred : np.ndarray
        Predicted values
    shap_result : Dict
        SHAP values and explainer
    output_dir : str
        Directory to save explanations
    n_examples : int, optional
        Number of examples to explain
    show_plots : bool, optional
        Whether to display plots
        
    Returns:
    --------
    List[Dict]
        List of example explanations
    """
    examples = []
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        for i in range(min(n_examples, len(X))):
            instance = X.iloc[i:i+1]
            instance_shap = shap_result['shap_values'][i:i+1]
            
            # Create force plot
            force_path = create_force_plot(
                shap_result['explainer'],
                instance_shap,
                instance,
                i,
                output_dir,
                plot_format='html',
                show_plots=show_plots
            )
            
            # Determine feature contributions
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
            
            examples.append({
                'index': i,
                'true_value': int(y_true.iloc[i]),
                'predicted_value': int(y_pred[i]),
                'force_plot': force_path,
                'top_features': feature_contributions[:5]
            })
            
        return examples
        
    except Exception as e:
        logger.error(f"Error creating example explanations: {str(e)}")
        return []

def create_error_analysis_report(
    analysis_results: Dict,
    output_dir: str = 'results/explanation/error_analysis'
) -> Dict:
    """
    Create a comprehensive error analysis report.
    
    Parameters:
    -----------
    analysis_results : Dict
        Results from analyze_correct_vs_incorrect
    output_dir : str, optional
        Directory to save the report
        
    Returns:
    --------
    Dict
        Dictionary with error analysis report
    """
    if not analysis_results:
        return {}
    
    try:
        # Extract key information
        sample_counts = analysis_results.get('sample_counts', {})
        importance_diff = analysis_results.get('feature_importance_diff', [])
        
        # Calculate error patterns
        accuracy = sample_counts.get('accuracy', 0)
        error_rate = 1 - accuracy
        
        # Create report
        report = {
            'performance_metrics': {
                'accuracy': float(accuracy),
                'error_rate': float(error_rate),
                'total_samples': sample_counts.get('total', 0),
                'correct_predictions': sample_counts.get('correct', 0),
                'incorrect_predictions': sample_counts.get('incorrect', 0)
            },
            'feature_analysis': {
                'most_important_correct': importance_diff[:3],
                'most_important_incorrect': sorted(
                    importance_diff, 
                    key=lambda x: x.get('importance_incorrect', 0),
                    reverse=True
                )[:3]
            },
            'visualization_paths': analysis_results.get('summary_plots', {}),
            'recommendations': []
        }
        
        # Generate recommendations
        if error_rate > 0.3:
            report['recommendations'].append(
                "High error rate detected. Consider model retraining or feature engineering."
            )
        
        # Look for features with large importance differences
        large_diffs = [x for x in importance_diff if abs(x.get('diff', 0)) > 0.1]
        if large_diffs:
            report['recommendations'].append(
                f"Large importance differences found for features: "
                f"{', '.join(x['feature'] for x in large_diffs[:3])}. "
                "Consider focusing on these features for improvement."
            )
        
        # Save report
        report_path = os.path.join(output_dir, 'error_analysis_report.json')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
        
    except Exception as e:
        logger.error(f"Error creating error analysis report: {str(e)}")
        return {}
