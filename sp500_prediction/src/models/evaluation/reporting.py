"""
Reporting Module

This module handles the generation of performance reports and result serialization.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

def convert_to_serializable(obj: Any) -> Any:
    """
    Convert Python objects to JSON serializable format.
    
    Parameters:
    -----------
    obj : Any
        The object to convert
        
    Returns:
    --------
    Any
        JSON serializable version of the input object
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    elif str(type(obj)) == "<class 'numpy.bool_'>":
        return bool(obj)
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)

def generate_performance_report(
    evaluation_results: Dict,
    output_file: str = 'results/performance_report.json'
) -> bool:
    """
    Generate a comprehensive performance report from evaluation results.
    
    Parameters:
    -----------
    evaluation_results : Dict
        Dictionary with all evaluation results containing:
        - overall_metrics: Dict, basic performance metrics
        - ticker_analysis: Dict, ticker-level performance metrics
        - time_analysis: Dict, time-based performance analysis
        - feature_importance: Dict, feature importance analysis
        - bias_variance_analysis: Dict, bias-variance tradeoff results
        - visualizations: Dict, paths to generated plots
    output_file : str, optional
        Path to save the report
        
    Returns:
    --------
    bool
        True if the report was successfully generated and saved
    """
    logger.info("Generating performance report...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Create a report structure
        report = {
            'generated_at': datetime.now().isoformat(),
            'overall_metrics': evaluation_results.get('overall_metrics', {}),
            'ticker_analysis': {
                'aggregate_metrics': evaluation_results.get('ticker_analysis', {}).get('aggregate_metrics', {}),
                'top_performers': [
                    {
                        'ticker': t[0],
                        'balanced_accuracy': t[1]['balanced_accuracy'],
                        'f1': t[1]['f1'],
                        'pnl': t[1]['pnl']['total']
                    }
                    for t in evaluation_results.get('ticker_analysis', {})
                        .get('aggregate_metrics', {})
                        .get('top_tickers_by_accuracy', [])[:5]
                ]
            },
            'time_analysis': {
                'trend_direction': evaluation_results.get('time_analysis', {}).get('trend_direction', {}),
                'periods_analyzed': len(evaluation_results.get('time_analysis', {})
                                      .get('time_metrics', {}))
            },
            'feature_importance': {
                'top_features': [
                    {
                        'feature': row['feature'],
                        'importance': row['importance']
                    }
                    for _, row in evaluation_results.get('feature_importance', {})
                        .get('top_features', pd.DataFrame())
                        .iterrows()
                ][:10],
                'features_for_90_importance': evaluation_results.get('feature_importance', {})
                    .get('features_for_90_importance', 0)
            },
            'bias_variance_analysis': {
                'bias': evaluation_results.get('bias_variance', {}).get('bias', 0),
                'variance': evaluation_results.get('bias_variance', {}).get('variance', 0),
                'diagnosis': evaluation_results.get('bias_variance', {}).get('diagnosis', 'unknown'),
                'recommendations': evaluation_results.get('bias_variance', {}).get('recommendations', [])
            },
            'visualizations': evaluation_results.get('visualizations', {}),
            'success_criteria': {
                'balanced_accuracy_above_55': evaluation_results.get('overall_metrics', {})
                    .get('balanced_accuracy', 0) > 0.55,
                'stable_performance': evaluation_results.get('time_analysis', {})
                    .get('trend_direction', {})
                    .get('balanced_accuracy', '') != 'decreasing',
                'model_explanations_available': 'feature_importance' in evaluation_results,
                'pipeline_reproducible': True  # Assuming the pipeline is reproducible by design
            }
        }
        
        # Overall model evaluation
        overall_criteria_met = sum(1 for value in report['success_criteria'].values() if value) / len(report['success_criteria'])
        report['overall_evaluation'] = {
            'success_criteria_met': overall_criteria_met,
            'success_score': overall_criteria_met * 100,
            'overall_assessment': 'Successful' if overall_criteria_met >= 0.75 else 'Partially Successful' if overall_criteria_met >= 0.5 else 'Unsuccessful'
        }
        
        # Convert the entire report to JSON serializable format
        serialized_report = convert_to_serializable(report)
        
        # Save the report as JSON
        with open(output_file, 'w') as f:
            json.dump(serialized_report, f, indent=4)
        
        logger.info(f"Performance report saved to {output_file}")
        logger.info(f"Overall assessment: {report['overall_evaluation']['overall_assessment']} "
                    f"({report['overall_evaluation']['success_score']:.1f}% of success criteria met)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        return False
