"""
Case Studies Module

This module handles creation of detailed explanations for specific stock predictions.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from .shap_explainer import generate_shap_values, create_force_plot

# Set up logging
logger = logging.getLogger(__name__)

def create_explanation_for_stock(
    model: Any,
    X: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
    output_dir: str = 'results/explanation/case_studies',
    shap_values: np.ndarray = None,
    explainer: Any = None,
    X_sample: pd.DataFrame = None,
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
        Pre-computed SHAP values
    explainer : Any, optional
        Pre-computed SHAP explainer
    X_sample : pd.DataFrame, optional
        Pre-computed sample data used for SHAP values
    show_plots : bool, optional
        Whether to display plots
        
    Returns:
    --------
    Dict
        Dictionary with explanation results
    """
    logger.info(f"Creating explanation for {ticker} on {date}...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Find specific instance
        if isinstance(X.index, pd.MultiIndex):
            try:
                instance = X.loc[(ticker, date)]
                instance_df = pd.DataFrame([instance])
            except KeyError:
                logger.error(f"No data found for {ticker} on {date}")
                return {}
        else:
            logger.error("Input data must have multi-index (ticker, date)")
            return {}
        
        # Make prediction
        prediction = model.predict_proba(instance_df)[0, 1]
        prediction_class = 1 if prediction >= 0.5 else 0
        prediction_label = "UP" if prediction_class == 1 else "DOWN"
        
        # Calculate SHAP values if not provided
        if shap_values is None or explainer is None:
            instance_result = generate_shap_values(model, instance_df)
            if not instance_result:
                return {
                    'prediction': prediction,
                    'prediction_class': prediction_class,
                    'prediction_label': prediction_label
                }
            local_shap_values = instance_result['shap_values']
            local_explainer = instance_result['explainer']
        else:
            # Find instance in pre-computed values
            try:
                if isinstance(X.index, pd.MultiIndex):
                    if X_sample is not None:
                        sample_loc = np.where((X_sample.index.get_level_values(0) == ticker) & 
                                            (X_sample.index.get_level_values(1) == date))[0]
                    else:
                        instance_result = generate_shap_values(model, instance_df)
                        local_shap_values = instance_result['shap_values']
                        local_explainer = instance_result['explainer']
                        sample_loc = []
                    
                    if len(sample_loc) > 0:
                        instance_idx = sample_loc[0]
                        local_shap_values = np.array([shap_values[instance_idx]])
                    else:
                        instance_result = generate_shap_values(model, instance_df)
                        local_shap_values = instance_result['shap_values']
                        local_explainer = instance_result['explainer']
                else:
                    logger.error("Cannot find instance in SHAP values")
                    return {}
            except Exception as e:
                logger.error(f"Error finding instance in SHAP values: {str(e)}")
                return {}
            local_explainer = explainer
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        import shap
        explanation = shap.Explanation(
            values=local_shap_values[0],
            base_values=local_explainer.expected_value[1] if isinstance(local_explainer.expected_value, list) else local_explainer.expected_value,
            data=instance_df.iloc[0],
            feature_names=instance_df.columns.tolist()
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        
        waterfall_path = os.path.join(output_dir, f'{ticker}_{date.strftime("%Y-%m-%d")}_waterfall.png')
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=300)
        
        if show_plots:
            plt.show()
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
                'value': float(instance_df.iloc[0, i]),
                'contribution': float(local_shap_values[0, i]),
                'abs_contribution': float(abs(local_shap_values[0, i]))
            })
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        # Identify driving factors
        top_positive = [fc for fc in feature_contributions if fc['contribution'] > 0][:3]
        top_negative = [fc for fc in feature_contributions if fc['contribution'] < 0][:3]
        
        return {
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
        
    except Exception as e:
        logger.error(f"Error creating explanation for {ticker} on {date}: {str(e)}")
        if plt.get_fignums():
            plt.close()
        return {}

def create_case_study_report(
    model: Any,
    X: pd.DataFrame,
    tickers: List[str],
    dates: List[pd.Timestamp],
    output_dir: str = 'results/explanation/case_studies',
    show_plots: bool = False
) -> Dict:
    """
    Create a comprehensive case study report for multiple stocks.
    
    Parameters:
    -----------
    model : Any
        Trained model to explain
    X : pd.DataFrame
        Feature matrix with multi-index (ticker, date)
    tickers : List[str]
        List of stock tickers to analyze
    dates : List[pd.Timestamp]
        List of dates to analyze
    output_dir : str, optional
        Directory to save the report
    show_plots : bool, optional
        Whether to display plots
        
    Returns:
    --------
    Dict
        Dictionary with case study report
    """
    logger.info("Creating case study report...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        case_studies = []
        
        # Calculate SHAP values for all instances
        all_instances = []
        for ticker, date in zip(tickers, dates):
            try:
                instance = X.loc[(ticker, date)]
                all_instances.append(instance)
            except KeyError:
                logger.warning(f"No data found for {ticker} on {date}")
                continue
        
        if not all_instances:
            logger.error("No valid instances found for case studies")
            return {}
        
        instances_df = pd.DataFrame(all_instances)
        shap_result = generate_shap_values(model, instances_df)
        
        if not shap_result:
            logger.error("Failed to generate SHAP values for case studies")
            return {}
        
        # Create explanations for each instance
        for i, (ticker, date) in enumerate(zip(tickers, dates)):
            explanation = create_explanation_for_stock(
                model, X, ticker, date,
                output_dir=os.path.join(output_dir, ticker),
                shap_values=shap_result['shap_values'],
                explainer=shap_result['explainer'],
                X_sample=shap_result['X_sample'],
                show_plots=show_plots
            )
            
            if explanation:
                case_studies.append(explanation)
        
        # Create summary report
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_cases': len(case_studies),
            'cases': case_studies,
            'summary': {
                'accuracy': sum(1 for case in case_studies 
                              if case.get('prediction', 0.5) >= 0.5) / len(case_studies),
                'avg_confidence': np.mean([case.get('prediction', 0.5) 
                                         for case in case_studies])
            },
            'common_factors': analyze_common_factors(case_studies)
        }
        
        # Save report
        report_path = os.path.join(output_dir, 'case_study_report.json')
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
        
    except Exception as e:
        logger.error(f"Error creating case study report: {str(e)}")
        return {}

def analyze_common_factors(case_studies: List[Dict]) -> Dict:
    """
    Analyze common factors across multiple case studies.
    
    Parameters:
    -----------
    case_studies : List[Dict]
        List of case study results
        
    Returns:
    --------
    Dict
        Dictionary with common factor analysis
    """
    try:
        # Collect all features and their contributions
        all_features = {}
        for case in case_studies:
            for contrib in case.get('feature_contributions', []):
                feature = contrib['feature']
                if feature not in all_features:
                    all_features[feature] = {
                        'positive_count': 0,
                        'negative_count': 0,
                        'total_contribution': 0,
                        'abs_contribution': 0
                    }
                
                contribution = contrib['contribution']
                all_features[feature]['total_contribution'] += contribution
                all_features[feature]['abs_contribution'] += abs(contribution)
                
                if contribution > 0:
                    all_features[feature]['positive_count'] += 1
                else:
                    all_features[feature]['negative_count'] += 1
        
        # Convert to list and sort by absolute contribution
        feature_summary = []
        for feature, stats in all_features.items():
            feature_summary.append({
                'feature': feature,
                'avg_contribution': stats['total_contribution'] / len(case_studies),
                'avg_abs_contribution': stats['abs_contribution'] / len(case_studies),
                'positive_ratio': stats['positive_count'] / len(case_studies),
                'negative_ratio': stats['negative_count'] / len(case_studies)
            })
        
        feature_summary.sort(key=lambda x: x['avg_abs_contribution'], reverse=True)
        
        return {
            'most_influential_features': feature_summary[:5],
            'most_consistent_features': sorted(
                feature_summary,
                key=lambda x: max(x['positive_ratio'], x['negative_ratio']),
                reverse=True
            )[:5]
        }
        
    except Exception as e:
        logger.error(f"Error analyzing common factors: {str(e)}")
        return {}
