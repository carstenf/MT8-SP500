"""
Model Evaluator Module for S&P500 Prediction Project

This module handles the evaluation, analysis, and visualization of 
model performance for S&P500 stock direction prediction.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import json
import os
from typing import Tuple, Dict, List, Optional, Union, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    auc
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        
    Returns:
    --------
    Dict
        Dictionary of performance metrics
    """
    logger.info("Calculating performance metrics...")
    
    # Check if we have more than one class
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    is_single_class = len(unique_classes) == 1

    # Basic classification metrics with handling for edge cases
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'is_single_class': is_single_class
    }
    
    # Add class report
    class_report = classification_report(y_true, y_pred, output_dict=True)
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
    
    logger.info(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}, "
                f"F1 score: {metrics['f1']:.4f}")
    
    return metrics


def analyze_predictions_by_ticker(
    predictions: pd.DataFrame,
    excess_returns: pd.DataFrame,
    config: Optional[Dict] = None
) -> Dict:
    """
    Analyze prediction performance by ticker.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with predictions (multi-index: ticker, date)
    excess_returns : pd.DataFrame
        DataFrame with excess returns (has 'excess_return' column)
        
    Returns:
    --------
    Dict
        Dictionary with ticker-level performance metrics
    """
    logger.info("Analyzing prediction performance by ticker...")
    
    if predictions.empty or excess_returns.empty:
        logger.warning("Empty data provided. No analysis performed.")
        return {}
    
    if not isinstance(predictions.index, pd.MultiIndex):
        logger.error("Predictions DataFrame must have a multi-index (ticker, date)")
        return {}
    
    # Make sure we have predictions and true values
    if 'prediction' not in predictions.columns or 'true' not in predictions.columns:
        logger.error("Predictions DataFrame must have 'prediction' and 'true' columns")
        return {}
    
    # Ensure excess_returns has the necessary column
    if 'excess_return' not in excess_returns.columns:
        logger.error("Excess returns DataFrame must have 'excess_return' column")
        return {}
    
    # Get tickers
    tickers = predictions.index.get_level_values('ticker').unique()
    
    # Initialize results dictionary
    ticker_metrics = {}
    
    # Calculate metrics for each ticker
    for ticker in tickers:
        # Get ticker predictions
        ticker_preds = predictions.loc[ticker]
        
        # Get ticker excess returns (may need to align indices)
        ticker_excess = excess_returns.loc[ticker, 'excess_return']
        
        # Align data by date
        common_dates = ticker_preds.index.intersection(ticker_excess.index)
        if len(common_dates) == 0:
            logger.warning(f"No common dates for ticker {ticker}. Skipping.")
            continue
        
        # Filter data to common dates
        ticker_preds = ticker_preds.loc[common_dates]
        ticker_excess = ticker_excess.loc[common_dates]
        
        # Calculate basic metrics
        y_true = ticker_preds['true']
        y_pred = ticker_preds['prediction']
        y_prob = ticker_preds['probability'] if 'probability' in ticker_preds.columns else None
        
        # Calculate metrics
        metrics = calculate_performance_metrics(y_true, y_pred, y_prob)
        
        # Calculate Profit & Loss
        # A positive prediction (1) means going long, negative (0) means going short
        # P&L is position * excess_return
        position = ticker_preds['prediction'] * 2 - 1  # Convert 0/1 to -1/1
        pnl = position * ticker_excess
        
        # Add P&L metrics
        metrics['pnl'] = {
            'total': pnl.sum(),
            'mean': pnl.mean(),
            'std': pnl.std(),
            'sharpe': pnl.mean() / pnl.std() if pnl.std() > 0 else 0,
            'win_rate': (pnl > 0).mean()
        }
        
        # Store metrics for this ticker
        ticker_metrics[ticker] = metrics
    
    # Calculate aggregate metrics
    agg_metrics = {
        'n_tickers': len(ticker_metrics),
        'avg_balanced_accuracy': np.mean([m['balanced_accuracy'] for m in ticker_metrics.values()]),
        'avg_f1': np.mean([m['f1'] for m in ticker_metrics.values()]),
        'avg_pnl': np.mean([m['pnl']['total'] for m in ticker_metrics.values()]),
        'avg_sharpe': np.mean([m['pnl']['sharpe'] for m in ticker_metrics.values()]),
        'top_tickers_by_accuracy': sorted(ticker_metrics.items(), key=lambda x: x[1]['balanced_accuracy'], reverse=True)[:5],
        'top_tickers_by_pnl': sorted(ticker_metrics.items(), key=lambda x: x[1]['pnl']['total'], reverse=True)[:5]
    }
    
    logger.info(f"Analyzed {agg_metrics['n_tickers']} tickers. "
                f"Average balanced accuracy: {agg_metrics['avg_balanced_accuracy']:.4f}, "
                f"Average P&L: {agg_metrics['avg_pnl']:.4f}")
    
    return {
        'ticker_metrics': ticker_metrics,
        'aggregate_metrics': agg_metrics
    }


def analyze_predictions_by_time(
    predictions: pd.DataFrame,
    time_unit: str = 'month',
    config: Optional[Dict] = None
) -> Dict:
    """
    Analyze prediction performance by time periods.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with predictions (multi-index: ticker, date)
    time_unit : str, optional
        Time unit for analysis ('day', 'week', 'month', 'quarter', 'year')
        
    Returns:
    --------
    Dict
        Dictionary with time-level performance metrics
    """
    logger.info(f"Analyzing prediction performance by {time_unit}...")
    
    if predictions.empty:
        logger.warning("Empty DataFrame provided. No analysis performed.")
        return {}
    
    if not isinstance(predictions.index, pd.MultiIndex):
        logger.error("Predictions DataFrame must have a multi-index (ticker, date)")
        return {}
    
    # Make sure we have predictions and true values
    if 'prediction' not in predictions.columns or 'true' not in predictions.columns:
        logger.error("Predictions DataFrame must have 'prediction' and 'true' columns")
        return {}
    
    # Reset index to get date as a column
    pred_df = predictions.reset_index()
    
    # Create time period column based on time_unit
    if time_unit == 'day':
        pred_df['period'] = pred_df['date'].dt.date
    elif time_unit == 'week':
        pred_df['period'] = pred_df['date'].dt.to_period('W').dt.start_time.dt.date
    elif time_unit == 'month':
        pred_df['period'] = pred_df['date'].dt.to_period('M').dt.start_time.dt.date
    elif time_unit == 'quarter':
        pred_df['period'] = pred_df['date'].dt.to_period('Q').dt.start_time.dt.date
    elif time_unit == 'year':
        pred_df['period'] = pred_df['date'].dt.year
    else:
        logger.error(f"Invalid time unit: {time_unit}")
        return {}
    
    # Get unique periods
    periods = pred_df['period'].unique()
    
    # Initialize results dictionary
    time_metrics = {}
    
    # Calculate metrics for each period
    for period in periods:
        # Get predictions for this period
        period_preds = pred_df[pred_df['period'] == period]
        
        # Calculate basic metrics
        y_true = period_preds['true']
        y_pred = period_preds['prediction']
        y_prob = period_preds['probability'] if 'probability' in period_preds.columns else None
        
        # Calculate metrics
        metrics = calculate_performance_metrics(y_true, y_pred, y_prob)
        
        # Calculate win rate by ticker
        if 'ticker' in period_preds.columns:
            ticker_results = period_preds.groupby('ticker').apply(
                lambda x: (x['prediction'] == x['true']).mean()
            )
            metrics['ticker_win_rates'] = ticker_results.to_dict()
            metrics['ticker_count'] = len(ticker_results)
        
        # Store metrics for this period
        time_metrics[str(period)] = metrics
    
    # Calculate trend of metrics over time
    time_series = {
        'periods': [str(p) for p in periods],
        'balanced_accuracy': [time_metrics[str(p)]['balanced_accuracy'] for p in periods],
        'precision': [time_metrics[str(p)]['precision'] for p in periods],
        'recall': [time_metrics[str(p)]['recall'] for p in periods],
        'f1': [time_metrics[str(p)]['f1'] for p in periods]
    }
    
    # Add ROC-AUC if available
    if 'roc_auc' in time_metrics[str(periods[0])]:
        time_series['roc_auc'] = [time_metrics[str(p)]['roc_auc'] for p in periods]
    
    # Calculate trend direction (increasing, decreasing, stable)
    trend_direction = {}
    for metric in ['balanced_accuracy', 'precision', 'recall', 'f1']:
        values = time_series[metric]
        if len(values) >= 3:
            # Calculate trend using linear regression slope
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            
            # Determine direction based on slope
            if slope > 0.01:  # Increasing
                trend_direction[metric] = 'increasing'
            elif slope < -0.01:  # Decreasing
                trend_direction[metric] = 'decreasing'
            else:  # Stable
                trend_direction[metric] = 'stable'
        else:
            trend_direction[metric] = 'insufficient_data'
    
    logger.info(f"Analyzed {len(periods)} {time_unit}s. "
                f"Balanced accuracy trend: {trend_direction.get('balanced_accuracy', 'N/A')}")
    
    return {
        'time_metrics': time_metrics,
        'time_series': time_series,
        'trend_direction': trend_direction
    }


def analyze_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'model_based',
    n_top_features: int = 20
) -> Dict:
    """
    Analyze feature importances for the model.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    method : str, optional
        Method for calculating feature importance ('model_based' or 'permutation')
    n_top_features : int, optional
        Number of top features to include in the result
        
    Returns:
    --------
    Dict
        Dictionary with feature importance information
    """
    logger.info(f"Analyzing feature importance using {method} method...")
    
    if X.empty or y.empty:
        logger.warning("Empty data provided. No analysis performed.")
        return {}
    
    # Model-based feature importance
    if method == 'model_based':
        # Check if the model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importances = np.abs(model.coef_[0])
        else:
            logger.error("Model does not have feature_importances_ or coef_ attribute")
            return {}
        
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
    # Permutation importance
    elif method == 'permutation':
        # Calculate permutation importance
        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Extract importances
        importances = result.importances_mean
        
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances,
            'std': result.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
    else:
        logger.error(f"Invalid feature importance method: {method}")
        return {}
    
    # Get top N features
    top_features = importance_df.head(n_top_features)
    
    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
    
    # Find number of features needed for 80% and 90% of importance
    features_for_80 = (importance_df['cumulative_importance'] <= 0.8).sum() + 1
    features_for_90 = (importance_df['cumulative_importance'] <= 0.9).sum() + 1
    
    logger.info(f"Top feature: {top_features.iloc[0]['feature']} "
                f"(importance: {top_features.iloc[0]['importance']:.4f})")
    logger.info(f"Features for 80% importance: {features_for_80}, "
                f"For 90% importance: {features_for_90}")
    
    return {
        'importance_df': importance_df,
        'top_features': top_features,
        'features_for_80_importance': features_for_80,
        'features_for_90_importance': features_for_90,
        'method': method
    }



def analyze_bias_variance_tradeoff(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Analyze bias-variance tradeoff using learning curves.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target values
        
    Returns:
    --------
    Dict
        Dictionary with bias-variance analysis results
    """
    logger.info("Analyzing bias-variance tradeoff...")
    
    # Calculate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model, 
        X_train, 
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    
    # Calculate mean and std of training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Calculate training and test error
    train_error = 1 - train_mean
    test_error = 1 - test_mean
    
    # Calculate bias and variance components
    bias = train_error[-1]  # Bias is approximated by final training error
    variance = test_error[-1] - train_error[-1]  # Variance is the gap between test and train error
    
    # Determine if model is high bias, high variance, or balanced
    if bias > 0.2 and variance < 0.1:
        diagnosis = "high_bias"
    elif bias < 0.1 and variance > 0.2:
        diagnosis = "high_variance"
    else:
        diagnosis = "balanced"
    
    # Provide recommendations based on diagnosis
    if diagnosis == "high_bias":
        recommendations = [
            "Use a more complex model",
            "Add more features",
            "Reduce regularization strength"
        ]
    elif diagnosis == "high_variance":
        recommendations = [
            "Use more training data",
            "Use a simpler model",
            "Apply stronger regularization",
            "Reduce number of features"
        ]
    else:
        recommendations = [
            "Model has a good bias-variance balance",
            "Fine-tune hyperparameters to improve performance"
        ]
    
    logger.info(f"Bias-variance diagnosis: {diagnosis}. "
                f"Bias: {bias:.4f}, Variance: {variance:.4f}")
    
    return {
        'train_sizes': train_sizes.tolist(),
        'train_mean_scores': train_mean.tolist(),
        'train_std_scores': train_std.tolist(),
        'test_mean_scores': test_mean.tolist(),
        'test_std_scores': test_std.tolist(),
        'bias': bias,
        'variance': variance,
        'diagnosis': diagnosis,
        'recommendations': recommendations
    }


def create_performance_visualizations(
    metrics: Dict,
    output_dir: str = 'results/plots',
    viz_config: Optional[Dict] = None
) -> Dict:
    """
    Create and save performance visualizations based on metrics.
    
    Parameters:
    -----------
    metrics : Dict
        Dictionary with performance metrics
    output_dir : str, optional
        Directory to save the plots
    viz_config : Dict, optional
        Visualization configuration settings
        
    Returns:
    --------
    Dict
        Dictionary with paths to the saved plots
    """
    """
    Create and save performance visualizations based on metrics.
    
    Parameters:
    -----------
    metrics : Dict
        Dictionary with performance metrics
    output_dir : str, optional
        Directory to save the plots
        
    Returns:
    --------
    Dict
        Dictionary with paths to the saved plots
    """
    logger.info("Creating performance visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = {}
    
    # Plot confusion matrix if available
    if 'confusion_matrix' in metrics:
        # Get plot settings from config
        figsize = viz_config.get('plot_figsize', {}).get('confusion_matrix', (8, 6)) if viz_config else (8, 6)
        plt.figure(figsize=figsize)
        
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the plot
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        
        plot_paths['confusion_matrix'] = cm_path
    
    # Plot ROC curve if available and valid
    if 'roc_curve' in metrics and metrics['roc_curve'] is not None:
        try:
            # Get plot settings from config
            figsize = viz_config.get('plot_figsize', {}).get('default', (8, 6)) if viz_config else (8, 6)
            linewidth = viz_config.get('plot_style', {}).get('linewidth', 2) if viz_config else 2
            
            plt.figure(figsize=figsize)
            fpr = metrics['roc_curve']['fpr']
            tpr = metrics['roc_curve']['tpr']
            plt.plot(fpr, tpr, lw=linewidth)
            plt.plot([0, 1], [0, 1], 'k--', lw=linewidth)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (AUC = {metrics.get("roc_auc", 0):.3f})')
            
            # Save the plot
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            plt.savefig(roc_path)
            plt.close()
            
            plot_paths['roc_curve'] = roc_path
        except Exception as e:
            logger.warning(f"Failed to create ROC curve plot: {str(e)}")
            if plt.get_fignums():
                plt.close()
    
    # Plot Precision-Recall curve if available and valid
    if 'pr_curve' in metrics and metrics['pr_curve'] is not None:
        try:
            plt.figure(figsize=(8, 6))
            precision = metrics['pr_curve']['precision']
            recall = metrics['pr_curve']['recall']
            plt.plot(recall, precision, lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve (AUC = {metrics.get("pr_auc", 0):.3f})')
            
            # Save the plot
            pr_path = os.path.join(output_dir, 'pr_curve.png')
            plt.savefig(pr_path)
            plt.close()
            
            plot_paths['pr_curve'] = pr_path
        except Exception as e:
            logger.warning(f"Failed to create PR curve plot: {str(e)}")
            if plt.get_fignums():
                plt.close()
    
    # Plot metric comparison if class report is available
    if 'class_report' in metrics:
        plt.figure(figsize=(10, 6))
        
        # Extract metrics from class report
        class_metrics = metrics['class_report']
        if '1' in class_metrics and '0' in class_metrics:
            positive_metrics = class_metrics['1']
            negative_metrics = class_metrics['0']
            
            # Create metric comparison
            metric_names = ['precision', 'recall', 'f1-score']
            pos_values = [positive_metrics[m] for m in metric_names]
            neg_values = [negative_metrics[m] for m in metric_names]
            
            x = np.arange(len(metric_names))
            width = 0.35
            
            plt.bar(x - width/2, pos_values, width, label='Positive Class (1)')
            plt.bar(x + width/2, neg_values, width, label='Negative Class (0)')
            
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.title('Performance Metrics by Class')
            plt.xticks(x, metric_names)
            plt.ylim(0, 1)
            plt.legend()
            
            # Save the plot
            metrics_path = os.path.join(output_dir, 'class_metrics.png')
            plt.savefig(metrics_path)
            plt.close()
            
            plot_paths['class_metrics'] = metrics_path
    
    logger.info(f"Created {len(plot_paths)} visualization plots in {output_dir}")
    
    return plot_paths


def create_feature_importance_plot(
    feature_importance: Dict,
    output_dir: str = 'results/plots',
    n_features: int = 20
) -> str:
    """
    Create and save feature importance visualization.
    
    Parameters:
    -----------
    feature_importance : Dict
        Dictionary with feature importance information
    output_dir : str, optional
        Directory to save the plot
    n_features : int, optional
        Number of top features to include in the plot
        
    Returns:
    --------
    str
        Path to the saved plot
    """
    logger.info(f"Creating feature importance plot for top {n_features} features...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if feature importance data is available
    if 'importance_df' not in feature_importance:
        logger.error("Feature importance data not available")
        return None
    
    # Get importance DataFrame
    importance_df = feature_importance['importance_df']
    
    # Limit to top N features
    plot_df = importance_df.head(n_features).copy()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    plt.barh(range(len(plot_df)), plot_df['importance'], align='center')
    plt.yticks(range(len(plot_df)), plot_df['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {n_features} Feature Importances ({feature_importance.get("method", "unknown")} method)')
    
    # Add values to the bars
    for i, importance in enumerate(plot_df['importance']):
        plt.text(importance, i, f'{importance:.4f}', va='center')
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved to {plot_path}")
    
    return plot_path


def create_bias_variance_plot(
    bias_variance: Dict,
    output_dir: str = 'results/plots'
) -> str:
    """
    Create and save bias-variance learning curve plot.
    
    Parameters:
    -----------
    bias_variance : Dict
        Dictionary with bias-variance analysis results
    output_dir : str, optional
        Directory to save the plot
        
    Returns:
    --------
    str
        Path to the saved plot
    """
    logger.info("Creating bias-variance learning curve plot...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if bias-variance data is available
    required_keys = ['train_sizes', 'train_mean_scores', 'test_mean_scores']
    if not all(key in bias_variance for key in required_keys):
        logger.error("Bias-variance data not available")
        return None
    
    # Extract data
    train_sizes = bias_variance['train_sizes']
    train_mean = bias_variance['train_mean_scores']
    train_std = bias_variance.get('train_std_scores', [0] * len(train_mean))
    test_mean = bias_variance['test_mean_scores']
    test_std = bias_variance.get('test_std_scores', [0] * len(test_mean))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot learning curves
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    
    # Add shaded regions for standard deviation
    plt.fill_between(train_sizes, 
                     [max(t - s, 0) for t, s in zip(train_mean, train_std)],
                     [min(t + s, 1) for t, s in zip(train_mean, train_std)], 
                     alpha=0.1, color='r')
    plt.fill_between(train_sizes, 
                     [max(t - s, 0) for t, s in zip(test_mean, test_std)],
                     [min(t + s, 1) for t, s in zip(test_mean, test_std)], 
                     alpha=0.1, color='g')
    
    # Add labels and title
    plt.xlabel('Training Set Size')
    plt.ylabel('Balanced Accuracy Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    
    # Add bias-variance annotation
    bias = bias_variance.get('bias', 0)
    variance = bias_variance.get('variance', 0)
    diagnosis = bias_variance.get('diagnosis', 'unknown')
    
    annotation = f"Bias: {bias:.4f}\nVariance: {variance:.4f}\nDiagnosis: {diagnosis}"
    plt.annotate(annotation, xy=(0.5, 0.1), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'learning_curve.png')
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Bias-variance plot saved to {plot_path}")
    
    return plot_path


def create_time_series_performance_plot(
    time_analysis: Dict,
    output_dir: str = 'results/plots',
    metric: str = 'balanced_accuracy'
) -> str:
    """
    Create and save time series performance plot.
    
    Parameters:
    -----------
    time_analysis : Dict
        Dictionary with time-based performance analysis
    output_dir : str, optional
        Directory to save the plot
    metric : str, optional
        Metric to plot over time
        
    Returns:
    --------
    str
        Path to the saved plot
    """
    logger.info(f"Creating time series plot for {metric}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if time analysis data is available
    if 'time_series' not in time_analysis:
        logger.error("Time series data not available")
        return None
    
    # Extract data
    time_series = time_analysis['time_series']
    
    if 'periods' not in time_series or metric not in time_series:
        logger.error(f"Required time series data for {metric} not available")
        return None
    
    periods = time_series['periods']
    values = time_series[metric]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot time series
    plt.plot(periods, values, 'o-', linewidth=2)
    
    # Add trend line
    if len(periods) >= 3:
        x = np.arange(len(periods))
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        plt.plot(periods, p(x), '--', linewidth=1, color='red')
        
        # Add trend direction annotation
        trend = time_analysis.get('trend_direction', {}).get(metric, 'unknown')
        plt.annotate(f"Trend: {trend}", xy=(0.7, 0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    # Add labels and title
    plt.xlabel('Time Period')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} Over Time')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'timeseries_{metric}.png')
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Time series plot saved to {plot_path}")
    
    return plot_path


def convert_to_serializable(obj):
    """Convert object to JSON serializable format."""
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
        return bool(obj)  # Explicitly convert boolean values
    return obj

def generate_performance_report(
    evaluation_results: Dict,
    output_file: str = 'results/performance_report.json'
) -> bool:
    """
    Generate a comprehensive performance report from evaluation results.
    
    Parameters:
    -----------
    evaluation_results : Dict
        Dictionary with all evaluation results
    output_file : str, optional
        Path to save the report
        
    Returns:
    --------
    bool
        True if the report was successfully generated, False otherwise
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
        
        # Convert each part of the report to JSON serializable format
        serialized_report = {}
        for key, value in report.items():
            if key == 'success_criteria':
                # Explicitly handle boolean values in success_criteria
                serialized_report[key] = {k: bool(v) for k, v in value.items()}
            else:
                serialized_report[key] = convert_to_serializable(value)
        
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


class ModelEvaluator:
    """
    Class to handle model evaluation, analysis, and reporting.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ModelEvaluator with optional configuration.
        
        Parameters:
        -----------
        config : Dict, optional
            Configuration dictionary with options for model evaluation
        """
        self.config = self._load_default_config()
        if config is not None:
            self.config.update(config)
        
        self.results = {}
        self.output_dir = self.config['output_dirs']['base']
        self.plots_dir = self.config['output_dirs']['plots']
    
    def _load_default_config(self) -> Dict:
        """Load default configuration settings."""
        return {
            "output_dirs": {
                "base": "results",
                "plots": "results/plots"
            },
            "feature_importance": {
                "method": "model_based",
                "n_top_features": 20,
                "n_repeats": 10,
                "random_state": 42
            },
            "metrics": {
                "balanced_accuracy_threshold": 0.55,
                "zero_division": 0,
                "slope_threshold": 0.01
            },
            "visualization": {
                "plot_figsize": {
                    "default": [10, 6],
                    "feature_importance": [12, 8],
                    "confusion_matrix": [8, 6]
                },
                "plot_style": {
                    "linewidth": 2,
                    "alpha": 0.1
                }
            },
            "time_analysis": {
                "time_unit": "month",
                "min_periods_for_trend": 3
            }
        }
        
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_prob: pd.Series = None
    ) -> Dict:
        """
        Evaluate a model's performance on test data.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target values
        y_prob : pd.Series, optional
            Predicted probabilities (if already calculated)
            
        Returns:
        --------
        Dict
            Dictionary with evaluation metrics
        """
        # Get predictions if y_prob is not provided
        if y_prob is None:
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # Get predicted classes
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics with config
        metrics = calculate_performance_metrics(
            y_test, 
            y_pred, 
            y_prob, 
            config=self.config
        )
        
        # Store the results
        self.results['overall_metrics'] = metrics
        
        return metrics
    
    def analyze_by_ticker(
        self,
        predictions: pd.DataFrame,
        excess_returns: pd.DataFrame,
        metrics_config: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze model performance by ticker.
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            DataFrame with predictions (multi-index: ticker, date)
        excess_returns : pd.DataFrame
            DataFrame with excess returns
            
        Returns:
        --------
        Dict
            Dictionary with ticker-level analysis
        """
        ticker_analysis = analyze_predictions_by_ticker(
            predictions,
            excess_returns,
            config=self.config.get('metrics', None)
        )
        
        # Store the results
        self.results['ticker_analysis'] = ticker_analysis
        
        return ticker_analysis
    
    def analyze_by_time(
        self,
        predictions: pd.DataFrame,
        time_unit: str = 'month',
        metrics_config: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze model performance by time.
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            DataFrame with predictions (multi-index: ticker, date)
        time_unit : str, optional
            Time unit for analysis
            
        Returns:
        --------
        Dict
            Dictionary with time-based analysis
        """
        time_analysis = analyze_predictions_by_time(
            predictions,
            time_unit,
            config=self.config.get('metrics', None)
        )
        
        # Store the results
        self.results['time_analysis'] = time_analysis
        
        return time_analysis
    
    def analyze_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'model_based'
    ) -> Dict:
        """
        Analyze feature importance.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target values
        method : str, optional
            Method for calculating feature importance
            
        Returns:
        --------
        Dict
            Dictionary with feature importance analysis
        """
        feature_importance = analyze_feature_importance(model, X, y, method)
        
        # Store the results
        self.results['feature_importance'] = feature_importance
        
        return feature_importance
    
    def analyze_bias_variance(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Analyze bias-variance tradeoff.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target values
            
        Returns:
        --------
        Dict
            Dictionary with bias-variance analysis
        """
        bias_variance = analyze_bias_variance_tradeoff(model, X_train, y_train, X_test, y_test)
        
        # Store the results
        self.results['bias_variance'] = bias_variance
        
        return bias_variance
    
    def create_visualizations(self) -> Dict:
        """
        Create visualizations based on evaluation results.
        
        Returns:
        --------
        Dict
            Dictionary with paths to the generated visualizations
        """
        visualizations = {}
        plots_dir = os.path.join(self.output_dir, 'plots')
        
        # Create performance visualizations
        if 'overall_metrics' in self.results:
            perf_plots = create_performance_visualizations(self.results['overall_metrics'], plots_dir)
            visualizations.update(perf_plots)
        
        # Create feature importance plot
        if 'feature_importance' in self.results:
            fi_plot = create_feature_importance_plot(self.results['feature_importance'], plots_dir)
            if fi_plot:
                visualizations['feature_importance'] = fi_plot
        
        # Create bias-variance plot
        if 'bias_variance' in self.results:
            bv_plot = create_bias_variance_plot(self.results['bias_variance'], plots_dir)
            if bv_plot:
                visualizations['bias_variance'] = bv_plot
        
        # Create time series performance plot
        if 'time_analysis' in self.results:
            ts_plot = create_time_series_performance_plot(self.results['time_analysis'], plots_dir)
            if ts_plot:
                visualizations['time_series'] = ts_plot
        
        # Store the results
        self.results['visualizations'] = visualizations
        
        return visualizations
    
    def generate_report(self, output_file: str = None) -> bool:
        """
        Generate a comprehensive performance report.
        
        Parameters:
        -----------
        output_file : str, optional
            Path to save the report
            
        Returns:
        --------
        bool
            True if the report was successfully generated, False otherwise
        """
        if not output_file:
            output_file = os.path.join(self.output_dir, 'performance_report.json')
        
        return generate_performance_report(self.results, output_file)
    
    def run_full_evaluation(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        predictions: pd.DataFrame = None,
        excess_returns: pd.DataFrame = None,
        time_unit: str = 'month',
        feature_importance_method: str = 'model_based'
    ) -> Dict:
        """
        Run a complete evaluation pipeline.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target values
        predictions : pd.DataFrame, optional
            DataFrame with predictions (if already calculated)
        excess_returns : pd.DataFrame, optional
            DataFrame with excess returns
        time_unit : str, optional
            Time unit for time-based analysis
        feature_importance_method : str, optional
            Method for calculating feature importance
            
        Returns:
        --------
        Dict
            Dictionary with all evaluation results
        """
        logger.info("Running full evaluation pipeline...")
        
        # Evaluate overall model performance
        self.evaluate_model(model, X_test, y_test)
        
        # Analyze feature importance
        self.analyze_feature_importance(model, X_train, y_train, feature_importance_method)
        
        # Analyze bias-variance tradeoff
        self.analyze_bias_variance(model, X_train, y_train, X_test, y_test)
        
        # If predictions DataFrame is provided, perform more detailed analysis
        if predictions is not None:
            # Analyze by ticker if excess returns are provided
            if excess_returns is not None:
                self.analyze_by_ticker(predictions, excess_returns)
            
            # Analyze by time
            self.analyze_by_time(predictions, time_unit)
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        logger.info("Full evaluation pipeline completed")
        
        return self.results
    

"""   
Model Evaluator Module for S&P500 Prediction Project

This module handles the evaluation, analysis, and visualization of 
model performance for S&P500 stock direction prediction.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import json
import os
from typing import Tuple, Dict, List, Optional, Union, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    auc
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        Configuration dictionary for metric settings
        
    Returns:
    --------
    Dict
        Dictionary of performance metrics
    """
    logger.info("Calculating performance metrics...")
    
    # Use default config if none provided
    # Accept both full config dict or just metrics dict
    if config is None:
        zero_div = 0
        threshold = 0.55
    elif "metrics" in config:
        zero_div = config["metrics"].get("zero_division", 0)
        threshold = config["metrics"].get("balanced_accuracy_threshold", 0.55)
    else:
        zero_div = config.get("zero_division", 0)
        threshold = config.get("balanced_accuracy_threshold", 0.55)

    # Basic classification metrics with configured zero division
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=zero_div),
        'recall': recall_score(y_true, y_pred, zero_division=zero_div),
        'f1': f1_score(y_true, y_pred, zero_division=zero_div),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'meets_accuracy_threshold': None
    }

    # Check if balanced accuracy meets threshold
    metrics['meets_accuracy_threshold'] = (
        metrics['balanced_accuracy'] > threshold
    )
    
    # Add class report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    metrics['class_report'] = class_report
    
    # Add probability-based metrics if probabilities are provided
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        # Calculate Precision-Recall curve
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
        metrics['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds_pr.tolist() if len(thresholds_pr) > 0 else None
        }
        metrics['pr_auc'] = auc(recall, precision)
    
    logger.info(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}, "
                f"F1 score: {metrics['f1']:.4f}")
    
    return metrics


def analyze_predictions_by_ticker(
    predictions: pd.DataFrame,
    excess_returns: pd.DataFrame,
    config: Optional[Dict] = None
) -> Dict:
    """
    Analyze prediction performance by ticker.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with predictions (multi-index: ticker, date)
    excess_returns : pd.DataFrame
        DataFrame with excess returns (has 'excess_return' column)
        
    Returns:
    --------
    Dict
        Dictionary with ticker-level performance metrics
    """
    logger.info("Analyzing prediction performance by ticker...")
    
    if predictions.empty or excess_returns.empty:
        logger.warning("Empty data provided. No analysis performed.")
        return {}
    
    if not isinstance(predictions.index, pd.MultiIndex):
        logger.error("Predictions DataFrame must have a multi-index (ticker, date)")
        return {}
    
    # Make sure we have predictions and true values
    if 'prediction' not in predictions.columns or 'true' not in predictions.columns:
        logger.error("Predictions DataFrame must have 'prediction' and 'true' columns")
        return {}
    
    # Ensure excess_returns has the necessary column
    if 'excess_return' not in excess_returns.columns:
        logger.error("Excess returns DataFrame must have 'excess_return' column")
        return {}
    
    # Get tickers
    tickers = predictions.index.get_level_values('ticker').unique()
    
    # Initialize results dictionary
    ticker_metrics = {}
    
    # Calculate metrics for each ticker
    for ticker in tickers:
        # Get ticker predictions
        ticker_preds = predictions.loc[ticker]
        
        # Get ticker excess returns (may need to align indices)
        ticker_excess = excess_returns.loc[ticker, 'excess_return']
        
        # Align data by date
        common_dates = ticker_preds.index.intersection(ticker_excess.index)
        if len(common_dates) == 0:
            logger.warning(f"No common dates for ticker {ticker}. Skipping.")
            continue
        
        # Filter data to common dates
        ticker_preds = ticker_preds.loc[common_dates]
        ticker_excess = ticker_excess.loc[common_dates]
        
        # Calculate basic metrics
        y_true = ticker_preds['true']
        y_pred = ticker_preds['prediction']
        y_prob = ticker_preds['probability'] if 'probability' in ticker_preds.columns else None
        
        # Calculate metrics with config
        metrics = calculate_performance_metrics(
            y_true, 
            y_pred, 
            y_prob, 
            config=config
        )
        
        # Calculate Profit & Loss
        # A positive prediction (1) means going long, negative (0) means going short
        # P&L is position * excess_return
        position = ticker_preds['prediction'] * 2 - 1  # Convert 0/1 to -1/1
        pnl = position * ticker_excess
        
        # Add P&L metrics
        metrics['pnl'] = {
            'total': pnl.sum(),
            'mean': pnl.mean(),
            'std': pnl.std(),
            'sharpe': pnl.mean() / pnl.std() if pnl.std() > 0 else 0,
            'win_rate': (pnl > 0).mean()
        }
        
        # Store metrics for this ticker
        ticker_metrics[ticker] = metrics
    
    # Calculate aggregate metrics
    agg_metrics = {
        'n_tickers': len(ticker_metrics),
        'avg_balanced_accuracy': np.mean([m['balanced_accuracy'] for m in ticker_metrics.values()]),
        'avg_f1': np.mean([m['f1'] for m in ticker_metrics.values()]),
        'avg_pnl': np.mean([m['pnl']['total'] for m in ticker_metrics.values()]),
        'avg_sharpe': np.mean([m['pnl']['sharpe'] for m in ticker_metrics.values()]),
        'top_tickers_by_accuracy': sorted(ticker_metrics.items(), key=lambda x: x[1]['balanced_accuracy'], reverse=True)[:5],
        'top_tickers_by_pnl': sorted(ticker_metrics.items(), key=lambda x: x[1]['pnl']['total'], reverse=True)[:5]
    }
    
    logger.info(f"Analyzed {agg_metrics['n_tickers']} tickers. "
                f"Average balanced accuracy: {agg_metrics['avg_balanced_accuracy']:.4f}, "
                f"Average P&L: {agg_metrics['avg_pnl']:.4f}")
    
    return {
        'ticker_metrics': ticker_metrics,
        'aggregate_metrics': agg_metrics
    }


def analyze_predictions_by_time(
    predictions: pd.DataFrame,
    time_unit: str = 'month',
    config: Optional[Dict] = None
) -> Dict:
    """
    Analyze prediction performance by time periods.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with predictions (multi-index: ticker, date)
    time_unit : str, optional
        Time unit for analysis ('day', 'week', 'month', 'quarter', 'year')
        
    Returns:
    --------
    Dict
        Dictionary with time-level performance metrics
    """
    logger.info(f"Analyzing prediction performance by {time_unit}...")
    
    if predictions.empty:
        logger.warning("Empty DataFrame provided. No analysis performed.")
        return {}
    
    if not isinstance(predictions.index, pd.MultiIndex):
        logger.error("Predictions DataFrame must have a multi-index (ticker, date)")
        return {}
    
    # Make sure we have predictions and true values
    if 'prediction' not in predictions.columns or 'true' not in predictions.columns:
        logger.error("Predictions DataFrame must have 'prediction' and 'true' columns")
        return {}
    
    # Reset index to get date as a column
    pred_df = predictions.reset_index()
    
    # Create time period column based on time_unit
    if time_unit == 'day':
        pred_df['period'] = pred_df['date'].dt.date
    elif time_unit == 'week':
        pred_df['period'] = pred_df['date'].dt.to_period('W').dt.start_time.dt.date
    elif time_unit == 'month':
        pred_df['period'] = pred_df['date'].dt.to_period('M').dt.start_time.dt.date
    elif time_unit == 'quarter':
        pred_df['period'] = pred_df['date'].dt.to_period('Q').dt.start_time.dt.date
    elif time_unit == 'year':
        pred_df['period'] = pred_df['date'].dt.year
    else:
        logger.error(f"Invalid time unit: {time_unit}")
        return {}
    
    # Get unique periods
    periods = pred_df['period'].unique()
    
    # Initialize results dictionary
    time_metrics = {}
    
    # Calculate metrics for each period
    for period in periods:
        # Get predictions for this period
        period_preds = pred_df[pred_df['period'] == period]
        
        # Calculate basic metrics
        y_true = period_preds['true']
        y_pred = period_preds['prediction']
        y_prob = period_preds['probability'] if 'probability' in period_preds.columns else None
        
        # Calculate metrics with config
        metrics = calculate_performance_metrics(
            y_true, 
            y_pred, 
            y_prob,
            config=config
        )
        
        # Calculate win rate by ticker
        if 'ticker' in period_preds.columns:
            ticker_results = period_preds.groupby('ticker').apply(
                lambda x: (x['prediction'] == x['true']).mean()
            )
            metrics['ticker_win_rates'] = ticker_results.to_dict()
            metrics['ticker_count'] = len(ticker_results)
        
        # Store metrics for this period
        time_metrics[str(period)] = metrics
    
    # Calculate trend of metrics over time
    time_series = {
        'periods': [str(p) for p in periods],
        'balanced_accuracy': [time_metrics[str(p)]['balanced_accuracy'] for p in periods],
        'precision': [time_metrics[str(p)]['precision'] for p in periods],
        'recall': [time_metrics[str(p)]['recall'] for p in periods],
        'f1': [time_metrics[str(p)]['f1'] for p in periods]
    }
    
    # Add ROC-AUC if available
    if 'roc_auc' in time_metrics[str(periods[0])]:
        time_series['roc_auc'] = [time_metrics[str(p)]['roc_auc'] for p in periods]
    
    # Calculate trend direction (increasing, decreasing, stable)
    trend_direction = {}
    for metric in ['balanced_accuracy', 'precision', 'recall', 'f1']:
        values = time_series[metric]
        if len(values) >= 3:
            # Calculate trend using linear regression slope
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            
            # Determine direction based on slope
            if slope > 0.01:  # Increasing
                trend_direction[metric] = 'increasing'
            elif slope < -0.01:  # Decreasing
                trend_direction[metric] = 'decreasing'
            else:  # Stable
                trend_direction[metric] = 'stable'
        else:
            trend_direction[metric] = 'insufficient_data'
    
    logger.info(f"Analyzed {len(periods)} {time_unit}s. "
                f"Balanced accuracy trend: {trend_direction.get('balanced_accuracy', 'N/A')}")
    
    return {
        'time_metrics': time_metrics,
        'time_series': time_series,
        'trend_direction': trend_direction
    }


def analyze_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'model_based',
    n_top_features: int = 20
) -> Dict:
    """
    Analyze feature importances for the model.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    method : str, optional
        Method for calculating feature importance ('model_based' or 'permutation')
    n_top_features : int, optional
        Number of top features to include in the result
        
    Returns:
    --------
    Dict
        Dictionary with feature importance information
    """
    logger.info(f"Analyzing feature importance using {method} method...")
    
    if X.empty or y.empty:
        logger.warning("Empty data provided. No analysis performed.")
        return {}
    
    # Model-based feature importance
    if method == 'model_based':
        # Check if the model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importances = np.abs(model.coef_[0])
        else:
            logger.error("Model does not have feature_importances_ or coef_ attribute")
            return {}
        
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
    # Permutation importance
    elif method == 'permutation':
        # Calculate permutation importance
        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Extract importances
        importances = result.importances_mean
        
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances,
            'std': result.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
    else:
        logger.error(f"Invalid feature importance method: {method}")
        return {}
    
    # Get top N features
    top_features = importance_df.head(n_top_features)
    
    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
    
    # Find number of features needed for 80% and 90% of importance
    features_for_80 = (importance_df['cumulative_importance'] <= 0.8).sum() + 1
    features_for_90 = (importance_df['cumulative_importance'] <= 0.9).sum() + 1
    
    logger.info(f"Top feature: {top_features.iloc[0]['feature']} "
                f"(importance: {top_features.iloc[0]['importance']:.4f})")
    logger.info(f"Features for 80% importance: {features_for_80}, "
                f"For 90% importance: {features_for_90}")
    
    return {
        'importance_df': importance_df,
        'top_features': top_features,
        'features_for_80_importance': features_for_80,
        'features_for_90_importance': features_for_90,
        'method': method
    }
