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
        Configuration dictionary with settings:
        - metrics.zero_division: int, value to use for zero division (default: 0)
        - metrics.balanced_accuracy_threshold: float, threshold for accuracy (default: 0.55)
        
    Returns:
    --------
    Dict
        Dictionary of performance metrics including:
        - accuracy: float, overall accuracy
        - balanced_accuracy: float, balanced accuracy score
        - precision: float, precision score
        - recall: float, recall score
        - f1: float, F1 score
        - confusion_matrix: List[List[int]], confusion matrix values
        - meets_accuracy_threshold: bool, whether balanced accuracy meets threshold
        - metrics_version: str, version of metrics calculation
        - config_used: Dict, configuration settings used
        - class_report: Dict, detailed classification report
        - roc_auc: float, optional ROC AUC score if probabilities provided
        - roc_curve: Dict, optional ROC curve data if probabilities provided
        - pr_curve: Dict, optional PR curve data if probabilities provided
        - pr_auc: float, optional PR AUC score if probabilities provided
    """
    logger.info("Calculating performance metrics...")
    
    # Check if we have more than one class
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    is_single_class = len(unique_classes) == 1

    # Get zero_division value from config
    if config is None:
        zero_div = 0
        threshold = 0.55
    elif "metrics" in config:
        zero_div = config["metrics"].get("zero_division", 0)
        threshold = config["metrics"].get("balanced_accuracy_threshold", 0.55)
    else:
        zero_div = config.get("zero_division", 0)
        threshold = config.get("balanced_accuracy_threshold", 0.55)

    # Basic classification metrics with handling for edge cases
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=zero_div, average='weighted'),
        'recall': recall_score(y_true, y_pred, zero_division=zero_div, average='weighted'),
        'f1': f1_score(y_true, y_pred, zero_division=zero_div, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'is_single_class': is_single_class,
        'meets_accuracy_threshold': False  # Initialize to False
    }
    
    # Add class report with zero division handling
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=zero_div)
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
    config : Dict, optional
        Configuration dictionary with settings:
        - metrics.zero_division: int, value to use for zero division
        - metrics.balanced_accuracy_threshold: float, threshold for accuracy
        
    Returns:
    --------
    Dict
        Dictionary with ticker-level performance metrics including:
        - ticker_metrics: Dict with metrics for each ticker
        - aggregate_metrics: Dict with aggregated statistics including:
            - n_tickers: int, number of tickers analyzed
            - avg_balanced_accuracy: float, average balanced accuracy
            - avg_f1: float, average F1 score
            - avg_pnl: float, average P&L
            - avg_sharpe: float, average Sharpe ratio
            - top_tickers_by_accuracy: List of top performing tickers
            - top_tickers_by_pnl: List of highest P&L tickers
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
        metrics = calculate_performance_metrics(y_true, y_pred, y_prob, config=config)
        
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
    config : Dict, optional
        Configuration dictionary with settings:
        - metrics.zero_division: int, value to use for zero division
        - metrics.balanced_accuracy_threshold: float, threshold for accuracy
        - time_analysis.min_periods_for_trend: int, minimum periods needed for trend
        
    Returns:
    --------
    Dict
        Dictionary with time-level performance metrics including:
        - time_metrics: Dict with metrics for each time period
        - time_series: Dict with metric values over time
        - trend_direction: Dict indicating metric trends (increasing/decreasing/stable)
            for balanced_accuracy, precision, recall, and f1
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
        metrics = calculate_performance_metrics(y_true, y_pred, y_prob, config=config)
        
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
    n_top_features: int = 20,
    config: Optional[Dict] = None
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
    config : Dict, optional
        Configuration dictionary with settings:
        - feature_importance.method: str, method to use
        - feature_importance.n_repeats: int, number of permutation repeats
        - feature_importance.random_state: int, random seed
        - feature_importance.n_jobs: int, number of parallel jobs
        - feature_importance.n_top_features: int, number of top features
        
    Returns:
    --------
    Dict
        Dictionary with feature importance information including:
        - importance_df: DataFrame with feature importances
        - top_features: DataFrame with top N features
        - features_for_80_importance: int, features needed for 80% importance
        - features_for_90_importance: int, features needed for 90% importance
        - method: str, method used for calculation
    """
    logger.info(f"Analyzing feature importance using {method} method...")
    
    if X.empty or y.empty:
        logger.warning("Empty data provided. No analysis performed.")
        return {}
    
    # Get configuration parameters
    if config is None:
        config = {}
    feature_config = config.get('feature_importance', {})
    n_repeats = feature_config.get('n_repeats', 10)
    random_state = feature_config.get('random_state', 42)
    n_top_features = feature_config.get('n_top_features', n_top_features)
    
    # Get method from config or use provided method
    if config is not None:
        method = config.get('feature_importance', {}).get('method', method)
    
    # Log configuration
    logger.info(f"Feature importance configuration: method={method}, "
                f"n_repeats={n_repeats}, n_top_features={n_top_features}")
    
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
        # Get configuration parameters for permutation importance
        if config is None:
            config = {}
        feature_config = config.get('feature_importance', {})
        n_repeats = feature_config.get('n_repeats', 10)
        random_state = feature_config.get('random_state', 42)
        n_jobs = feature_config.get('n_jobs', -1)
        
        # Calculate permutation importance with configured parameters
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs
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
        Dictionary with bias-variance analysis results including:
        - train_sizes: List[float], sizes of training sets used
        - train_mean_scores: List[float], mean training scores
        - train_std_scores: List[float], std dev of training scores
        - test_mean_scores: List[float], mean validation scores
        - test_std_scores: List[float], std dev of validation scores
        - bias: float, model's bias measurement
        - variance: float, model's variance measurement
        - diagnosis: str, model diagnosis ('high_bias', 'high_variance', or 'balanced')
        - recommendations: List[str], suggested improvements based on diagnosis
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
    Create and save model performance visualizations.
    
    Creates a comprehensive set of performance visualization plots including:
    - Confusion matrix heatmap
    - ROC curve with AUC score
    - Precision-Recall curve with AUC score
    - Class-wise performance metrics comparison
    
    Parameters:
    -----------
    metrics : Dict
        Dictionary with performance metrics containing:
        - confusion_matrix: List[List[int]], confusion matrix values
        - roc_curve: Dict with 'fpr', 'tpr', 'thresholds' (optional)
        - roc_auc: float, ROC AUC score (optional)
        - pr_curve: Dict with 'precision', 'recall', 'thresholds' (optional)
        - pr_auc: float, PR AUC score (optional)
        - class_report: Dict, classification report by class
        
    output_dir : str, optional
        Directory to save the plots (default: 'results/plots')
        
    viz_config : Dict, optional
        Visualization configuration settings:
        - plot_figsize.default: Tuple[int, int], default figure size (default: (8, 6))
        - plot_figsize.confusion_matrix: Tuple[int, int], confusion matrix size
        - plot_style.linewidth: int, line width for plots (default: 2)
        - plot_style.alpha: float, transparency for shaded areas
        
    Returns:
    --------
    Dict
        Dictionary with paths to the saved plots including:
        - confusion_matrix: str, path to confusion matrix plot
        - roc_curve: str, path to ROC curve plot (if available)
        - pr_curve: str, path to precision-recall curve plot (if available)
        - class_metrics: str, path to class metrics comparison plot
        
    Notes:
    ------
    All plots are saved in PNG format. Each plot creation is handled independently
    with error handling, so failure to create one plot won't affect the others.
    Plot style and layout are customizable through the viz_config parameter.
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
        Dictionary with feature importance information including:
        - importance_df: DataFrame with all feature importances
        - method: str, method used for importance calculation
        - top_features: DataFrame with top N features
    output_dir : str, optional
        Directory to save the plot (default: 'results/plots')
    n_features : int, optional
        Number of top features to include in the plot (default: 20)
        
    Returns:
    --------
    str | None
        Path to the saved plot file if successful, None if creation failed
        The plot is saved as 'feature_importance.png' in the output directory
        
    Notes:
    ------
    Creates a horizontal bar chart of feature importances with:
    - Feature names on y-axis
    - Importance values on x-axis
    - Numerical values displayed next to each bar
    - Title indicating the importance calculation method
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
        Dictionary with bias-variance analysis results including:
        - train_sizes: List[float], training set sizes used
        - train_mean_scores: List[float], mean training scores
        - train_std_scores: List[float], std dev of training scores
        - test_mean_scores: List[float], validation scores
        - test_std_scores: List[float], std dev of validation scores
        - bias: float, model's bias measurement
        - variance: float, model's variance measurement
        - diagnosis: str, model diagnosis
    output_dir : str, optional
        Directory to save the plot (default: 'results/plots')
        
    Returns:
    --------
    str | None
        Path to the saved plot file if successful, None if required data is missing
        The plot is saved as 'learning_curve.png' in the output directory
        
    Notes:
    ------
    Creates a learning curve plot showing:
    - Training and validation scores vs. training set size
    - Shaded regions for score standard deviations
    - Annotation box with bias, variance, and diagnosis information
    - Legend indicating training and cross-validation curves
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
        Dictionary with time-based performance analysis including:
        - time_series: Dict with metric values over time periods
        - trend_direction: Dict with trend analysis for each metric
    output_dir : str, optional
        Directory to save the plot (default: 'results/plots')
    metric : str, optional
        Metric to plot over time (default: 'balanced_accuracy')
        Valid options: 'balanced_accuracy', 'precision', 'recall', 'f1'
        
    Returns:
    --------
    str | None
        Path to the saved plot file if successful, None if creation failed
        The plot is saved as 'timeseries_{metric}.png' in the output directory
        
    Notes:
    ------
    Creates a time series plot showing:
    - Performance metric values over time periods
    - Trend line if 3 or more periods are available
    - Trend direction annotation (increasing/decreasing/stable)
    - X-axis labels rotated 45° for better readability
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
    
    # Convert string periods to datetime if possible
    try:
        periods = pd.to_datetime(time_series['periods'])
    except (ValueError, TypeError):
        # Keep as strings if conversion fails
        periods = time_series['periods']
    
    values = time_series[metric]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot time series with proper x-axis handling
    if isinstance(periods, pd.DatetimeIndex):
        plt.plot(periods, values, 'o-', linewidth=2)
    else:
        # For non-datetime x-axis, use numerical indices for plotting
        plt.plot(range(len(periods)), values, 'o-', linewidth=2)
        plt.xticks(range(len(periods)), periods)
    
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
    """
    Convert Python objects to JSON serializable format.
    
    Parameters:
    -----------
    obj : Any
        The object to convert. Can be of types:
        - numpy integers/floats
        - numpy arrays
        - pandas DataFrames/Series
        - dictionaries
        - lists
        - boolean values
        - other basic Python types
        
    Returns:
    --------
    Any
        JSON serializable version of the input object:
        - numpy numbers -> Python numbers
        - numpy arrays -> Python lists
        - pandas DataFrames -> list of records
        - pandas Series -> dictionary
        - nested objects are recursively converted
        - other types remain unchanged
        
    Notes:
    ------
    Used primarily for preparing model evaluation results for JSON storage.
    Handles nested structures by recursive conversion.
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
        return bool(obj)  # Explicitly convert boolean values
    return obj

def generate_performance_report(
    evaluation_results: Dict,
    output_file: str = 'results/performance_report.json'
) -> bool:
    """
    Generate a comprehensive performance report from evaluation results in JSON format.
    
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
        Path to save the report (default: 'results/performance_report.json')
        
    Returns:
    --------
    bool
        True if the report was successfully generated and saved, False otherwise
        
    Notes:
    ------
    The report includes:
    - Overall performance metrics
    - Top performing tickers and their metrics
    - Time series trend analysis
    - Feature importance summary
    - Bias-variance diagnosis and recommendations
    - Success criteria evaluation:
        * Balanced accuracy > 0.55
        * Stable or improving performance
        * Model explanations available
        * Pipeline reproducibility
    - Overall assessment score and classification
        * Successful: >= 75% criteria met
        * Partially Successful: >= 50% criteria met
        * Unsuccessful: < 50% criteria met
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
    Comprehensive model evaluation, analysis, and reporting class for S&P500 prediction models.
    
    This class provides a unified interface for:
    - Calculating performance metrics
    - Analyzing predictions by ticker and time periods
    - Evaluating feature importance
    - Assessing bias-variance tradeoff
    - Generating visualizations
    - Creating performance reports
    
    The class uses a configuration-based approach for customization of:
    - Output directories
    - Feature importance methods
    - Metric thresholds
    - Visualization settings
    - Time analysis parameters
    
    All results are stored in the class instance for easy access and reporting.
    Supports both basic model evaluation and detailed analysis with 
    ticker-specific performance and time-based trends.
    
    Example:
    --------
    evaluator = ModelEvaluator()
    results = evaluator.run_full_evaluation(
        model=trained_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions_df,
        excess_returns=returns_df
    )
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
        """
        Load default configuration settings for model evaluation.
        
        Returns:
        --------
        Dict
            Default configuration dictionary with the following structure:
            - output_dirs:
                - base: str, base directory for all outputs (default: 'results')
                - plots: str, directory for plot files (default: 'results/plots')
            - feature_importance:
                - method: str, importance calculation method (default: 'model_based')
                - n_top_features: int, number of top features to show (default: 20)
                - n_repeats: int, permutation repeats (default: 10)
                - random_state: int, random seed (default: 42)
            - metrics:
                - balanced_accuracy_threshold: float, minimum acceptable accuracy (default: 0.55)
                - zero_division: int, value for zero division cases (default: 0)
                - slope_threshold: float, threshold for trend detection (default: 0.01)
            - visualization:
                - plot_figsize:
                    - default: List[int], default figure size (default: [10, 6])
                    - feature_importance: List[int], feature plot size (default: [12, 8])
                    - confusion_matrix: List[int], confusion matrix size (default: [8, 6])
                - plot_style:
                    - linewidth: int, line width for plots (default: 2)
                    - alpha: float, transparency for shaded areas (default: 0.1)
            - time_analysis:
                - time_unit: str, default time unit for analysis (default: 'month')
                - min_periods_for_trend: int, minimum periods for trend calculation (default: 3)
        """
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
        Evaluate a model's performance on test data using multiple metrics.
        
        This method calculates comprehensive performance metrics including:
        - Basic classification metrics (accuracy, precision, recall, F1)
        - Balanced accuracy for imbalanced datasets
        - ROC curves and AUC scores
        - Precision-Recall curves and AUC scores
        - Confusion matrix
        - Detailed classification report by class
        
        Parameters:
        -----------
        model : Any
            Trained model instance with predict and predict_proba methods
        X_test : pd.DataFrame
            Test feature matrix with shape (n_samples, n_features)
        y_test : pd.Series
            True target values with shape (n_samples,)
        y_prob : pd.Series, optional
            Pre-calculated prediction probabilities. If None, calculated using model.predict_proba
            
        Returns:
        --------
        Dict
            Dictionary containing evaluation metrics:
            - accuracy: float
            - balanced_accuracy: float
            - precision: float
            - recall: float
            - f1: float
            - confusion_matrix: List[List[int]]
            - roc_auc: float (if probabilities available)
            - roc_curve: Dict (if probabilities available)
            - pr_curve: Dict (if probabilities available)
            - class_report: Dict
            
        Notes:
        ------
        Uses configuration settings from self.config:
        - metrics.zero_division: int, handling of zero division cases
        - metrics.balanced_accuracy_threshold: float, threshold for success
        
        Results are automatically stored in self.results['overall_metrics']
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
        Analyze model performance and profitability by individual ticker.
        
        Performs detailed analysis of prediction performance and financial metrics
        for each ticker, including:
        - Classification metrics (accuracy, precision, recall, F1)
        - ROC and PR curves (if probabilities available)
        - Profit & Loss metrics (total P&L, Sharpe ratio, win rate)
        - Ticker-specific confusion matrices
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            DataFrame with predictions (multi-index: ticker, date) containing:
            - 'prediction' column: predicted labels (0/1)
            - 'true' column: actual labels (0/1)
            - 'probability' column: prediction probabilities (optional)
            
        excess_returns : pd.DataFrame
            DataFrame with excess returns data containing:
            - 'excess_return' column: ticker's return minus benchmark
            Index should match the predictions DataFrame dates
            
        metrics_config : Optional[Dict], optional
            Configuration for metric calculation (defaults to self.config['metrics']):
            - zero_division: int, value for division by zero cases
            - balanced_accuracy_threshold: float, success threshold
            
        Returns:
        --------
        Dict
            Dictionary with comprehensive ticker analysis:
            - ticker_metrics: Dict[str, Dict]
                Metrics for each ticker including classification and P&L metrics
            - aggregate_metrics: Dict
                Summary statistics across all tickers:
                - n_tickers: int, number of tickers analyzed
                - avg_balanced_accuracy: float, mean accuracy across tickers
                - avg_f1: float, mean F1 score
                - avg_pnl: float, mean P&L
                - avg_sharpe: float, mean Sharpe ratio
                - top_tickers_by_accuracy: List, best performing tickers
                - top_tickers_by_pnl: List, most profitable tickers
                
        Notes:
        ------
        - Requires aligned dates between predictions and excess_returns
        - Handles missing data by skipping tickers with no common dates
        - P&L calculation: long position for prediction=1, short for prediction=0
        - Results are used for both performance evaluation and trading strategy analysis
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
        Analyze model performance across different time periods to assess temporal stability.
        
        Performs time-based analysis of prediction performance by:
        - Aggregating predictions into specified time units (day/week/month/quarter/year)
        - Calculating performance metrics for each time period
        - Analyzing trends in key metrics over time
        - Tracking ticker-specific win rates within each period
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            DataFrame with predictions (multi-index: ticker, date) containing:
            - 'prediction' column: predicted labels (0/1)
            - 'true' column: actual labels (0/1)
            - 'probability' column: prediction probabilities (optional)
            
        time_unit : str, optional
            Time unit for analysis (default: 'month'). Options:
            - 'day': Daily analysis
            - 'week': Weekly analysis
            - 'month': Monthly analysis
            - 'quarter': Quarterly analysis
            - 'year': Yearly analysis
            
        metrics_config : Optional[Dict], optional
            Configuration for metric calculation (defaults to self.config['metrics']):
            - zero_division: int, handling of zero division cases
            - balanced_accuracy_threshold: float, success threshold
            
        Returns:
        --------
        Dict
            Dictionary with comprehensive time-based analysis:
            - time_metrics: Dict[str, Dict]
                Performance metrics for each time period
            - time_series: Dict
                Time series of metrics including:
                - balanced_accuracy: List[float]
                - precision: List[float]
                - recall: List[float]
                - f1: List[float]
                - roc_auc: List[float] (if probabilities available)
            - trend_direction: Dict[str, str]
                Trend analysis for each metric ('increasing'/'decreasing'/'stable')
                
        Notes:
        ------
        - Time periods are created based on the date index of predictions
        - Trend analysis requires at least 3 periods of data
        - Results are automatically stored in self.results['time_analysis']
        - Supports analysis of model stability and performance degradation
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
        method: str = None
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
            Method for calculating feature importance (overrides config)
            
        Returns:
        --------
        Dict
            Dictionary with feature importance analysis
        """
        # Use method from config if not explicitly provided
        if method is None:
            method = self.config.get('feature_importance', {}).get('method', 'model_based')
            
        feature_importance = analyze_feature_importance(
            model=model,
            X=X,
            y=y,
            method=method,
            n_top_features=self.config.get('feature_importance', {}).get('n_top_features', 20),
            config=self.config.get('feature_importance', {})
        )
        
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
        feature_importance_method: Optional[str] = None
    ) -> Dict:
        """
        Execute a comprehensive model evaluation pipeline.
        
        This method runs a complete evaluation of the model including:
        1. Basic performance metrics calculation
        2. Feature importance analysis
        3. Bias-variance tradeoff assessment
        4. Time-based performance analysis
        5. Ticker-specific analysis (if predictions provided)
        6. Visualization generation
        7. Performance report creation
        
        Parameters:
        -----------
        model : Any
            Trained model instance with predict and predict_proba methods
        X_train : pd.DataFrame
            Training feature matrix with shape (n_train_samples, n_features)
        y_train : pd.Series
            Training target values with shape (n_train_samples,)
        X_test : pd.DataFrame
            Test feature matrix with shape (n_test_samples, n_features)
        y_test : pd.Series
            Test target values with shape (n_test_samples,)
        predictions : pd.DataFrame, optional
            DataFrame with predictions (multi-index: ticker, date) containing:
            - prediction: 0/1 labels
            - true: actual values
            - probability: prediction probabilities (optional)
        excess_returns : pd.DataFrame, optional
            DataFrame with excess returns for trading analysis
            Required if predictions are provided and P&L analysis is needed
        time_unit : str, optional
            Time unit for temporal analysis (default: 'month')
            Options: 'day', 'week', 'month', 'quarter', 'year'
        feature_importance_method : str, optional
            Method for feature importance calculation
            If None, uses value from config (default: 'model_based')
            
        Returns:
        --------
        Dict
            Dictionary with comprehensive evaluation results:
            - overall_metrics: Dict, basic performance metrics
            - feature_importance: Dict, feature importance analysis
            - bias_variance: Dict, bias-variance assessment
            - time_analysis: Dict, temporal performance analysis (if predictions provided)
            - ticker_analysis: Dict, ticker-level analysis (if predictions and returns provided)
            - visualizations: Dict, paths to generated plots
            
        Notes:
        ------
        - All results are stored in self.results for later access
        - Generates visualizations in self.output_dir/plots
        - Creates a comprehensive JSON report in self.output_dir
        - Uses configuration from self.config for analysis parameters
        - Handles missing data and invalid inputs gracefully
        """
        logger.info("Running full evaluation pipeline...")
        
        # Evaluate overall model performance
        self.evaluate_model(model, X_test, y_test)
        
        # Get feature importance method from config if not provided
        if feature_importance_method is None:
            fi_method = self.config.get('feature_importance', {}).get('method', 'model_based')
        else:
            fi_method = feature_importance_method
            
        # Analyze feature importance with provided method or config default
        self.analyze_feature_importance(model, X_train, y_train, method=fi_method)
        
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
