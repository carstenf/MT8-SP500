"""
Analysis Module

This module handles ticker-based and time-based analysis of model predictions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from .metrics_calculator import calculate_performance_metrics

# Set up logging
logger = logging.getLogger(__name__)

def analyze_predictions_by_ticker(
    predictions: pd.DataFrame,
    excess_returns: pd.DataFrame,
    config: Optional[Dict] = None
) -> Dict:
    """
    Analyze prediction performance and profitability by individual ticker.
    
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
    config : Dict, optional
        Configuration for metric calculation
        
    Returns:
    --------
    Dict
        Dictionary with comprehensive ticker analysis:
        - ticker_metrics: Dict[str, Dict]
            Metrics for each ticker including classification and P&L metrics
        - aggregate_metrics: Dict
            Summary statistics across all tickers
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
    Analyze prediction performance across different time periods.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with predictions (multi-index: ticker, date)
    time_unit : str, optional
        Time unit for analysis ('day', 'week', 'month', 'quarter', 'year')
    config : Dict, optional
        Configuration for metric calculation
        
    Returns:
    --------
    Dict
        Dictionary with time-based analysis:
        - time_metrics: Dict[str, Dict]
            Performance metrics for each time period
        - time_series: Dict
            Time series of metrics
        - trend_direction: Dict[str, str]
            Trend analysis for each metric
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
            # Calculate win rate per ticker without grouping on index
            ticker_results = period_preds.groupby('ticker', observed=True).agg({
                'prediction': lambda x: (x == period_preds.loc[x.index, 'true']).mean()
            })['prediction']
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
    slope_threshold = 0.01
    if config:
        slope_threshold = config.get('metrics', {}).get('slope_threshold', slope_threshold)
    
    for metric in ['balanced_accuracy', 'precision', 'recall', 'f1']:
        values = time_series[metric]
        if len(values) >= 3:
            # Calculate trend using linear regression slope
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            
            # Determine direction based on slope
            if slope > slope_threshold:  # Increasing
                trend_direction[metric] = 'increasing'
            elif slope < -slope_threshold:  # Decreasing
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
