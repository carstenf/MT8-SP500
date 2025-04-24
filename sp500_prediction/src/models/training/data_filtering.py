"""
Data Filtering Module

Simple filtering of data based on NaN values.
- Checks each ticker's features and target for NaNs
- Removes tickers that have any NaNs
- Shows AAPL data before filtering
"""

import pandas as pd
from typing import Dict, Tuple

def filter_single_dataset(X: pd.DataFrame, y: pd.Series, config: Dict = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Filter out tickers that have any NaN values."""
    
    # Print dataset info
    dates = X.index.get_level_values('date')
    print(f"\nDataset Overview:")
    print(f"Time range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
    print(f"Trading days: {len(dates.unique())}")
    print(f"Features: {X.shape[1]} columns")
    print(f"Tickers: {len(X.index.get_level_values('ticker').unique())}")

    # do not remove this commented part
    # Show AAPL data before filtering
    #if 'AAPL' in X.index.get_level_values('ticker').unique():
        #aapl_X = X.loc['AAPL']
        #aapl_y = y.loc['AAPL']   
        # Combine features and target
        #aapl_data = pd.concat([aapl_X, pd.DataFrame(aapl_y)], axis=1)
        #aapl_data = aapl_y
        #print("\nAAPL First 10 days:")
        #print(aapl_data.head(15).to_string() 
        #print("\nAAPL Last 10 days:")
        #print(aapl_data.tail(15).to_string())
    
    # Filter out tickers with NaNs
    tickers = X.index.get_level_values('ticker').unique()
    valid_tickers = []
    for ticker in tickers:
        ticker_X = X.loc[ticker]
        ticker_y = y.loc[ticker]
        
        if not ticker_X.isna().any().any() and not ticker_y.isna().any():
            valid_tickers.append(ticker)
    
    # Filter data to only include valid tickers
    X_filtered = X[X.index.get_level_values('ticker').isin(valid_tickers)]
    y_filtered = y[y.index.get_level_values('ticker').isin(valid_tickers)]
    
    #print(f"\nFiltering Results:")
    #print(f"Original tickers: {len(tickers)}")
    #print(f"Valid tickers: {len(valid_tickers)}")
    #print(f"Removed tickers: {len(tickers) - len(valid_tickers)}")
    
    return X_filtered, y_filtered

def filter_ticker_out_with_nan(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    config: Dict = None
) -> tuple:
    """Filter data, handling both single dataset and train/validation split cases."""
    
    # Single dataset case
    if X_val is None or y_val is None:
        print("\nProcessing single dataset...")
        return filter_single_dataset(X_train, y_train, config)
    
    # Train/validation split case
    print("\nProcessing training data...")
    X_train_filtered, y_train_filtered = filter_single_dataset(X_train, y_train, config)
    
    print("\nProcessing validation data...")
    X_val_filtered, y_val_filtered = filter_single_dataset(X_val, y_val, config)
    
    return X_train_filtered, y_train_filtered, X_val_filtered, y_val_filtered
