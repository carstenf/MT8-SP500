"""
Target Engineering Module for S&P500 Prediction Project

This module handles the calculation of target variables for the prediction model.
It supports various types of targets:
1. Returns
   - Raw returns (price changes)
   - Excess returns (stock return - market return)
   - Log returns
   - Percentage returns
2. Binary Classification
   - Threshold based
   - Standard deviation based
   - Zero based
   - Quantile based
3. Multi-class Classification
   - Standard deviation based
   - Quantile based
   - Fixed range based
"""

import pandas as pd
import numpy as np
from typing import Dict

class TargetEngineer:
    """Class to handle target engineering for S&P500 prediction."""
    
    def __init__(self, config: Dict = None):
        """Initialize the TargetEngineer with configuration."""
        self.config = config or {}
        self.target_config = self.config.get('target', {})

    def _calculate_returns(self, data: pd.DataFrame, horizon: int, return_type: str = 'raw') -> pd.Series:
        """Calculate returns for a given horizon and type."""
        price_col = self.config['features']['price_col']
        
        if return_type == 'raw':
            # Calculate pct_change then shift to align with current date
            returns = data.groupby('ticker')[price_col].pct_change(horizon).shift(-horizon)
            

        elif return_type == 'excess':
            # Calculate stock returns first
            stock_returns = data.groupby('ticker')[price_col].pct_change(horizon)
            
            # Calculate market returns 
            market_returns = data.groupby('date')[price_col].mean().pct_change(horizon)
            
            # Subtract market returns from stock returns
            excess_returns = stock_returns - market_returns
            
            # Shift to align prediction day with target
            returns = excess_returns.shift(-horizon)
            # Now Day 1 has return from Day 1 to Day 6, etc.
            returns.name = f'excess_return_{horizon}d'
            
        elif return_type == 'log':
            # Calculate log return then shift to align
            returns = (np.log(data[price_col]) - np.log(data[price_col].shift(horizon))).shift(-horizon)
            
        elif return_type == 'percentage':
            # Calculate percentage return then shift to align
            returns = ((data[price_col] - data[price_col].shift(horizon)) / data[price_col].shift(horizon) * 100).shift(-horizon)
        
        return returns

    def _create_binary_target(self, returns: pd.Series, calc_config: Dict) -> pd.Series:
        """Create binary target based on configuration method."""
        method = calc_config['method']
        target = pd.Series(index=returns.index)
        
        if method == 'threshold_based':
            threshold = calc_config['fixed_threshold']
            if calc_config['return_type'] == 'raw' and threshold == 0:
                target = (returns > 0).astype(int)
            else:
                target = (returns > threshold).astype(int)
            
        elif method == 'std_based':
            rolling_mean = returns.groupby(level='ticker').transform(
                lambda x: x.rolling(window=calc_config['rolling_window'], min_periods=1).mean()
            )
            rolling_std = returns.groupby(level='ticker').transform(
                lambda x: x.rolling(window=calc_config['rolling_window'], min_periods=1).std()
            )
            threshold = rolling_mean + (calc_config['std_threshold'] * rolling_std)
            target = (returns > threshold).astype(int)
            
        elif method == 'zero_based':
            target = (returns > 0).astype(int)
            
        elif method == 'quantile_based':
            # Split returns at median for each date
            returns_reset = returns.reset_index()
            def assign_classes(x):
                median = x.median()
                result = pd.Series(0, index=x.index)  # Initialize all as class 0
                result[x >= median] = 1  # Values >= median get class 1
                return result
            
            target = returns_reset.groupby('date')[returns.name].transform(assign_classes)
            target.index = returns.index

        # do not remove this commented part
        # Show AAPL data if available
        #if 'AAPL' in returns.index.get_level_values('ticker'):
        #    aapl_target = target.loc['AAPL']
        #    print("\nAAPL Binary Classifications:")
        #    print(aapl_target.head(15).to_string())
        #    print("\nAAPL Last 15 days:")
        #    print(aapl_target.tail(15).to_string())    
            
        return target

    def _create_multiclass_target(self, returns: pd.Series, calc_config: Dict) -> pd.Series:
        """Create multi-class target based on configuration method."""
        method = calc_config['method']
        target = pd.Series(1, index=returns.index)  # Default to neutral class
        
        if method == 'std_based':
            rolling_mean = returns.groupby(level='ticker').transform(
                lambda x: x.rolling(window=calc_config['rolling_window'], min_periods=1).mean()
            )
            rolling_std = returns.groupby(level='ticker').transform(
                lambda x: x.rolling(window=calc_config['rolling_window'], min_periods=1).std()
            )
            upper = rolling_mean + (calc_config['std_threshold'] * rolling_std)
            lower = rolling_mean - (calc_config['std_threshold'] * rolling_std)
            target[returns >= upper] = 2  # up class
            target[returns <= lower] = 0  # down class
            
        elif method == 'quantile_based':
            quantiles = calc_config.get('quantiles', [0.33, 0.67])
            rolling_lower = returns.groupby(level='ticker').transform(
                lambda x: x.rolling(window=calc_config['rolling_window'], min_periods=1).quantile(quantiles[0])
            )
            rolling_upper = returns.groupby(level='ticker').transform(
                lambda x: x.rolling(window=calc_config['rolling_window'], min_periods=1).quantile(quantiles[1])
            )
            target[returns >= rolling_upper] = 2
            target[returns <= rolling_lower] = 0
            
        elif method == 'fixed_range':
            threshold = calc_config.get('fixed_threshold', 0.01)
            target[returns >= threshold] = 2
            target[returns <= -threshold] = 0


        # Show AAPL data if available
        if 'AAPL' in returns.index.get_level_values('ticker'):
            aapl_target = target.loc['AAPL']
            print("\nAAPL Binary Classifications:")
            print(aapl_target.head(15).to_string())
            print("\nAAPL Last 15 days:")
            print(aapl_target.tail(15).to_string())
    
            
        return target

    def create_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variables based on configuration."""
        
        target_type = self.target_config['type']
        calc_config = self.target_config['calculation']
        targets = pd.DataFrame(index=data.index)
        
        # For each horizon, calculate returns and create targets
        for horizon in calc_config['horizon']:
            # Calculate returns based on configuration
            returns = self._calculate_returns(
                data,
                horizon=horizon,
                return_type=calc_config['return_type']
            )
            
            # Set target name for identification
            returns.name = f'target_{horizon}d'
            
            # Create targets based on type
            if target_type == 'returns':
                targets[f'target_{horizon}d'] = returns
            elif target_type == 'binary':
                targets[f'target_{horizon}d'] = self._create_binary_target(returns, calc_config)
            elif target_type == 'multiclass':
                targets[f'target_{horizon}d'] = self._create_multiclass_target(returns, calc_config)
            else:
                raise ValueError(f"Unsupported target type: {target_type}")
        
            
        return targets
