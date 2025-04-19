"""
Feature Engineering Module for S&P500 Prediction Project

This module handles the calculation of returns, excess returns,
and other features needed for the prediction model.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_returns(data: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Calculate daily returns for each stock using vectorized operations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        S&P500 data with multi-index (ticker, date)
    price_col : str, optional
        Column name containing price data (default: 'close')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily returns
    """
    logger.info(f"Calculating daily returns using {price_col} prices...")
    
    if data.empty:
        logger.warning("Empty DataFrame provided. No returns calculated.")
        return pd.DataFrame()
    
    if price_col not in data.columns:
        logger.error(f"Price column '{price_col}' not found in data.")
        return pd.DataFrame()
    
    # Create returns DataFrame
    returns = pd.DataFrame(index=data.index)
    
    # Calculate returns for all tickers at once using groupby
    returns['return'] = data.groupby(level='ticker')[price_col].pct_change()
    
    logger.info(f"Calculated returns for {len(returns.index.get_level_values('ticker').unique())} tickers")
    return returns


def calculate_market_returns(returns_data: pd.DataFrame) -> pd.Series:
    """
    Calculate daily market average returns.
    
    Parameters:
    -----------
    returns_data : pd.DataFrame
        DataFrame with stock returns (has 'return' column)
        
    Returns:
    --------
    pd.Series
        Series with daily market average returns indexed by date
    """
    logger.info("Calculating market average returns...")
    
    if returns_data.empty:
        logger.warning("Empty DataFrame provided. No market returns calculated.")
        return pd.Series()
    
    if 'return' not in returns_data.columns:
        logger.error("'return' column not found in returns data.")
        return pd.Series()
    
    # Calculate market average return for each date
    # Group by date and calculate mean of returns
    market_returns = returns_data.reset_index().groupby('date')['return'].mean()
    
    logger.info(f"Calculated market returns for {len(market_returns)} trading days")
    return market_returns


def calculate_excess_returns(returns_data: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
    """
    Calculate excess returns relative to market average.
    
    Parameters:
    -----------
    returns_data : pd.DataFrame
        DataFrame with stock returns (has 'return' column)
    market_returns : pd.Series
        Series with daily market average returns indexed by date
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with excess returns
    """
    logger.info("Calculating excess returns...")
    
    if returns_data.empty or market_returns.empty:
        logger.warning("Empty data provided. No excess returns calculated.")
        return pd.DataFrame()
    
    if 'return' not in returns_data.columns:
        logger.error("'return' column not found in returns data.")
        return pd.DataFrame()
    
    # Make a copy to avoid modifying original data
    excess_returns = returns_data.copy()
    
    # Create a new column for market returns aligned with stock dates
    # This ensures we're comparing stock returns with the correct market return for that day
    excess_returns['market_return'] = excess_returns.index.get_level_values('date').map(market_returns)
    
    # Calculate excess return: stock_return - market_return
    excess_returns['excess_return'] = excess_returns['return'] - excess_returns['market_return']
    
    logger.info("Excess returns calculated successfully")
    return excess_returns


def create_lagged_features(
    data: pd.DataFrame, 
    feature_col: str, 
    max_lag: int = 40, 
    long_term_lags: List[int] = None
) -> pd.DataFrame:
    """
    Create lagged features using vectorized operations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with feature data (multi-index: ticker, date)
    feature_col : str
        Column name to use for feature creation
    max_lag : int, optional
        Maximum number of days for daily lags (default: 40)
    long_term_lags : List[int], optional
        List of additional long-term lags to include
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with lagged features
    """
    logger.info(f"Creating lagged features using {feature_col} column...")
    
    if data.empty:
        logger.warning("Empty DataFrame provided. No features created.")
        return pd.DataFrame()
    
    if feature_col not in data.columns:
        logger.error(f"Feature column '{feature_col}' not found in data.")
        return pd.DataFrame()
    
    # Create features DataFrame with same index as input data
    features = pd.DataFrame(index=data.index)
    
    # Get the feature data
    feature_data = data[feature_col]
    
    # Set long-term lags if not provided
    if long_term_lags is None:
        # Create lags from 40 to 240 in steps of 10
        long_term_lags = list(range(40, 241, 10))
    
    # Create all lags in one go using groupby
    logger.info("Creating daily and long-term lags...")
    
    # Combine all lag periods
    all_lags = list(range(1, max_lag + 1)) + [lag for lag in long_term_lags if lag > max_lag]
    
    # Create lags using groupby and shift
    for lag in all_lags:
        features[f'lag_{lag}d'] = feature_data.groupby(level='ticker').shift(lag)
    
    # Create window-based features
    logger.info("Creating window-based features...")
    windows = [5, 10, 20, 60, 120, 240]
    
    # Group by ticker for rolling calculations
    grouped = feature_data.groupby(level='ticker')
    
    for window in windows:
        # Shift by 1 to avoid lookahead bias, then calculate rolling statistics
        shifted_data = grouped.shift(1)
        
        # Calculate window features all at once
        features[f'cum_{window}d'] = shifted_data.rolling(window=window).sum()
        features[f'avg_{window}d'] = shifted_data.rolling(window=window).mean()
        features[f'vol_{window}d'] = shifted_data.rolling(window=window).std()
    
    # Record the number of features created
    num_features = len(features.columns)
    logger.info(f"Created {num_features} lagged features")
    
    return features


def create_target_variable(excess_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable for next-day excess return direction.
    Uses vectorized operations for better performance.
    
    Parameters:
    -----------
    excess_returns : pd.DataFrame
        DataFrame with excess returns ('excess_return' column)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with target variable
    """
    logger.info("Creating target variable for next-day excess return direction...")
    
    if excess_returns.empty:
        logger.warning("Empty DataFrame provided. No target variable created.")
        return pd.DataFrame()
    
    if 'excess_return' not in excess_returns.columns:
        logger.error("'excess_return' column not found in data.")
        return pd.DataFrame()
    
    # Create target variable using groupby for vectorized operations
    target_df = pd.DataFrame(index=excess_returns.index)
    
    # Create next-day direction (1 = positive excess return, 0 = negative excess return)
    next_day_returns = excess_returns.groupby(level='ticker')['excess_return'].shift(-1)
    target_df['target'] = (next_day_returns > 0).astype(int)
    
    logger.info("Target variable created successfully")
    return target_df


def apply_feature_scaling(features: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply feature scaling to normalize features.
    
    Parameters:
    -----------
    features : pd.DataFrame
        DataFrame with features to scale
        
    Returns:
    --------
    Tuple[pd.DataFrame, StandardScaler]
        - DataFrame with scaled features
        - The fitted StandardScaler for later use
    """
    logger.info("Applying feature scaling...")
    
    if features.empty:
        logger.warning("Empty DataFrame provided. No scaling performed.")
        return pd.DataFrame(), None
    
    # Create scaler
    scaler = StandardScaler()
    
    # Create copy to avoid modifying original data
    scaled_features = pd.DataFrame(index=features.index)
    
    # Process each feature separately
    for col in features.columns:
        # Extract the feature values
        feature_values = features[col].values.reshape(-1, 1)
        
        # Fit and transform
        scaled_values = scaler.fit_transform(feature_values)
        
        # Store the scaled values
        scaled_features[col] = scaled_values.flatten()
    
    logger.info(f"Feature scaling applied to {len(features.columns)} features")
    return scaled_features, scaler


def analyze_class_distribution(target: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of target classes.
    
    Parameters:
    -----------
    target : pd.DataFrame
        DataFrame with target variable ('target' column)
        
    Returns:
    --------
    Dict
        Dictionary with class distribution statistics
    """
    logger.info("Analyzing target class distribution...")
    
    if target.empty:
        logger.warning("Empty DataFrame provided. No analysis performed.")
        return {}
    
    if 'target' not in target.columns:
        logger.error("'target' column not found in data.")
        return {}
        
    # Calculate overall class distribution
    class_counts = target['target'].value_counts()
    if len(class_counts) == 0:
        logger.error("No valid target values found after filtering.")
        return {}
    class_proportions = target['target'].value_counts(normalize=True)
    
    # Check if there's a severe imbalance
    majority_class = class_proportions.idxmax()
    majority_proportion = class_proportions.max()
    is_severely_imbalanced = majority_proportion > 0.7
    
    # Calculate class distribution by year
    yearly_distribution = target.reset_index()
    yearly_distribution['year'] = yearly_distribution['date'].dt.year
    yearly_stats = yearly_distribution.groupby('year')['target'].value_counts(normalize=True).unstack()
    
    # Calculate class weights for potential use in model training
    class_weights = {
        0: (1 / class_counts[0]) * (len(target) / 2),
        1: (1 / class_counts[1]) * (len(target) / 2)
    }
    
    distribution_stats = {
        'class_counts': class_counts.to_dict(),
        'class_proportions': class_proportions.to_dict(),
        'majority_class': int(majority_class),
        'majority_proportion': majority_proportion,
        'is_severely_imbalanced': is_severely_imbalanced,
        'class_weights': class_weights,
        'yearly_distribution': yearly_stats.to_dict() if not yearly_stats.empty else {}
    }
    
    logger.info(f"Class distribution: {class_proportions.to_dict()}")
    if is_severely_imbalanced:
        logger.warning(f"Severe class imbalance detected: {majority_proportion:.2f} for class {majority_class}")
    
    return distribution_stats


def prepare_features_targets(
    data: pd.DataFrame,
    price_col: str = 'close',
    max_lag: int = 40,
    long_term_lags: List[int] = None,
    apply_scaling: bool = True
) -> Dict:
    """
    Complete feature engineering pipeline to prepare features and targets.
    
    Parameters:
    -----------
    data : pd.DataFrame
        S&P500 data with multi-index (ticker, date)
    price_col : str, optional
        Column name containing price data (default: 'close')
    max_lag : int, optional
        Maximum number of days for daily lags (default: 40)
    long_term_lags : List[int], optional
        List of additional long-term lags to include
    apply_scaling : bool, optional
        Whether to apply feature scaling (default: True)
        
    Returns:
    --------
    Dict
        Dictionary with features, targets, and metadata
    """
    logger.info("Starting feature engineering pipeline...")
    
    if data.empty:
        logger.warning("Empty DataFrame provided. No features created.")
        return {}
    
    # Calculate daily returns
    returns = calculate_returns(data, price_col)
    
    # Calculate market returns
    market_returns = calculate_market_returns(returns)
    
    # Calculate excess returns
    excess_returns = calculate_excess_returns(returns, market_returns)
    
    # Create features using excess returns
    features = create_lagged_features(
        excess_returns, 
        'excess_return', 
        max_lag,
        long_term_lags
    )
    
    # Create target variable
    targets = create_target_variable(excess_returns)
    
    # Apply feature scaling if requested
    scaler = None
    if apply_scaling:
        scaled_features, scaler = apply_feature_scaling(features)
        features = scaled_features
    
    # Analyze class distribution
    class_distribution = analyze_class_distribution(targets)
    
    # Combine features and targets for final output
    # Keep only rows where we have both features and target
    valid_rows = features.index.intersection(targets.index)
    
    # Filter for valid rows
    filtered_features = features.loc[valid_rows]
    filtered_targets = targets.loc[valid_rows, 'target']
    
    # Count features by category
    feature_counts = {
        'daily_lags': len([col for col in filtered_features.columns if col.startswith('lag_') and int(col.split('_')[1].replace('d', '')) <= max_lag]),
        'long_term_lags': len([col for col in filtered_features.columns if col.startswith('lag_') and int(col.split('_')[1].replace('d', '')) > max_lag]),
        'cumulative': len([col for col in filtered_features.columns if col.startswith('cum_')]),
        'average': len([col for col in filtered_features.columns if col.startswith('avg_')]),
        'volatility': len([col for col in filtered_features.columns if col.startswith('vol_')]),
        'total': len(filtered_features.columns)
    }
    
    result = {
        'features': filtered_features,
        'targets': filtered_targets,
        'excess_returns': excess_returns,
        'feature_names': list(filtered_features.columns),
        'feature_counts': feature_counts,
        'class_distribution': class_distribution,
        'scaler': scaler,
        'n_samples': len(filtered_features),
        'n_features': len(filtered_features.columns),
        'max_lag_used': max_lag
    }
    
    logger.info(f"Feature engineering completed: {result['n_samples']} samples with {result['n_features']} features")
    return result


class FeatureEngineer:
    """
    Class to handle feature engineering for S&P500 prediction.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the FeatureEngineer with optional configuration.
        
        Parameters:
        -----------
        config : Dict, optional
            Configuration dictionary with options for feature engineering
        """
        self.config = config or {}
        self.price_col = self.config.get('price_col', 'close')
        self.max_lag = self.config.get('max_lag', 40)
        self.long_term_lags = self.config.get('long_term_lags', list(range(40, 241, 10)))
        self.apply_scaling = self.config.get('apply_scaling', True)
        self.scaler = None
        
    def create_features(self, data: pd.DataFrame) -> Dict:
        """
        Create features for the given data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            S&P500 data with multi-index (ticker, date)
            
        Returns:
        --------
        Dict
            Dictionary with features, targets, and metadata
        """
        result = prepare_features_targets(
            data,
            price_col=self.price_col,
            max_lag=self.max_lag,
            long_term_lags=self.long_term_lags,
            apply_scaling=self.apply_scaling
        )
        
        # Store the scaler for later use
        if 'scaler' in result:
            self.scaler = result['scaler']
        
        return result
    
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the stored scaler.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features to transform
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with transformed features
        """
        if self.scaler is None:
            logger.warning("No scaler available. Features will not be transformed.")
            return features
        
        # Create copy to avoid modifying original data
        transformed_features = pd.DataFrame(index=features.index)
        
        # Process each feature separately
        for col in features.columns:
            if col in self.scaler:
                # Extract the feature values
                feature_values = features[col].values.reshape(-1, 1)
                
                # Transform the values
                transformed_values = self.scaler[col].transform(feature_values)
                
                # Store the transformed values
                transformed_features[col] = transformed_values.flatten()
            else:
                # Keep the original values for features not in the scaler
                transformed_features[col] = features[col]
        
        logger.info(f"Transformed {len(features.columns)} features")
        return transformed_features
    
    def save_scaler(self, output_path: str) -> bool:
        """
        Save the feature scaler to file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the scaler
            
        Returns:
        --------
        bool
            True if save succeeded, False otherwise
        """
        if self.scaler is None:
            logger.warning("No scaler to save.")
            return False
        
        try:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"Saved scaler to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving scaler to {output_path}: {str(e)}")
            return False
    
    def load_scaler(self, input_path: str) -> bool:
        """
        Load the feature scaler from file.
        
        Parameters:
        -----------
        input_path : str
            Path to the saved scaler
            
        Returns:
        --------
        bool
            True if load succeeded, False otherwise
        """
        try:
            import pickle
            with open(input_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info(f"Loaded scaler from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading scaler from {input_path}: {str(e)}")
            return False
