"""
Feature Engineering Module for S&P500 Prediction Project

This module handles the calculation of technical indicators for the prediction model.

Technical Indicator Categories:
1. Basic Momentum & Trend (RSI, ROC, Momentum)
   * Capture short-term price dynamics
   * Early signals of trend changes
2. Advanced Trend (MACD, Bollinger Bands)
   * Complex trend analysis
   * Volatility-based signals
3. Oscillators (Stochastic, Williams %R)
   * Overbought/oversold conditions
   * Price reversals
4. Moving Averages (SMA, EMA, WMA)
   * Trend following
   * Support/resistance levels
5. Volume-based (OBV, A/D, Volume ROC)
   * Price-volume relationships
   * Market participation

Data Flow & NaN Handling:
- Features are calculated preserving NaN values
- NaN values are NOT filtered during feature engineering
- Final NaN filtering is handled by data_filtering.py during model training
- This approach maintains data integrity and allows flexible filtering

Feature Calculation:
- Technical indicators calculated on raw prices
- Input data must have MultiIndex (ticker, date)
- Missing values occur naturally due to:
  * Lookback periods at start of each ticker's data
  * Missing price data in original series
- Features are properly aligned with date index

Usage Notes:
- Let NaN propagate naturally through calculations
- Don't remove NaN values in feature engineering
- NaN filtering happens in model training pipeline
"""

import pandas as pd
import numpy as np
import talib
import logging
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler

# Set up logging
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class to handle feature engineering for S&P500 prediction."""
    
    def __init__(self, config: Dict = None):
        """Initialize the FeatureEngineer with configuration."""
        self.config = config or {}
        self.tech_indicators = self.config.get('features', {}).get('technical_indicators', {})
        self.scaler = None

    def _get_indicator_periods(self) -> Dict[str, Union[List[int], Dict[str, List[int]]]]:
        """Get periods for technical indicators."""
        return {
            'momentum': self._generate_momentum_periods(self.tech_indicators['momentum']),
            'rsi': [14],  # Standard RSI period
            'macd': {
                'fast': [12],  # Standard MACD fast period
                'slow': [26],  # Standard MACD slow period
                'signal': [9]  # Standard MACD signal period
            },
            'bollinger': [20],  # Standard Bollinger period
            'stochastic': [14],  # Standard Stochastic period
            'williams_r': [14],  # Standard Williams %R period
            'moving_averages': [5, 10, 20, 50, 200],  # Standard MA periods
            'volume': [10, 20]  # Standard volume indicator periods
        }

    def _generate_momentum_periods(self, config: Dict) -> List[int]:
        """Generate momentum periods based on configuration."""
        if 'timeperiod_generation' not in config:
            return [5, 10, 20, 60]  # Default periods
            
        gen_config = config['timeperiod_generation']
        if gen_config['method'] != 'range':
            logger.warning(f"Unsupported momentum period generation method")
            return [5, 10, 20, 60]
            
        params = gen_config['params']
        periods = []
        
        # Generate short-range periods
        if 'short_range' in params:
            short_range = params['short_range']
            periods.extend(range(
                short_range['start'],
                short_range['end'],
                short_range.get('step', 1)
            ))
            
        # Generate long-range periods
        if 'long_range' in params:
            long_range = params['long_range']
            periods.extend(range(
                long_range['start'],
                long_range['end'],
                long_range.get('step', 1)
            ))
            
        return sorted(list(set(periods))) if periods else [5, 10, 20, 60]

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators at once for the entire dataset."""
        # Setup initial variables
        price_col = self.config['features']['price_col']
        tickers = data.index.get_level_values('ticker').unique()

        # Get standard periods
        periods = self._get_indicator_periods()
        
        # Pre-allocate list for features
        all_features = []
        for ticker in tickers:
            # Get data for this ticker, maintaining original structure
            ticker_mask = data.index.get_level_values('ticker') == ticker
            ticker_data = data[ticker_mask].sort_index()
            
            # Extract values while preserving trading day alignment
            price_values = ticker_data[price_col].values
            high_values = ticker_data['high'].values if 'high' in ticker_data.columns else None
            low_values = ticker_data['low'].values if 'low' in ticker_data.columns else None
            volume_values = ticker_data['volume'].values if 'volume' in ticker_data.columns else None

            # Initialize features dictionary for this ticker
            ticker_features = {}
            
            # Basic momentum & trend indicators
            if self.tech_indicators.get('rsi', {}).get('enabled'):
                for period in periods['rsi']:
                    ticker_features[f'rsi_{period}'] = talib.RSI(price_values, timeperiod=period)
            
            if self.tech_indicators.get('momentum', {}).get('enabled'):
                for period in periods['momentum']:
                    ticker_features[f'mom_{period}'] = talib.MOM(price_values, timeperiod=period)
                    ticker_features[f'roc_{period}'] = talib.ROC(price_values, timeperiod=period)
            
            # Advanced trend indicators
            if self.tech_indicators.get('macd', {}).get('enabled'):
                macd_config = periods['macd']
                fast = macd_config['fast'][0]
                slow = macd_config['slow'][0]
                signal = macd_config['signal'][0]
                macd, signal_line, hist = talib.MACD(
                    price_values,
                    fastperiod=fast,
                    slowperiod=slow,
                    signalperiod=signal
                )
                period_str = f'_{fast}_{slow}_{signal}'
                ticker_features[f'macd{period_str}'] = macd
                ticker_features[f'macd_signal{period_str}'] = signal_line
                ticker_features[f'macd_hist{period_str}'] = hist
            
            if self.tech_indicators.get('bollinger_bands', {}).get('enabled'):
                config = self.tech_indicators['bollinger_bands']
                for period in periods['bollinger']:
                    upper, middle, lower = talib.BBANDS(
                        price_values,
                        timeperiod=period,
                        nbdevup=config['nbdevup'],
                        nbdevdn=config['nbdevdn']
                    )
                    ticker_features[f'bb_upper_{period}'] = upper
                    ticker_features[f'bb_middle_{period}'] = middle
                    ticker_features[f'bb_lower_{period}'] = lower
                    
                    # Handle division by zero in Bollinger Band calculations
                    bandwidth = np.zeros_like(middle)
                    percent_b = np.zeros_like(middle)
                    
                    # Calculate bandwidth where middle is not zero
                    valid_middle = middle != 0
                    bandwidth[valid_middle] = (upper[valid_middle] - lower[valid_middle]) / middle[valid_middle]
                    
                    # Calculate percent_b where band width is not zero
                    valid_band = (upper - lower) != 0
                    percent_b[valid_band] = (price_values[valid_band] - lower[valid_band]) / (upper[valid_band] - lower[valid_band])
                    
                    ticker_features[f'bb_bandwidth_{period}'] = bandwidth
                    ticker_features[f'bb_percent_b_{period}'] = percent_b

            # Convert numpy arrays to DataFrame directly
            if ticker_features:  # Only create DataFrame if we have features
                ticker_df = pd.DataFrame(ticker_features, index=data.loc[ticker].index)
                all_features.append((ticker, ticker_df))
        
        if not all_features:
            return pd.DataFrame()
        
        # Create DataFrame with proper MultiIndex structure
        features = pd.concat([df for _, df in all_features], keys=[t for t, _ in all_features], names=['ticker', 'date'])
        
        # Sort index for better performance
        features = features.sort_index()
        
        # Log feature quality metrics
        num_missing = features.isna().sum().sum()
        logger.info(f"Raw features - shape: {features.shape}, missing values: {num_missing}")
        
        return features

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        logger.info("Creating technical indicator features...")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data index levels: {data.index.names}")
        
        # Ensure data is properly indexed
        if not isinstance(data.index, pd.MultiIndex):
            logger.warning("Input data is not multi-indexed")
            if 'ticker' in data.columns and 'date' in data.columns:
                data = data.set_index(['ticker', 'date'])
                logger.info("Set multi-index using ticker and date columns")
        
        # Calculate all technical indicators at once
        features = self._calculate_technical_indicators(data)
        
        if features.empty:
            logger.warning("No valid features could be calculated!")
            return features
            
        logger.info(f"Raw features shape: {features.shape}")
        logger.info(f"Features index levels: {features.index.names}")
        
        # Apply scaling if configured
        if self.config['features'].get('apply_scaling', True):
            features, self.scaler = self._apply_scaling(features)
        
        num_tickers = len(features.index.get_level_values('ticker').unique())
        logger.info(f"Created {len(features.columns)} features for {num_tickers} tickers")
        
        # Report data quality metrics
        num_missing = features.isna().sum().sum()
        if num_missing > 0:
            logger.info(f"Features contain {num_missing} missing values")
            missing_by_column = features.isna().sum()
            logger.debug("Missing values by feature:\n" + missing_by_column[missing_by_column > 0].to_string())
            
        return features

    def _apply_scaling(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Apply feature scaling to normalize features."""
        logger.info("Applying feature scaling...")
        
        if features.empty:
            logger.warning("Empty DataFrame provided. No scaling performed.")
            return features, None
        
        # Create and fit scaler on all features at once
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(features)
        
        # Convert scaled array back to DataFrame with original index and columns
        scaled_features = pd.DataFrame(
            scaled_array,
            columns=features.columns,
            index=features.index
        )
        
        logger.info(f"Feature scaling applied to {len(features.columns)} features")
        return scaled_features, scaler

    def save_scaler(self, output_path: str) -> bool:
        """Save the feature scaler to file."""
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
        """Load the feature scaler from file."""
        try:
            import pickle
            with open(input_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info(f"Loaded scaler from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading scaler from {input_path}: {str(e)}")
            return False
