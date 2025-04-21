"""
Feature Engineering Module for S&P500 Prediction Project

This module handles the calculation of technical indicators and targets
for the prediction model.
"""

import pandas as pd
import numpy as np
import talib
import logging
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class to handle feature engineering for S&P500 prediction.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the FeatureEngineer with configuration.
        
        Parameters:
        -----------
        config : Dict, optional
            Configuration dictionary with options for feature engineering
        """
        self.config = config or {}
        self.tech_indicators = self.config.get('features', {}).get('technical_indicators', {})
        self.target_config = self.config.get('target', {})
        self.scaler = None

    def _get_max_window_size(self) -> int:
        """Calculate the maximum window size needed for all features and targets."""
        max_window = 1
        
        # Check technical indicators
        if self.tech_indicators.get('momentum', {}).get('enabled'):
            timeperiods = self._generate_timeperiods(self.tech_indicators['momentum'])
            if timeperiods:
                max_window = max(max_window, max(timeperiods))
        
        if self.tech_indicators.get('bollinger_bands', {}).get('enabled'):
            periods = self._generate_bollinger_periods(self.tech_indicators['bollinger_bands'])
            if periods:
                max_window = max(max_window, max(periods))
        
        if self.tech_indicators.get('rsi', {}).get('enabled'):
            periods = self._generate_rsi_periods(self.tech_indicators['rsi'])
            if periods:
                max_window = max(max_window, max(periods))
        
        if self.tech_indicators.get('macd', {}).get('enabled'):
            periods = self._generate_macd_periods(self.tech_indicators['macd'])
            max_window = max(max_window, max(periods['slow']))
        
        # Check target calculation window
        target_window = self.target_config['calculation']['rolling_window']
        max_window = max(max_window, target_window)
        
        # Check target horizon
        max_horizon = max(self.target_config['calculation']['horizon'])
        max_window = max(max_window, max_horizon)
        
        # Add extra day for calculation
        return max_window + 1

    def _get_valid_data_range(self, ticker_data: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Find first and last valid data point for a ticker."""
        valid_mask = ~ticker_data.isna()
        if not valid_mask.any():
            return None, None
        first_valid = ticker_data[valid_mask].index[0]
        last_valid = ticker_data[valid_mask].index[-1]
        return first_valid, last_valid

    def validate_ticker_data(self, ticker_data: pd.DataFrame, min_required_days: int) -> bool:
        """Check if ticker has enough valid data for feature/target calculation."""
        price_col = self.config['features']['price_col']
        valid_data = ~ticker_data[price_col].isna()
        if not valid_data.any():
            return False
        
        # Count consecutive valid data points
        consecutive_valid = valid_data.astype(int).groupby(
            (valid_data.astype(int).diff() != 0).cumsum()
        ).sum()
        return (consecutive_valid >= min_required_days).any()

    def _get_valid_calculation_periods(self, ticker_data: pd.DataFrame) -> pd.DatetimeIndex:
        """Get valid periods for feature and target calculation."""
        price_col = self.config['features']['price_col']
        
        # Get valid data range
        first_valid, last_valid = self._get_valid_data_range(ticker_data[price_col])
        if first_valid is None:
            return pd.DatetimeIndex([])
        
        # Calculate required windows
        feature_window = self._get_max_window_size()
        target_window = max(self.target_config['calculation']['horizon'])
        
        # Calculate valid periods
        feature_start = first_valid + pd.Timedelta(days=feature_window)
        target_end = last_valid - pd.Timedelta(days=target_window)
        
        if target_end < feature_start:
            return pd.DatetimeIndex([])
        
        # Get valid indices
        return ticker_data.loc[feature_start:target_end].index

    def _generate_timeperiods(self, config: Dict) -> List[int]:
        """Generate timeperiods based on configuration."""
        if 'timeperiod_generation' not in config:
            return config.get('timeperiods', [5, 10, 20, 60])
            
        gen_config = config['timeperiod_generation']
        if gen_config['method'] != 'range':
            logger.warning(f"Unsupported timeperiod generation method: {gen_config['method']}")
            return config.get('timeperiods', [5, 10, 20, 60])
            
        params = gen_config['params']
        timeperiods = []
        
        # Generate short-range periods
        if 'short_range' in params:
            short_range = params['short_range']
            timeperiods.extend(range(
                short_range['start'],
                short_range['end'],
                short_range.get('step', 1)
            ))
            
        # Generate long-range periods
        if 'long_range' in params:
            long_range = params['long_range']
            timeperiods.extend(range(
                long_range['start'],
                long_range['end'],
                long_range.get('step', 1)
            ))
            
        if not timeperiods:
            logger.warning("No timeperiods generated, using defaults")
            return [5, 10, 20, 60]
            
        return sorted(list(set(timeperiods)))  # Remove duplicates and sort

    def _generate_single_range(self, config: Dict) -> List[int]:
        """Generate timeperiods from a single range configuration."""
        if 'range' not in config['params']:
            return []
            
        range_config = config['params']['range']
        return list(range(
            range_config['start'],
            range_config['end'],
            range_config.get('step', 1)
        ))

    def _generate_rsi_periods(self, config: Dict) -> List[int]:
        """Generate RSI timeperiods based on configuration."""
        if 'timeperiod_generation' not in config:
            return [config.get('timeperiod', 14)]
            
        if config['timeperiod_generation']['method'] != 'range':
            logger.warning(f"Unsupported RSI period generation method")
            return [config.get('timeperiod', 14)]
            
        periods = self._generate_single_range(config['timeperiod_generation'])
        return periods if periods else [14]

    def _generate_bollinger_periods(self, config: Dict) -> List[int]:
        """Generate Bollinger Bands timeperiods based on configuration."""
        if 'timeperiod_generation' not in config:
            return [config.get('timeperiod', 20)]
            
        if config['timeperiod_generation']['method'] != 'range':
            logger.warning(f"Unsupported Bollinger Bands period generation method")
            return [config.get('timeperiod', 20)]
            
        periods = self._generate_single_range(config['timeperiod_generation'])
        return periods if periods else [20]

    def _generate_macd_periods(self, config: Dict) -> Dict[str, List[int]]:
        """Generate MACD periods based on configuration."""
        if 'period_generation' not in config:
            return {
                'fast': [config.get('fastperiod', 12)],
                'slow': [config.get('slowperiod', 26)],
                'signal': [config.get('signalperiod', 9)]
            }
            
        gen_config = config['period_generation']
        periods = {
            'fast': list(range(
                gen_config['fast']['start'],
                gen_config['fast']['end'],
                gen_config['fast'].get('step', 1)
            )) if 'fast' in gen_config else [12],
            
            'slow': list(range(
                gen_config['slow']['start'],
                gen_config['slow']['end'],
                gen_config['slow'].get('step', 1)
            )) if 'slow' in gen_config else [26],
            
            'signal': list(range(
                gen_config['signal']['start'],
                gen_config['signal']['end'],
                gen_config['signal'].get('step', 1)
            )) if 'signal' in gen_config else [9]
        }
        
        # Ensure we have at least default values
        if not periods['fast']: periods['fast'] = [12]
        if not periods['slow']: periods['slow'] = [26]
        if not periods['signal']: periods['signal'] = [9]
        
        return periods

    def _calculate_momentum_indicators(self, price_values: np.ndarray, config: Dict) -> Dict[str, np.ndarray]:
        """Calculate momentum indicators based on configuration."""
        features = {}
        
        # Generate timeperiods from configuration
        timeperiods = self._generate_timeperiods(config)
        logger.info(f"Calculating momentum indicators for periods: {timeperiods}")
        
        # Get enabled types
        types = config.get('types', {'momentum': True, 'roc': True})
        
        # Calculate indicators for each period
        for period in timeperiods:
            if types.get('momentum', True):
                features[f'mom_{period}'] = talib.MOM(price_values, timeperiod=period)
            if types.get('roc', True):
                features[f'roc_{period}'] = talib.ROC(price_values, timeperiod=period)
                
        return features

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators at once for the entire dataset."""
        price_col = self.config['features']['price_col']
        feature_dfs = []
        
        for ticker in data.index.get_level_values('ticker').unique():
            price_series = data.loc[ticker, price_col]
            if price_series.isna().all():
                continue
                
            price_values = price_series.values
            ticker_features = {}
            
            # Calculate RSI
            if self.tech_indicators.get('rsi', {}).get('enabled'):
                rsi_periods = self._generate_rsi_periods(self.tech_indicators['rsi'])
                for period in rsi_periods:
                    ticker_features[f'rsi_{period}'] = talib.RSI(price_values, timeperiod=period)
            
            # Calculate MACD
            if self.tech_indicators.get('macd', {}).get('enabled'):
                macd_periods = self._generate_macd_periods(self.tech_indicators['macd'])
                for fast in macd_periods['fast']:
                    for slow in macd_periods['slow']:
                        for signal in macd_periods['signal']:
                            if slow <= fast:  # Skip invalid combinations
                                continue
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
            
            # Calculate Bollinger Bands
            if self.tech_indicators.get('bollinger_bands', {}).get('enabled'):
                config = self.tech_indicators['bollinger_bands']
                bb_periods = self._generate_bollinger_periods(config)
                for period in bb_periods:
                    upper, middle, lower = talib.BBANDS(
                        price_values,
                        timeperiod=period,
                        nbdevup=config['nbdevup'],
                        nbdevdn=config['nbdevdn']
                    )
                    ticker_features[f'bb_upper_{period}'] = upper
                    ticker_features[f'bb_middle_{period}'] = middle
                    ticker_features[f'bb_lower_{period}'] = lower
                    ticker_features[f'bb_bandwidth_{period}'] = (upper - lower) / middle
                    ticker_features[f'bb_percent_b_{period}'] = (price_values - lower) / (upper - lower)
            
            # Calculate Momentum indicators
            if self.tech_indicators.get('momentum', {}).get('enabled'):
                momentum_features = self._calculate_momentum_indicators(
                    price_values, 
                    self.tech_indicators['momentum']
                )
                ticker_features.update(momentum_features)
            
            # Create DataFrame for this ticker
            ticker_df = pd.DataFrame(ticker_features, index=price_series.index)
            ticker_df['ticker'] = ticker
            feature_dfs.append(ticker_df)
        
        if not feature_dfs:
            return pd.DataFrame()
            
        features = pd.concat(feature_dfs, axis=0)
        features.set_index('ticker', append=True, inplace=True)
        features = features.reorder_levels(['ticker', 'date'])
        
        return features

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        logger.info("Creating technical indicator features...")
        
        # Calculate all technical indicators at once
        features = self._calculate_technical_indicators(data)
        
        if features.empty:
            logger.warning("No valid features could be calculated!")
            return features
        
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

    def _calculate_returns(self, data: pd.DataFrame, horizon: int, return_type: str = 'raw') -> pd.Series:
        """Calculate returns for a given horizon and type."""
        price_col = self.config['features']['price_col']
        returns = pd.Series(index=data.index)
        
        for ticker in data.index.get_level_values('ticker').unique():
            ticker_data = data.loc[ticker][price_col]
            
            if return_type == 'raw':
                returns.loc[ticker] = ticker_data.pct_change(horizon).shift(-horizon)
            elif return_type == 'excess':
                # Calculate market average return
                market_return = data[price_col].groupby('date').mean().pct_change(horizon).shift(-horizon)
                ticker_return = ticker_data.pct_change(horizon).shift(-horizon)
                returns.loc[ticker] = ticker_return - market_return
            elif return_type == 'log':
                returns.loc[ticker] = (np.log(ticker_data) - np.log(ticker_data.shift(horizon))).shift(-horizon)
            elif return_type == 'percentage':
                returns.loc[ticker] = ((ticker_data - ticker_data.shift(horizon)) / ticker_data.shift(horizon) * 100).shift(-horizon)
                
        return returns

    def _create_binary_target(self, returns: pd.Series, calc_config: Dict) -> pd.Series:
        """Create binary target based on configuration method."""
        method = calc_config['method']
        target = pd.Series(index=returns.index)
        
        if method == 'threshold_based':
            threshold = calc_config['fixed_threshold']
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
            rolling_quantile = returns.groupby(level='ticker').transform(
                lambda x: x.rolling(window=calc_config['rolling_window'], min_periods=1).quantile(0.5)
            )
            target = (returns > rolling_quantile).astype(int)
            
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
            
        return target

    def create_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variables based on configuration."""
        logger.info("Creating target variables...")
        
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
            
            # Create targets based on type
            if target_type == 'returns':
                targets[f'target_{horizon}d'] = returns
                
            elif target_type == 'binary':
                targets[f'target_{horizon}d'] = self._create_binary_target(returns, calc_config)
                
            elif target_type == 'multiclass':
                targets[f'target_{horizon}d'] = self._create_multiclass_target(returns, calc_config)
                
            else:
                raise ValueError(f"Unsupported target type: {target_type}")
        
        logger.info(f"Created {target_type} targets for horizons: {calc_config['horizon']}")
        
        # Log target distribution
        for col in targets.columns:
            value_counts = targets[col].value_counts(normalize=True)
            logger.info(f"Target distribution for {col}:\n{value_counts}")
            
        return targets

    def create_feature_target_dataset(self, data: pd.DataFrame) -> Dict:
        """Create complete feature and target dataset for full period."""
        logger.info("Creating complete feature and target dataset...")
        
        # Calculate minimum required samples
        min_required = self._get_max_window_size()
        logger.info(f"Minimum required samples: {min_required}")
        
        # For each ticker, determine valid indices
        valid_indices = []
        for ticker in data.index.get_level_values('ticker').unique():
            ticker_data = data.loc[ticker]
            # Skip if ticker doesn't have enough data
            if len(ticker_data) <= min_required:
                logger.warning(f"Insufficient data for ticker {ticker}, skipping...")
                continue
            # Get indices after lookback period
            valid_dates = ticker_data.index[min_required:]
            valid_indices.extend([(ticker, date) for date in valid_dates])
        
        if not valid_indices:
            logger.warning("No valid data after applying lookback period!")
            return {
                'features': pd.DataFrame(),
                'targets': pd.DataFrame(),
                'metadata': {
                    'min_required_samples': min_required,
                    'error': 'No valid data available'
                }
            }
        
        # Calculate features and targets
        features = self.create_features(data)
        targets = self.create_target(data)
        
        # Filter to valid indices only
        valid_indices = pd.MultiIndex.from_tuples(valid_indices, names=['ticker', 'date'])
        features = features.loc[valid_indices]
        targets = targets.loc[valid_indices]
        
        # Create metadata
        metadata = {
            'min_required_samples': self._get_max_window_size(),
            'feature_columns': list(features.columns),
            'target_columns': list(targets.columns),
            'data_start_date': features.index.get_level_values('date').min(),
            'data_end_date': features.index.get_level_values('date').max(),
            'num_samples': len(features),
            'num_features': len(features.columns),
            'num_targets': len(targets.columns),
            'num_tickers': len(features.index.get_level_values('ticker').unique()),
            'data_quality': {
                'missing_values': features.isna().sum().to_dict(),
                'unique_classes': {col: sorted(targets[col].unique().tolist()) 
                                 for col in targets.columns}
            }
        }
        
        logger.info(f"Created dataset with {metadata['num_samples']} samples")
        logger.info(f"Features: {metadata['num_features']}, Targets: {metadata['num_targets']}")
        logger.info(f"Date range: {metadata['data_start_date']} to {metadata['data_end_date']}")
        
        return {
            'features': features,
            'targets': targets,
            'metadata': metadata
        }

    def _apply_scaling(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
        """Apply feature scaling to normalize features."""
        logger.info("Applying feature scaling...")
        
        if features.empty:
            logger.warning("Empty DataFrame provided. No scaling performed.")
            return features, {}
        
        # Create scalers dictionary
        scalers = {}
        scaled_features = pd.DataFrame(index=features.index)
        
        # Process each feature separately
        for col in features.columns:
            # Create scaler for this feature
            scaler = StandardScaler()
            
            # Reshape data and fit scaler
            feature_values = features[col].values.reshape(-1, 1)
            scaled_values = scaler.fit_transform(feature_values)
            
            # Store scaled values and scaler
            scaled_features[col] = scaled_values.flatten()
            scalers[col] = scaler
        
        logger.info(f"Feature scaling applied to {len(features.columns)} features")
        return scaled_features, scalers

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
