"""
Unit tests for the feature engineering module.
"""

import unittest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Import the feature engineering module
import sys
sys.path.append('../src/features')  # Adjust path as needed
from feature_engineer import (
    calculate_returns,
    calculate_market_returns,
    calculate_excess_returns,
    create_lagged_features,
    create_target_variable,
    apply_feature_scaling,
    analyze_class_distribution,
    prepare_features_targets,
    FeatureEngineer
)

class TestFeatureEngineering(unittest.TestCase):
    """
    Test cases for feature engineering functions.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create date range
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        # Create multi-index data
        index = pd.MultiIndex.from_product(
            [tickers, dates],
            names=['ticker', 'date']
        )
        
        # Generate sample price data
        n = len(index)
        
        # Generate prices with trends to make the tests more realistic
        # Start with base prices for each ticker
        base_prices = {'AAPL': 100, 'MSFT': 150, 'GOOGL': 200}
        
        # Create price data with random walks
        prices = []
        for ticker in tickers:
            # Start with the base price
            price = base_prices[ticker]
            ticker_prices = []
            
            for _ in range(len(dates)):
                # Random daily change (-3% to +3%)
                daily_return = np.random.normal(0.0005, 0.02)  # Slight upward bias
                price = price * (1 + daily_return)
                ticker_prices.append(price)
            
            prices.extend(ticker_prices)
        
        # Create price DataFrame
        price_data = {
            'open': np.array(prices) * np.random.uniform(0.99, 1.0, n),
            'high': np.array(prices) * np.random.uniform(1.01, 1.02, n),
            'low': np.array(prices) * np.random.uniform(0.98, 0.99, n),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n)
        }
        
        self.price_data = pd.DataFrame(price_data, index=index)
        
        # Create returns data for testing
        self.returns_data = pd.DataFrame(index=index)
        
        # Generate realistic returns
        for ticker in tickers:
            ticker_prices = self.price_data.loc[ticker, 'close']
            ticker_returns = ticker_prices.pct_change()
            self.returns_data.loc[ticker, 'return'] = ticker_returns
        
        # Create market returns
        self.market_returns = self.returns_data.reset_index().groupby('date')['return'].mean()
        
        # Create excess returns
        self.excess_returns = pd.DataFrame(index=index)
        self.excess_returns['return'] = self.returns_data['return']
        self.excess_returns['market_return'] = self.excess_returns.index.get_level_values('date').map(self.market_returns)
        self.excess_returns['excess_return'] = self.excess_returns['return'] - self.excess_returns['market_return']
    
    def test_calculate_returns(self):
        """
        Test return calculation function.
        """
        result = calculate_returns(self.price_data)
        
        # Check if result has the expected structure
        self.assertEqual(len(result), len(self.price_data))
        self.assertIn('return', result.columns)
        
        # Check specific values (first day should be NaN, rest should be a number)
        for ticker in self.price_data.index.get_level_values('ticker').unique():
            ticker_result = result.loc[ticker]
            self.assertTrue(np.isnan(ticker_result['return'].iloc[0]))
            self.assertTrue(not np.isnan(ticker_result['return'].iloc[-1]))
    
    def test_calculate_market_returns(self):
        """
        Test market returns calculation.
        """
        result = calculate_market_returns(self.returns_data)
        
        # Check if result has the expected structure
        self.assertEqual(len(result), len(self.returns_data.index.get_level_values('date').unique()))
        
        # Check specific values
        self.assertTrue(all(isinstance(x, (int, float)) or np.isnan(x) for x in result))
    
    def test_calculate_excess_returns(self):
        """
        Test excess returns calculation.
        """
        result = calculate_excess_returns(self.returns_data, self.market_returns)
        
        # Check if result has the expected structure
        self.assertEqual(len(result), len(self.returns_data))
        self.assertIn('excess_return', result.columns)
        
        # Check if excess return = return - market_return
        for idx in result.index:
            if not np.isnan(result.loc[idx, 'return']) and not np.isnan(result.loc[idx, 'market_return']):
                self.assertAlmostEqual(
                    result.loc[idx, 'excess_return'],
                    result.loc[idx, 'return'] - result.loc[idx, 'market_return']
                )
    
    def test_create_lagged_features(self):
        """
        Test lagged feature creation.
        """
        max_lag = 5  # Small value for testing
        long_term_lags = [10, 20]  # Small values for testing
        
        result = create_lagged_features(
            self.excess_returns,
            'excess_return',
            max_lag,
            long_term_lags
        )
        
        # Check if result has the expected structure
        self.assertEqual(len(result), len(self.excess_returns))
        
        # Check if all expected features are created
        expected_cols = ([f'lag_{i}d' for i in range(1, max_lag + 1)] + 
                        [f'lag_{i}d' for i in long_term_lags] +
                        [f'cum_{i}d' for i in [5, 10, 20, 60, 120, 240]] +
                        [f'avg_{i}d' for i in [5, 10, 20, 60, 120, 240]] +
                        [f'vol_{i}d' for i in [5, 10, 20, 60, 120, 240]])
        
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Check if lags are correctly calculated
        # For a lag of 1, the value should be the previous day's value
        for ticker in self.excess_returns.index.get_level_values('ticker').unique():
            ticker_excess = self.excess_returns.loc[ticker, 'excess_return']
            ticker_lag1 = result.loc[ticker, 'lag_1d']
            
            # Check non-NaN values (skip first value which is NaN)
            for i in range(1, len(ticker_excess)):
                if not np.isnan(ticker_excess.iloc[i-1]) and not np.isnan(ticker_lag1.iloc[i]):
                    self.assertAlmostEqual(ticker_lag1.iloc[i], ticker_excess.iloc[i-1])
    
    def test_create_target_variable(self):
        """
        Test target variable creation.
        """
        result = create_target_variable(self.excess_returns)
        
        # Check if result has the expected structure
        self.assertEqual(len(result), len(self.excess_returns))
        self.assertIn('target', result.columns)
        
        # Check if target values are binary (0 or 1)
        self.assertTrue(all(x in [0, 1] or np.isnan(x) for x in result['target']))
        
        # Check if target values correctly reflect next day's excess return direction
        for ticker in self.excess_returns.index.get_level_values('ticker').unique():
            ticker_excess = self.excess_returns.loc[ticker, 'excess_return']
            ticker_target = result.loc[ticker, 'target']
            
            # Check non-NaN values (skip last value which would require next day data)
            for i in range(len(ticker_excess) - 1):
                next_return = ticker_excess.iloc[i+1]
                current_target = ticker_target.iloc[i]
                
                if not np.isnan(next_return) and not np.isnan(current_target):
                    expected_target = 1 if next_return > 0 else 0
                    self.assertEqual(current_target, expected_target)
    
    def test_apply_feature_scaling(self):
        """
        Test feature scaling.
        """
        # Create simple features for testing
        features = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        scaled_features, scaler = apply_feature_scaling(features)
        
        # Check if scaling preserves the DataFrame structure
        self.assertEqual(len(scaled_features), len(features))
        self.assertEqual(list(scaled_features.columns), list(features.columns))
        
        # Check if values are scaled (mean ~= 0, std ~= 1)
        for col in scaled_features.columns:
            self.assertAlmostEqual(scaled_features[col].mean(), 0, places=10)
            self.assertAlmostEqual(scaled_features[col].std(), 1, places=10)
    
    def test_analyze_class_distribution(self):
        """
        Test class distribution analysis.
        """
        # Create target data with known distribution
        index = self.excess_returns.index
        targets = pd.DataFrame(index=index)
        targets['target'] = np.random.choice([0, 1], size=len(index), p=[0.4, 0.6])
        
        result = analyze_class_distribution(targets)
        
        # Check if result has the expected structure
        self.assertIn('class_counts', result)
        self.assertIn('class_proportions', result)
        self.assertIn('majority_class', result)
        self.assertIn('majority_proportion', result)
        self.assertIn('is_severely_imbalanced', result)
        self.assertIn('class_weights', result)
        
        # Check if proportions match our setup (0.4, 0.6)
        self.assertAlmostEqual(result['class_proportions'][0], 0.4, places=1)
        self.assertAlmostEqual(result['class_proportions'][1], 0.6, places=1)
        
        # Check if majority class is correctly identified
        self.assertEqual(result['majority_class'], 1)
        
        # Check if severe imbalance is correctly detected (should be False for 0.6)
        self.assertFalse(result['is_severely_imbalanced'])
    
    def test_prepare_features_targets(self):
        """
        Test full feature engineering pipeline.
        """
        max_lag = 5  # Small value for testing
        
        result = prepare_features_targets(
            self.price_data,
            max_lag=max_lag,
            apply_scaling=True
        )
        
        # Check if result has the expected structure
        self.assertIn('features', result)
        self.assertIn('targets', result)
        self.assertIn('excess_returns', result)
        self.assertIn('feature_names', result)
        self.assertIn('class_distribution', result)
        self.assertIn('scaler', result)
        self.assertIn('n_samples', result)
        self.assertIn('n_features', result)
        
        # Check if features and targets have the same number of samples
        self.assertEqual(len(result['features']), len(result['targets']))
        
        # Check if lagged features were created
        for i in range(1, max_lag + 1):
            self.assertIn(f'lag_{i}d', result['feature_names'])
    
    def test_feature_engineer_class(self):
        """
        Test the FeatureEngineer class.
        """
        # Create a simple configuration
        config = {
            'price_col': 'close',
            'max_lag': 5,
            'long_term_lags': [10, 20],
            'apply_scaling': True
        }
        
        # Create a FeatureEngineer instance
        engineer = FeatureEngineer(config)
        
        # Create features
        result = engineer.create_features(self.price_data)
        
        # Check if features were created
        self.assertTrue('features' in result)
        self.assertTrue('targets' in result)
        
        # Check if scaler was stored
        self.assertIsNotNone(engineer.scaler)
        
        # Test scaler saving and loading
        temp_file = 'temp_scaler.pkl'
        
        try:
            # Save scaler
            save_success = engineer.save_scaler(temp_file)
            self.assertTrue(save_success)
            self.assertTrue(os.path.exists(temp_file))
            
            # Create a new instance and load the scaler
            new_engineer = FeatureEngineer(config)
            load_success = new_engineer.load_scaler(temp_file)
            
            self.assertTrue(load_success)
            self.assertIsNotNone(new_engineer.scaler)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == '__main__':
    unittest.main()
