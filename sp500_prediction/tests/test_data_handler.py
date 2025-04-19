"""
Unit tests for the data handler module.
"""

import unittest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Import the data handler module
import sys
sys.path.append('../src/data')  # Adjust path as needed
from data_handler import (
    read_sp500_data,
    clean_sp500_data,
    get_ticker_metadata,
    split_train_test_data,
    create_time_based_cv_folds,
    DataHandler
)

class TestDataHandler(unittest.TestCase):
    """
    Test cases for data handler functions.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a sample DataFrame
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        # Create multi-index data
        index = pd.MultiIndex.from_product(
            [tickers, dates],
            names=['ticker', 'date']
        )
        
        # Generate sample price data
        n = len(index)
        data = {
            'open': np.random.uniform(100, 200, n),
            'high': np.random.uniform(100, 200, n),
            'low': np.random.uniform(100, 200, n),
            'close': np.random.uniform(100, 200, n),
            'volume': np.random.randint(1000000, 10000000, n)
        }
        
        self.sample_data = pd.DataFrame(data, index=index)
        
        # Create DataFrame with missing values
        self.missing_data = self.sample_data.copy()
        mask = np.random.random(n) < 0.1  # 10% missing data
        for col in self.missing_data.columns:
            self.missing_data.loc[mask, col] = np.nan
        
        # Create data for specific years
        self.year_2020_data = self.sample_data.loc[:, self.sample_data.index.get_level_values('date').year == 2020]
        self.year_2021_data = self.sample_data.loc[:, self.sample_data.index.get_level_values('date').year == 2021]
        self.year_2022_data = self.sample_data.loc[:, self.sample_data.index.get_level_values('date').year == 2022]
    
    def test_clean_sp500_data(self):
        """
        Test data cleaning function.
        """
        cleaned_data = clean_sp500_data(self.missing_data)
        
        # Check if all NaN values are removed
        self.assertFalse(cleaned_data.isnull().any().any())
        
        # Check if multi-index is preserved
        self.assertTrue(isinstance(cleaned_data.index, pd.MultiIndex))
        self.assertEqual(cleaned_data.index.names, ['ticker', 'date'])
    
    def test_get_ticker_metadata(self):
        """
        Test ticker metadata generation.
        """
        metadata = get_ticker_metadata(self.sample_data)
        
        # Check if metadata has the expected columns
        expected_columns = ['ticker', 'first_date', 'last_date', 'total_trading_days', 'coverage_ratio']
        for col in expected_columns:
            self.assertIn(col, metadata.columns)
        
        # Check if all tickers are included
        tickers = self.sample_data.index.get_level_values('ticker').unique()
        self.assertEqual(set(metadata['ticker']), set(tickers))
    
    def test_split_train_test_data(self):
        """
        Test train-test splitting function.
        """
        train_years = [2020, 2021]
        test_years = [2022]
        
        train_data, test_data, eligible_tickers = split_train_test_data(
            self.sample_data,
            train_years,
            test_years,
            min_history_days=10  # Lower for testing
        )
        
        # Check if train data is from train years
        train_years_in_data = set(train_data.reset_index()['date'].dt.year.unique())
        self.assertEqual(train_years_in_data, set(train_years))
        
        # Check if test data is from test years
        test_years_in_data = set(test_data.reset_index()['date'].dt.year.unique())
        self.assertEqual(test_years_in_data, set(test_years))
        
        # Check if eligible tickers are in both train and test sets
        train_tickers = set(train_data.index.get_level_values('ticker').unique())
        test_tickers = set(test_data.index.get_level_values('ticker').unique())
        
        for ticker in eligible_tickers:
            self.assertIn(ticker, train_tickers)
            self.assertIn(ticker, test_tickers)
    
    def test_create_time_based_cv_folds(self):
        """
        Test creation of time-based CV folds.
        """
        num_folds = 2
        folds = create_time_based_cv_folds(
            self.sample_data,
            start_year=2020,
            num_folds=num_folds,
            train_window_years=1,
            test_window_years=1,
            min_history_days=10  # Lower for testing
        )
        
        # Check if correct number of folds is created
        self.assertEqual(len(folds), num_folds)
        
        # Check fold structure
        for i, fold in enumerate(folds):
            self.assertEqual(fold['fold_num'], i + 1)
            self.assertEqual(len(fold['train_years']), 1)
            self.assertEqual(len(fold['test_years']), 1)
            self.assertIsInstance(fold['train_data'], pd.DataFrame)
            self.assertIsInstance(fold['test_data'], pd.DataFrame)
            self.assertIsInstance(fold['eligible_tickers'], list)
    
    def test_data_handler_class(self):
        """
        Test the DataHandler class.
        """
        handler = DataHandler()
        
        # Create a temporary file for testing
        temp_file = 'temp_test_data.h5'
        
        try:
            # Store data directly in the handler
            handler.data = self.sample_data
            handler.metadata = {'test': 'metadata'}
            handler.ticker_metadata = get_ticker_metadata(self.sample_data)
            
            # Test save functionality
            save_success = handler.save_processed_data(temp_file)
            self.assertTrue(save_success)
            self.assertTrue(os.path.exists(temp_file))
            
            # Test load functionality
            new_handler = DataHandler()
            load_success = new_handler.load_processed_data(temp_file)
            
            self.assertTrue(load_success)
            self.assertEqual(len(new_handler.data), len(self.sample_data))
            
            # Test get_data_for_years
            year_data = new_handler.get_data_for_years([2021])
            self.assertTrue(all(d.year == 2021 for d in year_data.reset_index()['date']))
            
            # Test create_cv_folds
            cv_folds = new_handler.create_cv_folds(
                start_year=2020,
                num_folds=2,
                train_window_years=1,
                test_window_years=1
            )
            self.assertEqual(len(cv_folds), 2)
            
            # Test get_train_test_split
            train_data, test_data, eligible_tickers = new_handler.get_train_test_split(
                train_years=[2020, 2021],
                test_years=[2022]
            )
            self.assertTrue(len(train_data) > 0)
            self.assertTrue(len(test_data) > 0)
            self.assertTrue(len(eligible_tickers) > 0)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == '__main__':
    unittest.main()