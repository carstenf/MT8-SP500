"""
Data Handler Module for S&P500 Prediction Project

This module handles loading, cleaning, and preprocessing of S&P500 stock data.
It provides functions to prepare data for feature engineering and model training.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_sp500_data(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Read S&P 500 data from HDF5 file with flexible querying options.
    Works with the new modularized market data structure.
    
    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file containing S&P 500 data
        
    Returns:
    --------
    tuple: (DataFrame, dict)
        - DataFrame with S&P 500 data 
        - Dictionary with metadata information
    """
    logger.info(f"Reading data from {file_path}...")
    
    try:
        # Open the HDF5 file
        with pd.HDFStore(file_path, mode='r') as store:
            # Get available keys
            keys = store.keys()
            logger.info(f"Available keys in file: {keys}")
            
            # Select the appropriate dataset - merged_data contains prices and shares
            if '/market_data' in keys:
                data_key = 'market_data'
            else:
                logger.error(f"Required data keys not found in {file_path}")
                return pd.DataFrame(), {}
            
            # Read the data
            data = store.get(data_key)
            
            # Collect metadata
            metadata = {
                'source_file': file_path,
                'keys_available': [k.strip('/') for k in keys],
                'data_key_used': data_key,
                'original_records': len(data)
            }
            
            # If data is not indexed properly, set appropriate index
            if not isinstance(data.index, pd.MultiIndex) and 'ticker' in data.columns and 'date' in data.columns:
                data = data.set_index(['ticker', 'date'])
            elif not isinstance(data.index, pd.MultiIndex) and 'date' in data.columns:
                data = data.set_index('date')
                        
            # Update metadata with filtered data info
            metadata.update({
                'num_records': len(data),
                'num_tickers': len(data.index.get_level_values('ticker').unique()) if isinstance(data.index, pd.MultiIndex) else 'N/A',
                'start_date': data.index.get_level_values('date').min() if isinstance(data.index, pd.MultiIndex) else data.index.min(),
                'end_date': data.index.get_level_values('date').max() if isinstance(data.index, pd.MultiIndex) else data.index.max()
            })
            
            logger.info(f"Retrieved {len(data)} records")
            return data, metadata
            
    except Exception as e:
        logger.error(f"Error reading data from {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}  # Return empty DataFrame and dict on error


def clean_sp500_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the S&P500 data by handling missing values and ensuring consistent format.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw S&P500 data with multi-index (ticker, date)
        
    Returns:
    --------
    pd.DataFrame
        Cleaned S&P500 data
    """
    logger.info("Cleaning S&P500 data...")
    
    if data.empty:
        logger.warning("Empty DataFrame provided. No cleaning performed.")
        return data
    
    # Make a copy to avoid modifying the original data
    cleaned_data = data.copy()
    
    # Ensure proper multi-index format (ticker, date)
    if not isinstance(cleaned_data.index, pd.MultiIndex):
        if 'ticker' in cleaned_data.columns and 'date' in cleaned_data.columns:
            logger.info("Converting to multi-index format (ticker, date)")
            cleaned_data = cleaned_data.set_index(['ticker', 'date'])
        else:
            logger.error("Cannot convert to multi-index: missing ticker or date columns")
            return cleaned_data
    
    # Check multi-index order - should be (ticker, date)
    index_names = cleaned_data.index.names
    if index_names[0] != 'ticker' or index_names[1] != 'date':
        logger.info("Reordering multi-index to (ticker, date) format")
        cleaned_data = cleaned_data.reorder_levels(['ticker', 'date'])

    # Essential columns for prediction - adjust based on your data structure
    essential_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in essential_columns if col not in cleaned_data.columns]
    
    if missing_columns:
        logger.warning(f"Missing essential columns: {missing_columns}")
    
    # Drop rows with missing values in essential columns
    # Only consider columns that are present in the data
    present_essential_cols = [col for col in essential_columns if col in cleaned_data.columns]
    if present_essential_cols:
        before_count = len(cleaned_data)
        cleaned_data = cleaned_data.dropna(subset=present_essential_cols)
        after_count = len(cleaned_data)
        logger.info(f"Dropped {before_count - after_count} rows with missing values in essential columns")
    
    return cleaned_data


def get_ticker_metadata(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create metadata for each ticker, including start and end dates and data availability.
    
    Parameters:
    -----------
    data : pd.DataFrame
        S&P500 data with multi-index (ticker, date)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with ticker metadata
    """
    logger.info("Generating ticker metadata...")
    
    if data.empty:
        logger.warning("Empty DataFrame provided. No metadata generated.")
        return pd.DataFrame()
    
    # Ensure we have a multi-index DataFrame
    if not isinstance(data.index, pd.MultiIndex):
        logger.error("DataFrame must have a multi-index (ticker, date)")
        return pd.DataFrame()
    
    # Extract all unique tickers
    tickers = data.index.get_level_values('ticker').unique()
    
    # Initialize list to store ticker metadata
    ticker_metadata = []
    
    for ticker in tickers:
        # Get data for this ticker
        ticker_data = data.loc[ticker]
        
        # Get first and last date
        first_date = ticker_data.index.min()
        last_date = ticker_data.index.max()
        
        # Count total trading days
        total_days = len(ticker_data)
        
        # Count total calendar days in the date range
        calendar_days = (last_date - first_date).days + 1
        
        # Calculate coverage ratio (trading days / calendar days)
        # This gives an idea of data completeness
        coverage_ratio = total_days / max(calendar_days / 7 * 5, 1)  # Approximate trading days
        
        # Store metadata
        ticker_metadata.append({
            'ticker': ticker,
            'first_date': first_date,
            'last_date': last_date,
            'total_trading_days': total_days,
            'coverage_ratio': coverage_ratio
        })
    
    # Convert to DataFrame
    metadata_df = pd.DataFrame(ticker_metadata)
    
    logger.info(f"Generated metadata for {len(metadata_df)} tickers")
    return metadata_df


def split_train_test_data(
    data: pd.DataFrame, 
    train_years: List[int], 
    test_years: List[int],
    min_history_days: int = 240
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Split data into training and testing sets based on years,
    and filter tickers to include only those with sufficient data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        S&P500 data with multi-index (ticker, date)
    train_years : List[int]
        List of years to include in the training set
    test_years : List[int]
        List of years to include in the testing set
    min_history_days : int, optional
        Minimum number of trading days required for feature calculation
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        - Training data
        - Testing data
        - List of eligible tickers
    """
    logger.info(f"Splitting data into train ({train_years}) and test ({test_years}) years...")
    
    if data.empty:
        logger.warning("Empty DataFrame provided. No splitting performed.")
        return pd.DataFrame(), pd.DataFrame(), []
    
    # Create year column if working with multi-index
    if isinstance(data.index, pd.MultiIndex):
        # Extract dates from the index
        all_data = data.reset_index()
        all_data['year'] = all_data['date'].dt.year
    else:
        logger.error("DataFrame must have a multi-index (ticker, date)")
        return pd.DataFrame(), pd.DataFrame(), []
    
    # Filter data for train and test years
    train_data = all_data[all_data['year'].isin(train_years)]
    test_data = all_data[all_data['year'].isin(test_years)]
    
    # Find tickers that have sufficient data in both train and test periods
    train_ticker_counts = train_data.groupby('ticker').size()
    test_ticker_counts = test_data.groupby('ticker').size()
    
    # Find common tickers with sufficient data
    eligible_tickers = [
        ticker for ticker in train_ticker_counts.index
        if ticker in test_ticker_counts.index
        and train_ticker_counts[ticker] >= min_history_days
        and test_ticker_counts[ticker] > 0
    ]
    
    logger.info(f"Found {len(eligible_tickers)} eligible tickers with sufficient data")
    
    # Filter data to include only eligible tickers
    train_data = train_data[train_data['ticker'].isin(eligible_tickers)]
    test_data = test_data[test_data['ticker'].isin(eligible_tickers)]
    
    # Reset index and set multi-index again
    train_data = train_data.set_index(['ticker', 'date']).sort_index()
    test_data = test_data.set_index(['ticker', 'date']).sort_index()
    
    logger.info(f"Training set: {len(train_data)} records")
    logger.info(f"Testing set: {len(test_data)} records")
    
    return train_data, test_data, eligible_tickers


def create_time_based_cv_folds(
    data: pd.DataFrame,
    start_year: int,
    num_folds: int = 5,
    train_window_years: int = 3,
    test_window_years: int = 1,
    min_history_days: int = 240
) -> List[Dict]:
    """
    Create time-based cross-validation folds with sliding windows.
    
    Parameters:
    -----------
    data : pd.DataFrame
        S&P500 data with multi-index (ticker, date)
    start_year : int
        First year to use in the first training fold
    num_folds : int, optional
        Number of folds to create
    train_window_years : int, optional
        Number of years in each training window
    test_window_years : int, optional
        Number of years in each testing window
    min_history_days : int, optional
        Minimum number of trading days required for feature calculation
        
    Returns:
    --------
    List[Dict]
        List of dictionaries, each containing train_years, test_years, 
        train_data, test_data, and eligible_tickers for a fold
    """
    logger.info(f"Creating {num_folds} time-based CV folds starting from year {start_year}...")
    
    if data.empty:
        logger.warning("Empty DataFrame provided. No CV folds created.")
        return []
    
    # Create year column if working with multi-index
    if isinstance(data.index, pd.MultiIndex):
        # Extract dates from the index
        all_data = data.reset_index()
        all_data['year'] = all_data['date'].dt.year
    else:
        logger.error("DataFrame must have a multi-index (ticker, date)")
        return []
    
    # Create folds
    folds = []
    
    for fold_num in range(num_folds):
        # Calculate train and test years for this fold
        train_start_year = start_year + fold_num
        train_end_year = train_start_year + train_window_years - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + test_window_years - 1
        
        train_years = list(range(train_start_year, train_end_year + 1))
        test_years = list(range(test_start_year, test_end_year + 1))
        
        # Filter data for this fold
        fold_train_data = all_data[all_data['year'].isin(train_years)]
        fold_test_data = all_data[all_data['year'].isin(test_years)]
        
        # Find eligible tickers for this fold
        train_ticker_counts = fold_train_data.groupby('ticker').size()
        test_ticker_counts = fold_test_data.groupby('ticker').size()
        
        eligible_tickers = [
            ticker for ticker in train_ticker_counts.index
            if ticker in test_ticker_counts.index
            and train_ticker_counts[ticker] >= min_history_days
            and test_ticker_counts[ticker] > 0
        ]
        
        # Filter data to include only eligible tickers
        fold_train_data = fold_train_data[fold_train_data['ticker'].isin(eligible_tickers)]
        fold_test_data = fold_test_data[fold_test_data['ticker'].isin(eligible_tickers)]
        
        # Reset index and set multi-index again
        fold_train_data = fold_train_data.set_index(['ticker', 'date']).sort_index()
        fold_test_data = fold_test_data.set_index(['ticker', 'date']).sort_index()
        
        # Store fold information
        fold_info = {
            'fold_num': fold_num + 1,
            'train_years': train_years,
            'test_years': test_years,
            'train_data': fold_train_data,
            'test_data': fold_test_data,
            'eligible_tickers': eligible_tickers,
            'num_eligible_tickers': len(eligible_tickers)
        }
        
        folds.append(fold_info)
        
        logger.info(f"Fold {fold_num + 1}: "
                   f"Train years {train_years}, "
                   f"Test years {test_years}, "
                   f"{len(eligible_tickers)} eligible tickers")
    
    return folds


class DataHandler:
    """
    Class to handle all data operations for S&P500 prediction.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the DataHandler with optional configuration.
        
        Parameters:
        -----------
        config : Dict, optional
            Configuration dictionary with options for data handling
        """
        self.config = config or {}
        self.data = pd.DataFrame()
        self.metadata = {}
        self.ticker_metadata = pd.DataFrame()
        
    def load_data(self, file_path: str) -> bool:
        """
        Load data from file and perform initial cleaning.
        
        Parameters:
        -----------
        file_path : str
            Path to the HDF5 file containing S&P500 data
            
        Returns:
        --------
        bool
            True if data loading succeeded, False otherwise
        """
        # Load raw data
        data, metadata = read_sp500_data(file_path)
        
        if data.empty:
            logger.error("Failed to load data.")
            return False
        
        # Clean the data
        cleaned_data = clean_sp500_data(data)
        
        if cleaned_data.empty:
            logger.error("Data cleaning resulted in empty dataset.")
            return False
        
        # Generate ticker metadata
        ticker_metadata = get_ticker_metadata(cleaned_data)
        
        # Store the results
        self.data = cleaned_data
        self.metadata = metadata
        self.ticker_metadata = ticker_metadata
        
        logger.info(f"Successfully loaded and cleaned data with {len(cleaned_data)} records")
        return True
    
    def get_data_for_years(self, years: List[int]) -> pd.DataFrame:
        """
        Get data for specific years.
        
        Parameters:
        -----------
        years : List[int]
            List of years to include
            
        Returns:
        --------
        pd.DataFrame
            Data filtered for the specified years
        """
        if self.data.empty:
            logger.warning("No data loaded. Call load_data() first.")
            return pd.DataFrame()
        
        # Get dates from index and filter by year
        filtered_data = self.data.reset_index()
        filtered_data['year'] = filtered_data['date'].dt.year
        filtered_data = filtered_data[filtered_data['year'].isin(years)]
        
        # Remove the temporary year column and reset the index
        filtered_data = filtered_data.drop(columns=['year'])
        filtered_data = filtered_data.set_index(['ticker', 'date']).sort_index()
        
        logger.info(f"Filtered data for years {years}: {len(filtered_data)} records")
        return filtered_data
    
    def create_cv_folds(self, 
                       start_year: int, 
                       num_folds: int = 5,
                       train_window_years: int = 3,
                       test_window_years: int = 1) -> List[Dict]:
        """
        Create time-based cross-validation folds.
        
        Parameters:
        -----------
        start_year : int
            First year to use in the first training fold
        num_folds : int, optional
            Number of folds to create
        train_window_years : int, optional
            Number of years in each training window
        test_window_years : int, optional
            Number of years in each testing window
            
        Returns:
        --------
        List[Dict]
            List of CV fold information dictionaries
        """
        min_history_days = self.config.get('min_history_days', 240)
        
        return create_time_based_cv_folds(
            self.data,
            start_year,
            num_folds,
            train_window_years,
            test_window_years,
            min_history_days
        )
    
    def get_train_test_split(self, 
                           train_years: List[int], 
                           test_years: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Get train and test split for specified years.
        
        Parameters:
        -----------
        train_years : List[int]
            List of years to include in training set
        test_years : List[int]
            List of years to include in testing set
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]
            Training data, testing data, and list of eligible tickers
        """
        min_history_days = self.config.get('min_history_days', 240)
        
        return split_train_test_data(
            self.data,
            train_years,
            test_years,
            min_history_days
        )
    
    def save_processed_data(self, output_path: str) -> bool:
        """
        Save processed data to file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the processed data
            
        Returns:
        --------
        bool
            True if save succeeded, False otherwise
        """
        if self.data.empty:
            logger.warning("No data to save. Call load_data() first.")
            return False
        
        try:
            # Save the data
            self.data.to_hdf(output_path, key='processed_data', mode='w')
            
            # Save ticker metadata
            if not self.ticker_metadata.empty:
                self.ticker_metadata.to_hdf(output_path, key='ticker_metadata', mode='a')
            
            # Save general metadata as a DataFrame for easy storage
            metadata_df = pd.DataFrame([self.metadata])
            metadata_df.to_hdf(output_path, key='metadata', mode='a')
            
            logger.info(f"Successfully saved processed data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {str(e)}")
            return False
    
    def load_processed_data(self, input_path: str) -> bool:
        """
        Load previously processed data from file.
        
        Parameters:
        -----------
        input_path : str
            Path to the processed data file
            
        Returns:
        --------
        bool
            True if load succeeded, False otherwise
        """
        try:
            # Load the data
            with pd.HDFStore(input_path, mode='r') as store:
                keys = store.keys()
                
                if '/processed_data' in keys:
                    self.data = store.get('processed_data')
                    logger.info(f"Loaded {len(self.data)} processed data records")
                else:
                    logger.error("No processed data found in the file")
                    return False
                
                if '/ticker_metadata' in keys:
                    self.ticker_metadata = store.get('ticker_metadata')
                    logger.info(f"Loaded metadata for {len(self.ticker_metadata)} tickers")
                
                if '/metadata' in keys:
                    metadata_df = store.get('metadata')
                    self.metadata = metadata_df.iloc[0].to_dict() if not metadata_df.empty else {}
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading processed data from {input_path}: {str(e)}")
            return False
