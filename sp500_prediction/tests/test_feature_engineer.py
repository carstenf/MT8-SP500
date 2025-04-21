import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineer import FeatureEngineer

@pytest.fixture
def sample_data():
    # Create sample price data for testing
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'MSFT']
    
    # Create multi-index
    idx = pd.MultiIndex.from_product([tickers, dates], names=['ticker', 'date'])
    
    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame(
        {
            'open': np.random.randn(200).cumsum() + 100,
            'high': np.random.randn(200).cumsum() + 102,
            'low': np.random.randn(200).cumsum() + 98,
            'close': np.random.randn(200).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 200)
        },
        index=idx
    )
    return data

@pytest.fixture
def config():
    return {
        "features": {
            "price_col": "close",
            "technical_indicators": {
                "rsi": {
                    "enabled": True,
                    "timeperiod": 14
                },
                "macd": {
                    "enabled": True,
                    "fastperiod": 12,
                    "slowperiod": 26,
                    "signalperiod": 9
                },
                "bollinger_bands": {
                    "enabled": True,
                    "timeperiod": 20,
                    "nbdevup": 2,
                    "nbdevdn": 2
                },
                "momentum": {
                    "enabled": True,
                    "timeperiods": [5, 10, 20, 60]
                }
            },
            "apply_scaling": True
        },
        "target": {
            "type": "multiclass",
            "calculation": {
                "method": "std_based",
                "return_type": "excess",
                "horizon": [1, 5, 10],
                "rolling_window": 20,
                "std_threshold": 1.0
            }
        }
    }

def test_feature_target_dataset_creation(sample_data, config):
    # Initialize feature engineer
    fe = FeatureEngineer(config)
    
    # Create complete dataset
    result = fe.create_feature_target_dataset(sample_data)
    
    # Check that result contains all required keys
    assert 'features' in result, "Result missing features"
    assert 'targets' in result, "Result missing targets"
    assert 'metadata' in result, "Result missing metadata"
    
    features = result['features']
    targets = result['targets']
    metadata = result['metadata']
    
    # Check features
    assert not features.empty, "Features DataFrame should not be empty"
    
    # Check expected features exist
    expected_features = [
        'rsi_14',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth', 'bb_percent_b'
    ]
    expected_features.extend([f'mom_{period}' for period in [5, 10, 20, 60]])
    expected_features.extend([f'roc_{period}' for period in [5, 10, 20, 60]])
    
    for feature in expected_features:
        assert feature in features.columns, f"Expected feature {feature} not found"
    
    # Check targets
    assert not targets.empty, "Targets DataFrame should not be empty"
    
    # Check target columns
    expected_horizons = config['target']['calculation']['horizon']
    expected_columns = [f'target_{h}d' for h in expected_horizons]
    
    for col in expected_columns:
        assert col in targets.columns, f"Expected target column {col} not found"
        
        # Verify target values
        unique_values = targets[col].unique()
        valid_values = [0, 1, 2]
        invalid_values = [val for val in unique_values if val not in valid_values]
        if len(invalid_values) > 0:
            print(f"\nColumn: {col}")
            print(f"Invalid values found: {invalid_values}")
            print(f"All unique values: {sorted(unique_values)}")
            print(f"Value types: {[type(val) for val in unique_values]}")
        assert len(invalid_values) == 0, f"Invalid target values found in {col}"
        
        # Check class distribution
        value_counts = targets[col].value_counts(normalize=True)
        assert 0.5 < value_counts[1] < 0.85, f"Unexpected class distribution in {col}"
    
    # Check that feature scaling worked
    assert features.mean().abs().mean() < 1.0, "Features should be scaled with mean close to 0"
    assert features.std().mean() == pytest.approx(1.0, rel=0.5), "Features should be scaled with std close to 1"
    
    # Check that we have expected number of samples
    min_required = metadata['min_required_samples']
    for ticker in features.index.get_level_values('ticker').unique():
        ticker_features = features.loc[ticker]
        ticker_data = sample_data.loc[ticker]
        
        # Check that we have enough data for the ticker
        assert len(ticker_features) <= len(ticker_data), \
            f"More feature samples than input data for {ticker}"
        assert len(ticker_features) <= len(ticker_data) - min_required, \
            f"Not enough samples removed for lookback period for {ticker}"
        
        # Check that dates are within valid range
        first_feature_date = ticker_features.index[0]
        last_feature_date = ticker_features.index[-1]
        first_data_date = ticker_data.index[0]
        last_data_date = ticker_data.index[-1]
        
        assert first_feature_date >= first_data_date + pd.Timedelta(days=min_required), \
            f"Features start too early for {ticker}"
        assert last_feature_date <= last_data_date, \
            f"Features extend beyond data range for {ticker}"
