"""
Test module for feature engineering functionality.
"""

import pytest
import pandas as pd
import numpy as np

from src.features.feature_engineer import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create date range with more history for lookback periods
    dates = pd.date_range(start='2019-01-01', end='2020-04-10', freq='D')
    
    # Create tickers
    tickers = ['AAPL', 'MSFT']
    
    # Create multi-index
    index = pd.MultiIndex.from_product([tickers, dates], names=['ticker', 'date'])
    
    # Create random data with trend to make features more realistic
    np.random.seed(42)
    n_rows = len(index)
    trend = np.linspace(80, 120, n_rows).reshape(-1, 1)
    noise = np.random.normal(0, 5, size=(n_rows, 5))
    data = pd.DataFrame(
        trend + noise,
        index=index,
        columns=['open', 'high', 'low', 'close', 'volume']
    )
    
    # Ensure high/low prices make sense
    data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 1, n_rows))
    data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 1, n_rows))
    
    # Ensure volume is positive and larger
    data['volume'] = np.abs(noise[:, -1]) * 500000 + 1000000
    
    return data

@pytest.fixture
def config():
    """Create sample configuration for testing."""
    return {
        'features': {
            'price_col': 'close',
            'technical_indicators': {
                'rsi': {
                    'enabled': True,
                    'timeperiod': 14
                },
                'macd': {
                    'enabled': True,
                    'fastperiod': 12,
                    'slowperiod': 26,
                    'signalperiod': 9
                },
                'bollinger_bands': {
                    'enabled': True,
                    'timeperiod': 20,
                    'nbdevup': 2,
                    'nbdevdn': 2
                },
                'momentum': {
                    'enabled': True,
                    'timeperiods': [5, 10, 20, 60],
                    'types': {
                        'momentum': True,
                        'roc': True
                    }
                }
            },
            'apply_scaling': True
        },
        'target': {
            'type': 'multiclass',
            'calculation': {
                'method': 'std_based',
                'return_type': 'excess',
                'horizon': [1, 5, 10],
                'rolling_window': 20,
                'std_threshold': 1.0
            }
        }
    }

def test_feature_target_dataset_creation(sample_data, config):
    """Test creation of complete feature and target dataset."""
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
        'macd_12_26_9', 'macd_signal_12_26_9', 'macd_hist_12_26_9',
        'bb_upper_20', 'bb_middle_20', 'bb_lower_20', 'bb_bandwidth_20', 'bb_percent_b_20'
    ]
    expected_features.extend([f'mom_{period}' for period in [5, 10, 20, 60]])
    expected_features.extend([f'roc_{period}' for period in [5, 10, 20, 60]])
    
    for feature in expected_features:
        assert feature in features.columns, f"Expected feature {feature} not found"
    
    # Check targets
    assert not targets.empty, "Targets DataFrame should not be empty"
    
    # Check expected target columns
    expected_targets = [f'target_{h}d' for h in [1, 5, 10]]
    for target in expected_targets:
        assert target in targets.columns, f"Expected target {target} not found"
    
    # Check metadata
    assert metadata['num_features'] == len(features.columns), "Incorrect feature count in metadata"
    assert metadata['num_targets'] == len(targets.columns), "Incorrect target count in metadata"
    assert metadata['num_samples'] > 0, "No samples reported in metadata"
    assert metadata['num_tickers'] == len(sample_data.index.get_level_values('ticker').unique()), "Incorrect ticker count"
    
    # Check data quality metrics
    assert 'missing_values' in metadata['data_quality'], "Missing values not reported"
    assert 'unique_classes' in metadata['data_quality'], "Class distribution not reported"
    
    # Check index alignment
    assert (features.index == targets.index).all(), "Features and targets index mismatch"
    
    # Check scaling
    if config['features']['apply_scaling']:
        # Check if features are scaled (mean close to 0, std close to 1)
        feature_means = features.mean()
        feature_stds = features.std()
        assert (abs(feature_means) < 0.1).all(), "Features not properly scaled (mean)"
        assert ((feature_stds > 0.9) & (feature_stds < 1.1)).all(), "Features not properly scaled (std)"
