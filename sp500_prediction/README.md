# S&P500 Stock Direction Prediction

A machine learning system to predict stock movements using technical indicators and historical data, with configurable target variables and efficient memory handling.

## Project Overview

This project implements a pipeline to predict S&P500 stock movements using technical indicators and various target definitions. The system supports multiple prediction types (returns, binary, multi-class) with configurable calculation methods and horizons.

## Feature Engineering

### Technical Indicators Configuration

The system supports multiple technical indicators with flexible period generation, configurable via `config.json`:

```json
"features": {
  "price_col": "close",
  "technical_indicators": {
    "rsi": {
      "enabled": true,
      "timeperiod_generation": {
        "method": "range",
        "params": {
          "range": {"start": 9, "end": 30, "step": 7}  // Generates [9, 16, 23]
        }
      }
    },
    "macd": {
      "enabled": true,
      "period_generation": {
        "fast": {"start": 8, "end": 15, "step": 4},    // Generates [8, 12]
        "slow": {"start": 20, "end": 30, "step": 6},   // Generates [20, 26]
        "signal": {"start": 7, "end": 10, "step": 2}   // Generates [7, 9]
      }
    },
    "bollinger_bands": {
      "enabled": true,
      "timeperiod_generation": {
        "method": "range",
        "params": {
          "range": {"start": 10, "end": 31, "step": 10}  // Generates [10, 20, 30]
        }
      },
      "nbdevup": 2,
      "nbdevdn": 2
    },
    "momentum": {
      "enabled": true,
      "timeperiod_generation": {
        "method": "range",
        "params": {
          "short_range": {"start": 1, "end": 21, "step": 1},
          "long_range": {"start": 40, "end": 241, "step": 20}
        }
      },
      "types": {
        "momentum": true,  // Absolute price difference
        "roc": true       // Rate of Change (percentage)
      }
    }
  },
  "apply_scaling": true
}
```

### Target Configuration

The system supports multiple target types and calculation methods:

```json
"target": {
  "type": "returns|binary|multiclass",  // Target variable type
  "calculation": {
    "method": "std_based|threshold_based|quantile_based|raw",  // Calculation method
    "return_type": "raw|excess|log|percentage",                // Return calculation type
    "horizon": [1, 5, 10],                // Prediction horizons in days
    "rolling_window": 20,                 // Window for statistics
    "std_threshold": 1.0,                 // For std-based method
    "fixed_threshold": 0.01,              // For threshold-based method
    "quantiles": [0.33, 0.67]            // For quantile-based method
  }
}
```

#### Target Types

1. Return-Based Targets
   - Raw returns: Simple price changes
   - Excess returns: Returns relative to market average
   - Log returns: Natural logarithm of returns
   - Percentage returns: Percentage price changes

2. Binary Classification (0/1)
   - Threshold-based: Fixed value threshold
   - Standard deviation based: Dynamic threshold
   - Median-based: Above/below median
   - Zero-based: Positive/negative returns

3. Multi-class Classification (0/1/2)
   - Standard deviation based classes
   - Quantile-based boundaries
   - Fixed range thresholds

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sp500_prediction.git
cd sp500_prediction
```

2. Create environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --data_file data/raw/sp500_data.h5 --output_dir results
```

### Advanced Options

```bash
python main.py --data_file data/raw/sp500_data.h5 \
               --output_dir results \
               --model_type xgboost \
               --cv_folds 5 \
               --train_start_year 2010 \
               --test_years 2020 2021 2022 \
               --save_model \
               --feature_selection \
               --config_file configs/config.json
```

## Data Requirements

- HDF5 format with multi-index (ticker, date)
- Required columns:
  * open, high, low, close: Price data
  * volume: Trading volume

## Models

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Neural Network (MLP)

## Evaluation

- Classification metrics
- Time-based analysis
- Stock-specific analysis
- Feature importance
- Model explainability

## Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Model training
- talib: Technical indicators
- matplotlib/seaborn: Visualization

## License

[MIT License](LICENSE)

## Contributing

Contributions welcome! Please submit a Pull Request.

## Model Explainability

The project includes comprehensive model explanation tools:

```python
from src.explanation.model_explainer import ModelExplainer

# Initialize explainer
explainer = ModelExplainer(config={'output_dir': 'results/explanation'})

# Generate explanations
explanation = explainer.explain_model(
    model=trained_model,
    X=X_test,
    y=y_test,
    n_top_features=20
)

# Stock-specific analysis
stock_explanation = explainer.explain_stock_prediction(
    model=trained_model,
    X=X_test,
    ticker='AAPL',
    date=pd.Timestamp('2022-01-15')
)
