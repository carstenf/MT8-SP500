# S&P500 Stock Direction Prediction

A machine learning system to predict stock movements using technical indicators and historical data, with configurable target variables and efficient memory handling.

## Project Overview

This project implements a pipeline to predict S&P500 stock movements using technical indicators and various target definitions. The system supports multiple prediction types (returns, binary, multi-class) with configurable calculation methods and horizons.

### Key Features

- Technical indicators (RSI, MACD, Bollinger Bands)
- Configurable target variables
- Memory-efficient data processing
- Vectorized calculations
- Comprehensive model evaluation
- Model explainability

## Project Structure

```
sp500_prediction/
├── data/                  # Data storage
│   ├── raw/               # Original HDF5 files
│   └── processed/         # Processed datasets
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── data/              # Data handling
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementation
│   ├── visualization/     # Visualization
│   └── explanation/       # Model explanation
├── tests/                 # Test cases
├── configs/               # Configuration
├── results/               # Output files
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Configuration Guide

### Pipeline Configuration
```json
"pipeline": {
  "data_file": "data/raw/sp500_data.h5",  // Input data location
  "output_dir": "results",                 // Output directory
  "model_type": "random_forest|xgboost|lightgbm|neural_network|logistic_regression",
  "cv_folds": 5,                          // Number of cross-validation folds
  "save_model": true,                     // Whether to save trained model
  "feature_selection": false,             // Enable feature selection
  "split": {
    "train_start": "2010-01-01",         // Training period start
    "test_start": "2018-01-01",          // Test period start
    "test_end": "2019-12-31"             // Test period end
  }
}
```

### Feature Configuration
```json
"features": {
  "price_col": "close|open|high|low",     // Price column for calculations
  "technical_indicators": {
    "rsi": {
      "enabled": true|false,              // Enable/disable indicator
      "timeperiod": 14                    // RSI calculation period
    },
    "macd": {
      "enabled": true|false,
      "fastperiod": 12,                   // Fast EMA period
      "slowperiod": 26,                   // Slow EMA period
      "signalperiod": 9                   // Signal line period
    },
    "bollinger_bands": {
      "enabled": true|false,
      "timeperiod": 20,                   // Moving average period
      "nbdevup": 2,                       // Upper band deviation
      "nbdevdn": 2                        // Lower band deviation
    },
    "momentum": {
      "enabled": true|false,
      "timeperiods": [5, 10, 20, 60],     // List of calculation periods
      "types": {
        "momentum": true|false,            // Absolute price difference
        "roc": true|false                  // Rate of Change (percentage)
      }
    }
  },
  "apply_scaling": true|false              // Standardize features
}
```

### Target Configuration
```json
"target": {
  "type": "returns|binary|multiclass",     // Type of prediction target
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

### Model Configuration
```json
"model": {
  "random_forest": {
    "n_estimators": 100,                  // Number of trees
    "max_depth": 15,                      // Maximum tree depth
    "min_samples_split": 10,              // Minimum samples for split
    "min_samples_leaf": 5,                // Minimum samples in leaf
    "max_features": "sqrt|log2|auto",     // Feature selection method
    "criterion": "gini|entropy"           // Split criterion
  },
  "xgboost": {
    "n_estimators": 100,                  // Number of boosting rounds
    "max_depth": 6,                       // Maximum tree depth
    "learning_rate": 0.1,                 // Learning rate
    "subsample": 0.8,                     // Sample ratio for trees
    "colsample_bytree": 0.8              // Column ratio for trees
  },
  "lightgbm": {
    "num_leaves": 31,                     // Number of leaves
    "learning_rate": 0.1,                 // Learning rate
    "feature_fraction": 0.8,              // Feature sampling ratio
    "bagging_fraction": 0.8,              // Row sampling ratio
    "bagging_freq": 5                     // Bagging frequency
  },
  "neural_network": {
    "hidden_layer_sizes": [100, 50, 25],  // Network architecture
    "activation": "relu|tanh|sigmoid",     // Activation function
    "solver": "adam|sgd|lbfgs",           // Optimization algorithm
    "learning_rate": "constant|adaptive",  // Learning rate schedule
    "early_stopping": true|false          // Enable early stopping
  }
}
```

### Evaluation Configuration
```json
"evaluation": {
  "metrics": {
    "balanced_accuracy_threshold": 0.55,   // Minimum acceptable accuracy
    "zero_division": 0,                   // Handling zero division
    "slope_threshold": 0.01               // Trend detection threshold
  },
  "visualization": {
    "plot_figsize": {
      "default": [10, 6],                // Default plot size
      "feature_importance": [12, 8],      // Feature importance plot size
      "confusion_matrix": [8, 6]          // Confusion matrix plot size
    }
  }
}
```

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
