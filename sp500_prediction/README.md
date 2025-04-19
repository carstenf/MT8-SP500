# S&P500 Stock Direction Prediction

A machine learning system to predict daily price movements (up/down) for S&P500 stocks using historical price data.

## Project Overview

This project implements a pipeline to predict whether S&P500 stocks will have positive or negative excess returns (relative to the market average) the next day. The system uses historical price data and various derivative features such as lagged returns and technical indicators to train multiple classification models.

### Key Features

- Multi-stage data processing pipeline
- Feature engineering with diverse time horizons
- Cross-validated model training
- Comprehensive model evaluation
- Time-based and ticker-based performance analysis
- Model explainability with feature importance analysis

## Project Structure

```
sp500_prediction/
├── data/                  # Data storage
│   ├── raw/               # Original HDF5 files
│   └── processed/         # Processed datasets
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data/              # Data handling modules
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementation
│   ├── visualization/     # Visualization utilities
│   └── explanation/       # Model explanation
├── tests/                 # Test cases
├── configs/               # Configuration files
├── results/               # Output files
│   ├── models/            # Saved model files
│   ├── plots/             # Generated visualizations
│   └── metrics/           # Performance metrics
├── requirements.txt       # Dependencies
├── main.py                # Main execution script
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sp500_prediction.git
cd sp500_prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script with the required data file:

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

### Configuration

The project can be configured using a JSON configuration file. See `configs/config.json` for an example.

## Data Format

The system expects S&P500 stock data in HDF5 format with a multi-index structure (ticker, date). Required columns include:
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

## Feature Engineering

The system generates several types of features:
- Daily lagged excess returns (1-40 days)
- Long-term lagged excess returns (40-240 days)
- Cumulative returns over various windows
- Average returns over various windows
- Volatility over various windows

## Models

The system supports multiple model types:
- Logistic Regression (with L1, L2, or no regularization)
- Random Forest
- XGBoost
- LightGBM
- Neural Network (MLP)

## Evaluation

The evaluation includes:
- Standard classification metrics (accuracy, balanced accuracy, precision, recall, F1 score)
- ROC-AUC and PR-AUC
- Time-based performance analysis
- Ticker-based performance analysis
- Feature importance analysis
- Bias-variance tradeoff analysis

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- lightgbm
- h5py
- tables (PyTables)

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
