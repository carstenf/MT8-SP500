"""
Main script for S&P500 Stock Direction Prediction Project.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
from src.data.data_handler import DataHandler
from src.features.feature_engineer import FeatureEngineer
from src.features.target_engineer import TargetEngineer
from src.models.training import (
    ModelTrainer, 
    perform_feature_selection
)
from src.models.training.data_filtering import filter_ticker_out_with_nan
from src.evaluation import YellowbrickEvaluator
from src.explanation.model_explainer import ModelExplainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sp500_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_file):
    """Load configuration from a JSON file."""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}

def main(config_file: str = 'configs/config.json'):
    """Main execution function."""
    start_time = datetime.now()
    
    ###############################
    # 1. Load Configuration
    ###############################
    logger.info("Loading configuration...")
    config = load_config(config_file)
    
    pipeline_config = config.get('pipeline', {})
    data_file = pipeline_config.get('data_file')
    output_dir = pipeline_config.get('output_dir')
    feature_target_file = pipeline_config.get('feature_target_file')
    model_type = pipeline_config.get('model_type')
    cv_folds = int(pipeline_config.get('cv_folds', 5))
    
    # Extract years from split configuration
    split_config = pipeline_config.get('split', {})
    train_start = split_config.get('train_start')
    test_start = split_config.get('test_start')
    test_end = split_config.get('test_end')
    
    # Parse years from date strings
    train_start_year = int(train_start.split('-')[0]) if train_start else None
    test_start_year = int(test_start.split('-')[0]) if test_start else None
    test_end_year = int(test_end.split('-')[0]) if test_end else None
    
    # Create list of test years
    test_years = list(range(test_start_year, test_end_year + 1)) if test_start_year and test_end_year else None
    save_model = pipeline_config.get('save_model', True)
    feature_selection = pipeline_config.get('feature_selection', False)
    
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(feature_target_file), exist_ok=True)
    
    ###############################
    # 2. Data Preparation
    ###############################
    logger.info("Preparing data...")
    data_config = config.get('data', {})
    data_handler = DataHandler(data_config)
    
    logger.info(f"Loading raw data from {data_file}...")
    success = data_handler.load_data(data_file)

    #if success:
    #    unique_dates = data_handler.data.index.get_level_values('date').unique()
    #    print("\nFirst 10 dates:")
    #    print(unique_dates[:10])
    
    if not success:
        logger.error("Failed to load data. Exiting.")
        return
    
    ###############################
    # 3. Feature Engineering
    ###############################
    logger.info("Performing feature engineering...")
    # Create features
    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.create_features(data_handler.data)
    if features.empty:
        logger.error("Failed to create features")
        return

    # Create targets
    target_engineer = TargetEngineer(config)
    targets = target_engineer.create_targets(data_handler.data)
    if targets.empty:
        logger.error("Failed to create targets")
        return
    
    # Create metadata
    metadata = {
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
            'unique_classes': {col: sorted(targets[col].unique().tolist()) for col in targets.columns}
        }
    }

    # Log dataset information from metadata
    if metadata:
        logger.info(f"Dataset creation complete:")
        logger.info(f"Features shape: {metadata['num_features']} features, {metadata['num_samples']} samples")
        logger.info(f"Date range: {metadata['data_start_date']} to {metadata['data_end_date']}")
        logger.info(f"Number of tickers: {metadata['num_tickers']}")
    
    # Get AAPL data
    #aapl_data = features.loc['AAPL']
    #print("\nFirst 10 days of AAPL:")
    #print(aapl_data.head(10).to_string())
    #print("\n" + "="*80 + "\n")  # Separator
    #print("Last 10 days of AAPL:")
    #print(aapl_data.tail(10).to_string())
    #print(f"\nTotal trading days for AAPL: {len(aapl_data)}")
        
    ###############################
    # 4. Data Split
    ###############################
    logger.info("Splitting data into train and test sets...")
    
    # Get years from the data
    all_years = pd.DatetimeIndex(features.index.get_level_values('date')).year.unique()
    all_years = sorted(all_years)
    
    if train_start_year is None:
        train_start_year = all_years[0]
    
    if test_years is None:
        test_years = all_years[-5:]
    
    train_years = [y for y in all_years if train_start_year <= y < test_start_year]
    logger.info(f"Training years: {train_years}")
    logger.info(f"Testing years: {test_years}")
    
    # Split features and targets by years
    features_idx = features.index.get_level_values('date')
    targets_idx = targets.index.get_level_values('date')
    
    X_train = features[features_idx.year.isin(train_years)]
    X_test = features[features_idx.year.isin(test_years)]
    y_train = targets[targets_idx.year.isin(train_years)]
    y_test = targets[targets_idx.year.isin(test_years)]
    
    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Testing features shape: {X_test.shape}")
    
    ###############################
    # 5. Feature Selection
    ###############################
    
    # Select target horizon for training (use first horizon by default)
    target_horizon = config.get('target', {}).get('calculation', {}).get('horizon', [1])[0]
    target_col = f'target_{target_horizon}d'
    
    if target_col not in targets.columns:
        logger.error(f"Target column {target_col} not found in targets")
        return
    
    logger.info(f"Training model for target horizon: {target_horizon} days")
    
    if feature_selection:
        logger.info(f"Performing feature selection for {target_col}...")
        feature_selection_result = perform_feature_selection(
            X_train,
            y_train[target_col],
            method='model_based',
            model_type='random_forest',
            n_top_features=40
        )
        
        # Get selected features based on method used
        if 'top_features' in feature_selection_result:
            selected_features = feature_selection_result['top_features']  # model_based returns list directly
        elif 'selected_features' in feature_selection_result:
            selected_features = feature_selection_result['selected_features']  # recursive returns list directly
        else:
            selected_features = feature_selection_result['confirmed_features']  # boruta returns list directly
        
        X_train = feature_selection_result['X_reduced']
        X_test = X_test[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features")
    else:
        selected_features = X_train.columns.tolist()
    
    ###############################
    # 6. Model Training
    ###############################
    logger.info("Training model...")
    model_config = config.get('model', {})
    model_trainer = ModelTrainer(model_config)
    
    # Extract target column for training
    y_train_single = y_train[target_col]
    y_test_single = y_test[target_col]
    
    # Train final model
    logger.info(f"Training final {model_type} model...")
    model_result = model_trainer.train_model(
        X_train, 
        y_train_single,
        model_type=model_type
    )
    
    model = model_result['model']
    
    # Save model if requested
    if save_model:
        model_file = os.path.join(output_dir, f"{model_type}_model.pkl")
        model_trainer.save_model(model, model_file)
        logger.info(f"Model saved to {model_file}")
    
    ###############################
    # 7. Model Evaluation
    ###############################
    logger.info("Evaluating model performance...")
    # Initialize model evaluator with comprehensive configuration
    eval_config = {
        'output_dir': output_dir,
        'model_type': model_type,
        'model': config.get('model', {}),
        'evaluation': config.get('evaluation', {}),
        'train_years': train_years,
        'test_years': test_years,
        'feature_selection': {
            'enabled': feature_selection,
            'method': 'filter_ticker_out_with_nan',
            'original_features': X_train.shape[1],
            'selected_features': len(selected_features)
        }
    }
    model_evaluator = YellowbrickEvaluator(eval_config)
    
    # Create predictions DataFrame
    logger.info("Making predictions on test data...")
    y_pred = model_trainer.predict(model=model, X=X_test)
    
    # Filter test data independently
    X_test_filtered, y_test_filtered = filter_ticker_out_with_nan(X_test, y_test_single)
    
    # Create predictions DataFrame with probabilities
    probabilities = model.predict_proba(X_test_filtered)[:, 1] if hasattr(model, 'predict_proba') else None
    
    predictions = pd.DataFrame({
        'prediction': y_pred,
        'true': y_test_single,
        'probability': None
    }, index=X_test.index)
    
    # Update probabilities for valid samples
    if probabilities is not None:
        predictions.loc[X_test_filtered.index, 'probability'] = probabilities
    
    # Run evaluation on filtered data
    logger.info("Running model evaluation...")
    eval_results = model_evaluator.evaluate_model(
        model=model,
        X_test=X_test_filtered,
        y_test=y_test_filtered
    )

    # Generate performance report
    model_evaluator.generate_report()
    
    # Create learning curve using filtered training data
    # Filter training data independently
    X_train_filtered, y_train_filtered = filter_ticker_out_with_nan(X_train, y_train_single)
    
    model_evaluator.create_learning_curve(
        model=model,
        X_train=X_train_filtered,
        y_train=y_train_filtered,
        cv=3,
        max_samples=2500
    )
    
    logger.info("Model evaluation completed")
    
    ###############################
    # 8. Pipeline Summary
    ###############################
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Pipeline completed in {duration.total_seconds() / 60:.2f} minutes")

if __name__ == "__main__":
    main()
