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
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
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
    
    logger.info("SECTION 1: Loading configuration...")
    config = load_config(config_file)
    
    pipeline_config = config.get('pipeline', {})
    data_file = pipeline_config.get('data_file')
    output_dir = pipeline_config.get('output_dir')
    feature_target_file = pipeline_config.get('feature_target_file')
    model_type = pipeline_config.get('model_type')
    cv_folds = int(pipeline_config.get('cv_folds', 5))
    train_start_year = pipeline_config.get('train_start_year')
    test_years = pipeline_config.get('test_years')
    save_model = pipeline_config.get('save_model', True)
    feature_selection = pipeline_config.get('feature_selection', False)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(feature_target_file), exist_ok=True)
    
    logger.info("SECTION 2: Data Preparation...")
    data_config = config.get('data', {})
    data_handler = DataHandler(data_config)
    
    logger.info(f"Loading raw data from {data_file}...")
    success = data_handler.load_data(data_file)
    
    if not success:
        logger.error("Failed to load data. Exiting.")
        return
    
    logger.info("SECTION 3: Feature Engineering...")
    feature_engineer = FeatureEngineer(config)
    
    logger.info("Creating features and targets for entire dataset...")
    features_targets = feature_engineer.create_feature_target_dataset(data_handler.data)
    
    if 'features' not in features_targets or features_targets['features'].empty:
        logger.error("Failed to create features")
        return
    
    features = features_targets['features']
    targets = features_targets['targets']
    
    # Get years from the data
    all_years = pd.DatetimeIndex(features.index.get_level_values('date')).year.unique()
    all_years = sorted(all_years)
    
    if train_start_year is None:
        train_start_year = all_years[0]
    
    if test_years is None:
        test_years = all_years[-5:]
    
    train_years = [y for y in all_years if y not in test_years and y >= train_start_year]
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
    
    # Feature selection if requested
    if feature_selection:
        logger.info("Performing feature selection...")
        from src.models.model_trainer import perform_feature_selection
        
        feature_selection_result = perform_feature_selection(
            X_train, 
            y_train,
            method='model_based',
            model_type='random_forest',
            n_top_features=40
        )
        
        X_train = feature_selection_result['X_reduced']
        selected_features = feature_selection_result['top_features']['feature'].tolist()
        X_test = X_test[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features")
    else:
        selected_features = X_train.columns.tolist()
    
    logger.info("SECTION 4: Model Training...")
    model_config = config.get('model', {})
    model_trainer = ModelTrainer(model_config)
    
    # Select target horizon for training (use first horizon by default)
    target_horizon = config.get('target', {}).get('calculation', {}).get('horizon', [1])[0]
    target_col = f'target_{target_horizon}d'
    
    if target_col not in y_train.columns:
        logger.error(f"Target column {target_col} not found in targets")
        return
    
    # Extract single target column
    y_train_single = y_train[target_col]
    y_test_single = y_test[target_col]
    
    logger.info(f"Training model for target horizon: {target_horizon} days")
    
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
    
    logger.info("SECTION 5: Model Evaluation...")
    eval_config = config.get('evaluation', {})
    eval_config['output_dir'] = output_dir
    model_evaluator = ModelEvaluator(eval_config)
    
    # Make predictions on test data
    logger.info("Making predictions on test data...")
    predictions = model_trainer.predict(model, X_test)
    
    # Run evaluation
    logger.info("Running model evaluation...")
    eval_results = model_evaluator.evaluate_model(
        model,
        X_test,
        y_test_single,
        predictions
    )
    
    # Save evaluation results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_file}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Pipeline completed in {duration.total_seconds() / 60:.2f} minutes")

if __name__ == "__main__":
    main()
