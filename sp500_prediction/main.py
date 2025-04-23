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
from src.models.training import ModelTrainer, perform_feature_selection
from src.models.evaluation import YellowbrickEvaluator
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
    
    if not success:
        logger.error("Failed to load data. Exiting.")
        return
    
    ###############################
    # 3. Feature Engineering
    ###############################
    logger.info("Performing feature engineering...")
    feature_engineer = FeatureEngineer(config)
    
    logger.info("Creating features and targets for entire dataset...")
    features_targets = feature_engineer.create_feature_target_dataset(data_handler.data)
    
    if 'features' not in features_targets or features_targets['features'].empty:
        logger.error("Failed to create features")
        return
    
    features = features_targets['features']
    targets = features_targets['targets']
    
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
    if feature_selection:
        logger.info("Performing feature selection...")
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
    
    ###############################
    # 6. Model Training
    ###############################
    logger.info("Training model...")
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
    
    ###############################
    # 7. Model Evaluation
    ###############################
    logger.info("Evaluating model performance...")
    # Initialize model evaluator with output directory
    eval_config = config.get('evaluation', {})
    eval_config['output_dir'] = output_dir
    eval_config['model_type'] = model_type
    model_evaluator = YellowbrickEvaluator(eval_config)
    
    # Create predictions DataFrame
    logger.info("Making predictions on test data...")
    y_pred = model_trainer.predict(model=model, X=X_test)
    
    predictions = pd.DataFrame({
        'prediction': y_pred,
        'true': y_test_single,
        'probability': model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    }, index=X_test.index)
    
    # Run evaluation
    logger.info("Running model evaluation...")
    eval_results = model_evaluator.evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test_single
    )

    # Generate performance report
    model_evaluator.generate_report()
    
    # Create learning curve
    model_evaluator.create_learning_curve(
        model=model,
        X_train=X_train,
        y_train=y_train_single,
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
