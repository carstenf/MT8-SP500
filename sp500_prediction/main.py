"""
Main script for S&P500 Stock Direction Prediction Project.

This script orchestrates the full pipeline from data loading to model evaluation.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import project modules
from src.data.data_handler import DataHandler
from src.features.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator

# Set up logging
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
    # Load configuration
    config = load_config(config_file)
    
    # Get pipeline configuration
    pipeline_config = config.get('pipeline', {})
    data_file = pipeline_config.get('data_file')
    output_dir = pipeline_config.get('output_dir')
    model_type = pipeline_config.get('model_type')
    cv_folds = pipeline_config.get('cv_folds', 5)
    train_start_year = pipeline_config.get('train_start_year')
    test_years = pipeline_config.get('test_years')
    save_model = pipeline_config.get('save_model', True)
    feature_selection = pipeline_config.get('feature_selection', False)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure data handler
    data_config = config.get('data', {})
    data_handler = DataHandler(data_config)
    
    # Load and prepare data
    logger.info(f"Loading data from {data_file}...")
    success = data_handler.load_data(data_file)
    
    if not success:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Get data overview
    ticker_metadata = data_handler.ticker_metadata
    n_tickers = len(ticker_metadata)
    date_range = (data_handler.metadata.get('start_date'), data_handler.metadata.get('end_date'))
    
    logger.info(f"Loaded data for {n_tickers} tickers from {date_range[0]} to {date_range[1]}")
    
    # Determine train and test years if not provided
    all_years = pd.DatetimeIndex(data_handler.data.index.get_level_values('date')).year.unique()
    all_years = sorted(all_years)
    
    if train_start_year is None:
        train_start_year = all_years[0]
    
    if test_years is None:
        test_years = all_years[-5:]  # Use last 5 years for testing
    
    # Define train years (all years except test years)
    train_years = [y for y in all_years if y not in test_years and y >= train_start_year]
    
    logger.info(f"Training years: {train_years}")
    logger.info(f"Testing years: {test_years}")
    
    # Split data into training and testing sets
    train_data, test_data, eligible_tickers = data_handler.get_train_test_split(train_years, test_years)
    
    logger.info(f"Training data: {len(train_data)} records for {len(eligible_tickers)} tickers")
    logger.info(f"Testing data: {len(test_data)} records")
    
    # Configure feature engineer
    feature_config = config.get('features', {})
    feature_engineer = FeatureEngineer(feature_config)
    
    # Create features for training data
    logger.info("Creating features for training data...")
    train_features_result = feature_engineer.create_features(train_data)
    
    X_train = train_features_result['features']
    y_train = train_features_result['targets']
    class_distribution = train_features_result['class_distribution']
    
    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Class distribution: {class_distribution['class_proportions']}")
    
    # Apply feature selection if requested
    if feature_selection:
        logger.info("Performing feature selection...")
        
        # Import from model_trainer to use feature selection function
        from src.models.model_trainer import perform_feature_selection
        
        feature_selection_result = perform_feature_selection(
            X_train, 
            y_train,
            method='model_based',
            model_type='random_forest',
            n_top_features=40
        )
        
        # Use selected features
        X_train = feature_selection_result['X_reduced']
        selected_features = feature_selection_result['top_features']['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
    else:
        selected_features = X_train.columns.tolist()
    
    # Create features for testing data
    logger.info("Creating features for testing data...")
    test_features_result = feature_engineer.create_features(test_data)
    
    X_test = test_features_result['features']
    y_test = test_features_result['targets']
    
    # Ensure test features have the same columns as train features
    X_test = X_test[selected_features]
    
    logger.info(f"Testing features shape: {X_test.shape}")
    
    # Configure model trainer
    model_config = config.get('model', {})
    model_trainer = ModelTrainer(model_config)
    
    # Get class weights from class distribution
    class_weights = class_distribution['class_weights'] if class_distribution['is_severely_imbalanced'] else None
    
    if cv_folds > 0:
        # Perform cross-validation
        logger.info(f"Creating {cv_folds} cross-validation folds...")
        
        cv_folds = data_handler.create_cv_folds(
            train_start_year,
            cv_folds,
            train_window_years=3,
            test_window_years=1
        )
        
        # Process each fold for feature engineering
        for i, fold in enumerate(cv_folds):
            fold_num = fold['fold_num']
            logger.info(f"Processing fold {fold_num}...")
            
            # Create features for this fold
            train_fold_features = feature_engineer.create_features(fold['train_data'])
            test_fold_features = feature_engineer.create_features(fold['test_data'])
            
            # Check if we have valid features and targets
            if ('features' not in train_fold_features or 'targets' not in train_fold_features or
                'features' not in test_fold_features or 'targets' not in test_fold_features):
                logger.error(f"Missing features or targets in fold {fold_num}")
                continue
            
            # Store the features and targets
            try:
                fold['train_features'] = train_fold_features['features'][selected_features]
                fold['train_targets'] = train_fold_features['targets']
                fold['test_features'] = test_fold_features['features'][selected_features]
                fold['test_targets'] = test_fold_features['targets']
                logger.info(f"Successfully processed features for fold {fold_num}")
            except Exception as e:
                logger.error(f"Error processing features for fold {fold_num}: {str(e)}")
                continue
        
        # Perform cross-validation
        logger.info(f"Performing cross-validation with {model_type} model...")
        
        cv_results = model_trainer.perform_cross_validation(
            cv_folds,
            model_type=model_type,
            class_weights=class_weights
        )
        
        logger.info(f"Cross-validation results: {cv_results['avg_metrics']}")
        
        # Save CV results
        cv_results_file = os.path.join(output_dir, 'cv_results.json')
        with open(cv_results_file, 'w') as f:
            def convert_for_json(obj):
                if isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.number):
                    return obj.item()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                else:
                    return obj

            # Convert cross-validation results to JSON-serializable format
            cv_results_json = {}
            for key, value in cv_results.items():
                if key == 'fold_results':
                    # Skip fold_results as they contain model objects that can't be serialized
                    continue
                cv_results_json[key] = convert_for_json(value)
            
            json.dump(cv_results_json, f, indent=2)
    
    # Train the final model
    logger.info(f"Training final {model_type} model...")
    
    model_result = model_trainer.train_model(
        X_train, 
        y_train,
        model_type=model_type,
        class_weights=class_weights
    )
    
    # Get the trained model
    if model_type == 'logistic_regression':
        model = model_result['best_model']
    else:
        model = model_result['model']
    
    # Save the model if requested
    if save_model:
        model_file = os.path.join(output_dir, f"{model_type}_model.pkl")
        model_trainer.save_model(model, model_file)
        logger.info(f"Model saved to {model_file}")
    
    # Configure model evaluator
    eval_config = {'output_dir': output_dir}
    model_evaluator = ModelEvaluator(eval_config)
    
    # Make predictions on test data
    logger.info("Making predictions on test data...")
    predictions = model_trainer.predict(X_test, model)
    
    # Combine predictions with true values
    prediction_df = pd.DataFrame(index=X_test.index)
    prediction_df['prediction'] = predictions['prediction']
    prediction_df['probability'] = predictions['probability']
    prediction_df['true'] = y_test
    
    # Save predictions
    predictions_file = os.path.join(output_dir, 'predictions.csv')
    prediction_df.to_csv(predictions_file)
    logger.info(f"Predictions saved to {predictions_file}")
    
    # Get excess returns for evaluation
    excess_returns = test_features_result['excess_returns']
    
    # Run full evaluation
    logger.info("Running full model evaluation...")
    eval_results = model_evaluator.run_full_evaluation(
        model, 
        X_train, 
        y_train,
        X_test, 
        y_test,
        prediction_df,
        excess_returns
    )
    
    logger.info("Evaluation complete.")
    logger.info(f"Performance report saved to {os.path.join(output_dir, 'performance_report.json')}")
    
    # Final summary
    if 'overall_metrics' in eval_results:
        metrics = eval_results['overall_metrics']
        logger.info(f"Final model performance:")
        logger.info(f"  Balanced accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
        logger.info(f"  F1 score: {metrics.get('f1', 0):.4f}")
        logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Starting S&P500 prediction pipeline at {start_time}")
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main execution: {str(e)}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Pipeline completed in {duration.total_seconds() / 60:.2f} minutes")
