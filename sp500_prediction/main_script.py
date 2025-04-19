"""
Main script for S&P500 Stock Direction Prediction Project.

This script orchestrates the full pipeline from data loading to model evaluation.
"""

import os
import logging
import pandas as pd
import numpy as np
import argparse
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='S&P500 Stock Direction Prediction')
    
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to HDF5 file containing S&P500 data')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--model_type', type=str, default='xgboost',
                        choices=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'neural_network'],
                        help='Type of model to train')
    
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    
    parser.add_argument('--train_start_year', type=int, default=None,
                        help='Start year for training data (if None, will be inferred)')
    
    parser.add_argument('--test_years', type=int, nargs='+', default=None,
                        help='Years to use for testing (if None, will use the last 5 years of data)')
    
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model')
    
    parser.add_argument('--feature_selection', action='store_true',
                        help='Perform feature selection')
    
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to JSON configuration file')
    
    return parser.parse_args()


def load_config(config_file):
    """Load configuration from a JSON file."""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure data handler
    data_config = config.get('data', {})
    data_handler = DataHandler(data_config)
    
    # Load and prepare data
    logger.info(f"Loading data from {args.data_file}...")
    success = data_handler.load_data(args.data_file)
    
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
    
    if args.train_start_year is None:
        train_start_year = all_years[0]
    else:
        train_start_year = args.train_start_year
    
    if args.test_years is None:
        test_years = all_years[-5:]  # Use last 5 years for testing
    else:
        test_years = args.test_years
    
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
    if args.feature_selection:
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
    
    if args.cv_folds > 0:
        # Perform cross-validation
        logger.info(f"Creating {args.cv_folds} cross-validation folds...")
        
        cv_folds = data_handler.create_cv_folds(
            train_start_year,
            args.cv_folds,
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
            
            # Store the features and targets
            fold['train_features'] = train_fold_features['features'][selected_features]
            fold['train_targets'] = train_fold_features['targets']
            fold['test_features'] = test_fold_features['features'][selected_features]
            fold['test_targets'] = test_fold_features['targets']
        
        # Perform cross-validation
        logger.info(f"Performing cross-validation with {args.model_type} model...")
        
        cv_results = model_trainer.perform_cross_validation(
            cv_folds,
            model_type=args.model_type,
            class_weights=class_weights
        )
        
        logger.info(f"Cross-validation results: {cv_results['avg_metrics']}")
        
        # Save CV results
        cv_results_file = os.path.join(args.output_dir, 'cv_results.json')
        with open(cv_results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            cv_results_json = {}
            for key, value in cv_results.items():
                if key == 'fold_results':
                    # Skip fold_results as they contain model objects that can't be serialized
                    continue
                elif isinstance(value, dict):
                    cv_results_json[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            cv_results_json[key][k] = v.tolist()
                        else:
                            cv_results_json[key][k] = v
                else:
                    cv_results_json[key] = value
            
            json.dump(cv_results_json, f, indent=2)
    
    # Train the final model
    logger.info(f"Training final {args.model_type} model...")
    
    model_result = model_trainer.train_model(
        X_train, 
        y_train,
        model_type=args.model_type,
        class_weights=class_weights
    )
    
    # Get the trained model
    if args.model_type == 'logistic_regression':
        model = model_result['best_model']
    else:
        model = model_result['model']
    
    # Save the model if requested
    if args.save_model:
        model_file = os.path.join(args.output_dir, f"{args.model_type}_model.pkl")
        model_trainer.save_model(model, model_file)
        logger.info(f"Model saved to {model_file}")
    
    # Configure model evaluator
    eval_config = {'output_dir': args.output_dir}
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
    predictions_file = os.path.join(args.output_dir, 'predictions.csv')
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
    logger.info(f"Performance report saved to {os.path.join(args.output_dir, 'performance_report.json')}")
    
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
