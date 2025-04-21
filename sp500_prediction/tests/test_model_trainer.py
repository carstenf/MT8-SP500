"""
Unit tests for the model trainer module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta

# Import from the new modular structure
from src.models.training import (
    train_baseline_model,
    train_neural_network,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    evaluate_model,
    perform_cross_validation,
    perform_feature_selection,
    ModelTrainer
)

class TestModelTraining(unittest.TestCase):
    """
    Test cases for model training functions.
    """
    
    def setUp(self):
        """
        Set up test data for model training.
        """
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 500
        n_features = 20
        
        # Generate features
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Generate target with slight class imbalance (60/40)
        self.y_train = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]))
        
        # Create smaller validation and test sets
        n_val = 100
        n_test = 200
        
        self.X_val = pd.DataFrame(
            np.random.randn(n_val, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_val = pd.Series(np.random.choice([0, 1], size=n_val, p=[0.4, 0.6]))
        
        self.X_test = pd.DataFrame(
            np.random.randn(n_test, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_test = pd.Series(np.random.choice([0, 1], size=n_test, p=[0.4, 0.6]))
        
        # Create class weights for handling imbalance
        self.class_weights = {
            0: 1.0,
            1: 0.67  # Approximately 1.0 / (0.6/0.4)
        }
        
        # Create sample CV folds
        self.cv_folds = []
        for i in range(3):  # Create 3 folds for testing
            fold = {
                'fold_num': i + 1,
                'train_years': [2020 + i, 2021 + i],
                'test_years': [2022 + i],
                'train_features': pd.DataFrame({
                    'feature_0': np.random.randn(100),
                    'feature_1': np.random.randn(100),
                    'feature_2': np.random.randn(100)
                }),
                'train_targets': pd.Series(np.random.choice([0, 1], size=100, p=[0.4, 0.6])),
                'test_features': pd.DataFrame({
                    'feature_0': np.random.randn(50),
                    'feature_1': np.random.randn(50),
                    'feature_2': np.random.randn(50)
                }),
                'test_targets': pd.Series(np.random.choice([0, 1], size=50, p=[0.4, 0.6])),
                'eligible_tickers': ['AAPL', 'MSFT', 'GOOGL']
            }
            self.cv_folds.append(fold)
    
    def test_baseline_model(self):
        """Test baseline logistic regression model training."""
        result = train_baseline_model(self.X_train, self.y_train, self.class_weights)
        
        self.assertIsNotNone(result)
        self.assertIn('best_model', result)
        self.assertIn('train_scores', result)
        self.assertIn('feature_importances', result)
    
    def test_random_forest(self):
        """Test random forest model training."""
        result = train_random_forest(self.X_train, self.y_train, self.class_weights)
        
        self.assertIsNotNone(result)
        self.assertIn('model', result)
        self.assertIn('train_scores', result)
        self.assertIn('feature_importances', result)
    
    def test_xgboost(self):
        """Test XGBoost model training."""
        result = train_xgboost(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.class_weights
        )
        
        self.assertIsNotNone(result)
        self.assertIn('model', result)
        self.assertIn('train_scores', result)
        self.assertIn('val_scores', result)
        self.assertIn('feature_importances', result)
    
    def test_lightgbm(self):
        """Test LightGBM model training."""
        result = train_lightgbm(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.class_weights
        )
        
        self.assertIsNotNone(result)
        self.assertIn('model', result)
        self.assertIn('train_scores', result)
        self.assertIn('val_scores', result)
        self.assertIn('feature_importances', result)
    
    def test_neural_network(self):
        """Test neural network model training."""
        result = train_neural_network(self.X_train, self.y_train, self.class_weights)
        
        self.assertIsNotNone(result)
        self.assertIn('model', result)
        self.assertIn('train_scores', result)
        self.assertIn('loss_curve', result)
    
    def test_model_evaluation(self):
        """Test model evaluation functionality."""
        # Train a simple model for testing
        model_result = train_baseline_model(self.X_train, self.y_train)
        model = model_result['best_model']
        
        # Evaluate the model
        eval_result = evaluate_model(model, self.X_test, self.y_test)
        
        self.assertIsNotNone(eval_result)
        self.assertIn('metrics', eval_result)
        self.assertIn('results', eval_result)
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        cv_result = perform_cross_validation(
            self.cv_folds,
            model_type='random_forest',
            class_weights=self.class_weights
        )
        
        self.assertIsNotNone(cv_result)
        self.assertIn('fold_results', cv_result)
        self.assertIn('avg_metrics', cv_result)
    
    def test_feature_selection(self):
        """Test feature selection functionality."""
        result = perform_feature_selection(
            self.X_train,
            self.y_train,
            method='model_based',
            model_type='random_forest'
        )
        
        self.assertIsNotNone(result)
        self.assertIn('importance_df', result)
        self.assertIn('top_features', result)
    
    def test_model_trainer_class(self):
        """Test the ModelTrainer class functionality."""
        trainer = ModelTrainer()
        
        # Test single model training
        result = trainer.train_model(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            model_type='xgboost',
            class_weights=self.class_weights
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(trainer.best_model)
        
        # Test prediction
        predictions = trainer.predict(model=trainer.best_model, X=self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Test model saving and loading
        # Create test directory if it doesn't exist
        test_dir = 'test_results'
        os.makedirs(test_dir, exist_ok=True)
        save_path = os.path.join(test_dir, 'test_model.pkl')
        trainer.save_model(trainer.best_model, save_path)
        loaded_model = trainer.load_model(save_path)
        self.assertIsNotNone(loaded_model)
        
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == '__main__':
    unittest.main()
