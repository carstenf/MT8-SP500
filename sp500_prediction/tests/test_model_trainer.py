"""
Unit tests for the model trainer module.
"""

import unittest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Import the model trainer module
import sys
sys.path.append('../src/models')  # Adjust path as needed
from model_trainer import (
    train_baseline_model,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    train_neural_network,
    evaluate_model,
    train_multiple_models,
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
                'train_data': pd.DataFrame({
                    'feature_0': np.random.randn(100),
                    'feature_1': np.random.randn(100),
                    'feature_2': np.random.randn(100),
                    'target': np.random.choice([0, 1], size=100, p=[0.4, 0.6])
                }),
                'test_data': pd.DataFrame({
                    'feature_0': np.random.randn(50),
                    'feature_1': np.random.randn(50),
                    'feature_2': np.random.randn(50),
                    'target': np.random.choice([0, 1], size=50, p=[0.4, 0.6])
                }),
                'eligible_tickers': ['AAPL', 'MSFT', 'GOOGL']
            }
            self.cv_folds.append(fold)