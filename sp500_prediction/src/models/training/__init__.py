"""
Model Training Package

This package provides comprehensive tools for training and evaluating
machine learning models for S&P500 stock direction prediction.

Components:
- baseline_models: Logistic regression and neural network models
- tree_models: Random Forest, XGBoost, and LightGBM models
- feature_selection: Feature selection methods
- evaluation: Model evaluation and cross-validation
- model_trainer: Main ModelTrainer class that coordinates all components
"""

from .baseline_models import (
    train_baseline_model,
    train_neural_network
)
from .tree_models import (
    train_random_forest,
    train_xgboost,
    train_lightgbm
)
from .feature_selection import (
    perform_feature_selection,
    perform_model_based_selection,
    perform_recursive_selection,
    perform_boruta_selection
)
from .evaluation import (
    evaluate_model,
    perform_cross_validation
)
from .model_trainer import ModelTrainer

__all__ = [
    # Main interface
    'ModelTrainer',
    
    # Baseline models
    'train_baseline_model',
    'train_neural_network',
    
    # Tree-based models
    'train_random_forest',
    'train_xgboost',
    'train_lightgbm',
    
    # Feature selection
    'perform_feature_selection',
    'perform_model_based_selection',
    'perform_recursive_selection',
    'perform_boruta_selection',
    
    # Evaluation
    'evaluate_model',
    'perform_cross_validation'
]
