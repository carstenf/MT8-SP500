"""
Baseline Models Module

This module handles training and evaluation of baseline models like logistic regression.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

# Set up logging
logger = logging.getLogger(__name__)

def train_baseline_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    class_weights: Dict = None
) -> Dict:
    """
    Train a baseline logistic regression model with regularization.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    class_weights : Dict, optional
        Class weights for handling imbalanced classes
        
    Returns:
    --------
    Dict
        Dictionary with trained model and training information
    """
    logger.info("Training baseline logistic regression model...")
    
    # Create logistic regression models with different regularization
    models = {
        'l1': LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            class_weight=class_weights,
            max_iter=1000,
            random_state=42
        ),
        'l2': LogisticRegression(
            penalty='l2', 
            solver='liblinear', 
            class_weight=class_weights,
            max_iter=1000,
            random_state=42
        ),
        'no_penalty': LogisticRegression(
            penalty=None, 
            solver='lbfgs', 
            class_weight=class_weights,
            max_iter=1000,
            random_state=42
        )
    }
    
    # Train each model
    trained_models = {}
    train_scores = {}
    feature_importances = {}
    training_times = {}
    
    for name, model in models.items():
        start_time = datetime.now()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Record training time
        training_time = (datetime.now() - start_time).total_seconds()
        training_times[name] = training_time
        
        # Get training scores
        y_train_pred = model.predict(X_train)
        train_scores[name] = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        }
        
        # Get feature importances (coefficients for logistic regression)
        feature_importances[name] = {
            'features': X_train.columns.tolist(),
            'importances': model.coef_[0].tolist()
        }
        
        # Store trained model
        trained_models[name] = model
        
        logger.info(f"Trained {name} logistic regression: "
                    f"Balanced accuracy: {train_scores[name]['balanced_accuracy']:.4f}, "
                    f"Training time: {training_time:.2f} seconds")
    
    # Select best model based on balanced accuracy
    best_reg_type = max(train_scores, key=lambda k: train_scores[k]['balanced_accuracy'])
    best_model = trained_models[best_reg_type]
    best_score = train_scores[best_reg_type]['balanced_accuracy']
    
    logger.info(f"Best baseline model: {best_reg_type} with balanced accuracy: {best_score:.4f}")
    
    return {
        'best_model': best_model,
        'best_reg_type': best_reg_type,
        'best_score': best_score,
        'all_models': trained_models,
        'train_scores': train_scores,
        'feature_importances': feature_importances,
        'training_times': training_times
    }

def train_neural_network(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    class_weights: Dict = None,
    params: Dict = None
) -> Dict:
    """
    Train a simple Neural Network (MLP) classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    class_weights : Dict, optional
        Class weights for handling imbalanced classes
    params : Dict, optional
        Model hyperparameters
        
    Returns:
    --------
    Dict
        Dictionary with trained model and training information
    """
    from sklearn.neural_network import MLPClassifier
    
    logger.info("Training Neural Network classifier...")
    
    # Default parameters
    default_params = {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 'auto',
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 200,
        'early_stopping': True,
        'validation_fraction': 0.2,
        'random_state': 42
    }
    
    # Update with provided parameters if any
    if params:
        default_params.update(params)
    
    # Create and train the model
    start_time = datetime.now()
    
    model = MLPClassifier(**default_params)
    
    # MLPClassifier doesn't accept class_weight or sample_weight
    # Just fit without weights
    model.fit(X_train, y_train)
    
    # Record training time
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Get training scores
    y_train_pred = model.predict(X_train)
    train_scores = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    }
    
    logger.info(f"Trained Neural Network: "
                f"Balanced accuracy: {train_scores['balanced_accuracy']:.4f}, "
                f"Training time: {training_time:.2f} seconds")
    
    return {
        'model': model,
        'train_scores': train_scores,
        'feature_importances': None,  # Neural networks don't have built-in feature importances
        'training_time': training_time,
        'params': default_params,
        'loss_curve': model.loss_curve_ if hasattr(model, 'loss_curve_') else None
    }
