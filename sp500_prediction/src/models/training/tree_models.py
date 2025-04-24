"""
Tree-based Models Module

This module handles training and evaluation of tree-based models like
Random Forest, XGBoost, and LightGBM.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

# Set up logging
logger = logging.getLogger(__name__)

def train_random_forest(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    class_weights: Dict = None,
    params: Dict = None
) -> Dict:
    """
    Train a Random Forest classifier.
    
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
    logger.info("Training Random Forest classifier...")
    
    # Default parameters
    default_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True,
        'criterion': 'entropy',
        'n_jobs': -1,  # Enable parallel processing by default
        'random_state': 42
    }
    
    # Update with provided parameters if any
    if params:
        default_params.update(params)
    
    # Create and train the model
    start_time = datetime.now()
    
    # Use standard RandomForestClassifier
    model = RandomForestClassifier(
        **default_params,
        class_weight=class_weights
    )
    
    # Log parallelization settings
    logger.info(f"Training Random Forest with n_jobs={default_params.get('n_jobs')}")
    
    # Fit model - scikit-learn will automatically handle feature names from DataFrame
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
    
    # Get feature importances
    feature_importances = {
        'features': X_train.columns.tolist(),
        'importances': model.feature_importances_.tolist()
    }
    
    logger.info(f"Trained Random Forest: "
                f"Balanced accuracy: {train_scores['balanced_accuracy']:.4f}, "
                f"Training time: {training_time:.2f} seconds")
    
    return {
        'model': model,
        'train_scores': train_scores,
        'feature_importances': feature_importances,
        'training_time': training_time,
        'params': default_params
    }

def train_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    class_weights: Dict = None,
    params: Dict = None
) -> Dict:
    """
    Train an XGBoost classifier with early stopping if validation data is provided.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    X_val : pd.DataFrame, optional
        Validation features for early stopping
    y_val : pd.Series, optional
        Validation target values for early stopping
    class_weights : Dict, optional
        Class weights for handling imbalanced classes
    params : Dict, optional
        Model hyperparameters
        
    Returns:
    --------
    Dict
        Dictionary with trained model and training information
    """
    logger.info("Training XGBoost classifier...")
    
    # Default parameters
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'n_jobs': -1,  # Enable parallel processing
        'random_state': 42
    }
    
    # Update with provided parameters if any
    if params:
        default_params.update(params)
    
    # Handle class weights
    if class_weights:
        # Convert class weights to scale_pos_weight
        scale_pos_weight = class_weights[0] / class_weights[1]
        default_params['scale_pos_weight'] = scale_pos_weight
    
    # Create and train the model
    start_time = datetime.now()
    
    model = xgb.XGBClassifier(
        base_score=0.5,
        importance_type='gain',
        **default_params
    )
    
    # Log parallelization settings
    logger.info(f"Training XGBoost with n_jobs={default_params.get('n_jobs')}")
    
    # Fit the model
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
    else:
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
    
    # Get validation scores if validation data is provided
    val_scores = None
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        val_scores = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred),
            'recall': recall_score(y_val, y_val_pred),
            'f1': f1_score(y_val, y_val_pred),
            'roc_auc': roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        }
    
    # Get feature importances
    feature_importances = {
        'features': X_train.columns.tolist(),
        'importances': model.feature_importances_.tolist()
    }
    
    logger.info(f"Trained XGBoost: "
                f"Balanced accuracy: {train_scores['balanced_accuracy']:.4f}, "
                f"Training time: {training_time:.2f} seconds")
    
    return {
        'model': model,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'feature_importances': feature_importances,
        'training_time': training_time,
        'params': default_params,
        'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None
    }

def train_lightgbm(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    class_weights: Dict = None,
    params: Dict = None
) -> Dict:
    """
    Train a LightGBM classifier with early stopping if validation data is provided.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    X_val : pd.DataFrame, optional
        Validation features for early stopping
    y_val : pd.Series, optional
        Validation target values for early stopping
    class_weights : Dict, optional
        Class weights for handling imbalanced classes
    params : Dict, optional
        Model hyperparameters
        
    Returns:
    --------
    Dict
        Dictionary with trained model and training information
    """
    logger.info("Training LightGBM classifier...")
    
    # Default parameters
    default_params = {
        'n_estimators': 100,
        'num_leaves': 31,
        'max_depth': -1,  # -1 means no limit
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'objective': 'binary',
        'metric': 'auc',
        'n_jobs': -1,  # Enable parallel processing
        'random_state': 42
    }
    
    # Update with provided parameters if any
    if params:
        default_params.update(params)
    
    # Handle class weights
    if class_weights:
        default_params['class_weight'] = class_weights
    
    # Create and train the model
    start_time = datetime.now()
    
    model = lgb.LGBMClassifier(**default_params)
    
    # Log parallelization settings
    logger.info(f"Training LightGBM with n_jobs={default_params.get('n_jobs')}")
    
    # If validation data is provided, use early stopping
    if X_val is not None and y_val is not None:
            eval_result = {}
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=10)],
                eval_metric='binary_logloss'
            )
    else:
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
    
    # Get validation scores if validation data is provided
    val_scores = None
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        val_scores = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred),
            'recall': recall_score(y_val, y_val_pred),
            'f1': f1_score(y_val, y_val_pred),
            'roc_auc': roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        }
    
    # Get feature importances
    feature_importances = {
        'features': X_train.columns.tolist(),
        'importances': model.feature_importances_.tolist()
    }
    
    logger.info(f"Trained LightGBM: "
                f"Balanced accuracy: {train_scores['balanced_accuracy']:.4f}, "
                f"Training time: {training_time:.2f} seconds")
    
    return {
        'model': model,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'feature_importances': feature_importances,
        'training_time': training_time,
        'params': default_params,
        'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else None
    }
