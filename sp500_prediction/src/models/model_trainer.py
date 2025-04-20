"""
Model Trainer Module for S&P500 Prediction Project

This module handles the training, cross-validation, and evaluation of 
machine learning models for S&P500 stock direction prediction.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from typing import Tuple, Dict, List, Optional, Union, Any
from datetime import datetime
import os

# ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from xgboost import XGBClassifier



# For Neural Network
from sklearn.neural_network import MLPClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        'none': LogisticRegression(
            penalty='none', 
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
    
    # Return information about the baseline training
    return {
        'best_model': best_model,
        'best_reg_type': best_reg_type,
        'best_score': best_score,
        'all_models': trained_models,
        'train_scores': train_scores,
        'feature_importances': feature_importances,
        'training_times': training_times
    }


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
        'random_state': 42
    }
    
    # Update with provided parameters if any
    if params:
        default_params.update(params)
    
    # Create and train the model
    start_time = datetime.now()
    
    model = RandomForestClassifier(
        **default_params,
        class_weight=class_weights
    )
    
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
    
    # Return model information
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
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
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
        'random_state': 42
    }
    
    # Update with provided parameters if any
    if params:
        default_params.update(params)
    
    # Handle class weights
    if class_weights:
        # Convert class weights to scale_pos_weight
        # scale_pos_weight is the ratio of negative to positive samples
        scale_pos_weight = class_weights[0] / class_weights[1]
        default_params['scale_pos_weight'] = scale_pos_weight
    
    # Create and train the model
    start_time = datetime.now()



    # Build the model
    model = XGBClassifier(
        base_score=0.5,
        importance_type='gain',
        **default_params  # this should include eval_metric already
    )

    # Fit the model
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True  # ✅ works with your version
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
    
    # Return model information
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
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
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
        'random_state': 42
    }
    
    # Update with provided parameters if any
    if params:
        default_params.update(params)
    
    # Handle class weights
    if class_weights:
        # LightGBM has a different way to handle class weights
        # class_weight parameter accepts a dict mapping class indices to weights
        default_params['class_weight'] = class_weights
    
    # Create and train the model
    start_time = datetime.now()
    
    model = lgb.LGBMClassifier(**default_params)
    
    # If validation data is provided, use early stopping
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
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
    
    # Return model information
    return {
        'model': model,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'feature_importances': feature_importances,
        'training_time': training_time,
        'params': default_params,
        'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else None
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
    
    # MLPClassifier doesn't accept class_weight directly
    # If class weights are provided, we'll use sample weights
    if class_weights:
        # Create sample weights based on class weights
        sample_weights = np.ones(len(y_train))
        for class_val, weight in class_weights.items():
            sample_weights[y_train == class_val] = weight
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
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
    
    # Neural networks don't have built-in feature importances
    # We could use a permutation-based approach, but it's costly
    # For now, we'll just return None
    feature_importances = None
    
    logger.info(f"Trained Neural Network: "
                f"Balanced accuracy: {train_scores['balanced_accuracy']:.4f}, "
                f"Training time: {training_time:.2f} seconds")
    
    # Return model information
    return {
        'model': model,
        'train_scores': train_scores,
        'feature_importances': feature_importances,
        'training_time': training_time,
        'params': default_params,
        'loss_curve': model.loss_curve_ if hasattr(model, 'loss_curve_') else None
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict:
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : Any
        Trained model with predict and predict_proba methods
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target values
    threshold : float, optional
        Probability threshold for positive class prediction
        
    Returns:
    --------
    Dict
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model on test data...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Apply custom threshold if different from default
    if threshold != 0.5:
        y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Get predictions and actual values for detailed analysis
    results_df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred,
        'prob': y_prob
    })
    
    # Log evaluation results
    logger.info(f"Model evaluation results: "
                f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}, "
                f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return {
        'metrics': metrics,
        'results': results_df,
        'threshold': threshold
    }


def train_multiple_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    class_weights: Dict = None,
    model_configs: Dict = None
) -> Dict:
    """
    Train multiple model types with different configurations.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    X_val : pd.DataFrame, optional
        Validation features
    y_val : pd.Series, optional
        Validation target values
    class_weights : Dict, optional
        Class weights for handling imbalanced classes
    model_configs : Dict, optional
        Configuration parameters for different models
        
    Returns:
    --------
    Dict
        Dictionary with trained models and their performance
    """
    logger.info("Training multiple models...")
    
    # Default model configurations
    default_configs = {
        'logistic_regression': {},
        'random_forest': {},
        'xgboost': {},
        'lightgbm': {},
        'neural_network': {}
    }
    
    # Update with provided configurations if any
    if model_configs:
        for model_type, config in model_configs.items():
            if model_type in default_configs:
                default_configs[model_type] = config
    
    # Dictionary to store trained models and results
    models = {}
    
    # Train logistic regression models
    logger.info("Training logistic regression models...")
    lr_result = train_baseline_model(
        X_train, 
        y_train,
        class_weights=class_weights
    )
    models['logistic_regression'] = lr_result
    
    # Train Random Forest
    logger.info("Training Random Forest model...")
    rf_result = train_random_forest(
        X_train, 
        y_train,
        class_weights=class_weights,
        params=default_configs['random_forest']
    )
    models['random_forest'] = rf_result
    
    # Train XGBoost
    logger.info("Training XGBoost model...")
    xgb_result = train_xgboost(
        X_train, 
        y_train,
        X_val=X_val,
        y_val=y_val,
        class_weights=class_weights,
        params=default_configs['xgboost']
    )
    models['xgboost'] = xgb_result
    
    # Train LightGBM
    logger.info("Training LightGBM model...")
    lgb_result = train_lightgbm(
        X_train, 
        y_train,
        X_val=X_val,
        y_val=y_val,
        class_weights=class_weights,
        params=default_configs['lightgbm']
    )
    models['lightgbm'] = lgb_result
    
    # Train Neural Network
    logger.info("Training Neural Network model...")
    nn_result = train_neural_network(
        X_train, 
        y_train,
        class_weights=class_weights,
        params=default_configs['neural_network']
    )
    models['neural_network'] = nn_result
    
    # Determine the best model based on validation scores (or training if no validation)
    best_model_name = None
    best_score = -1
    model_scores = {}
    
    for model_name, model_info in models.items():
        # Get the validation score if available, otherwise use training score
        if model_name == 'logistic_regression':
            # Logistic regression has a different structure with multiple regularization types
            best_reg_type = model_info['best_reg_type']
            score = model_info['train_scores'][best_reg_type]['balanced_accuracy']
            model_scores[model_name] = score
        elif X_val is not None and y_val is not None and 'val_scores' in model_info and model_info['val_scores'] is not None:
            score = model_info['val_scores']['balanced_accuracy']
            model_scores[model_name] = score
        else:
            score = model_info['train_scores']['balanced_accuracy']
            model_scores[model_name] = score
        
        # Update best model if the current one is better
        if score > best_score:
            best_score = score
            best_model_name = model_name
    
    # Get the best model object
    if best_model_name == 'logistic_regression':
        best_model = models[best_model_name]['best_model']
    else:
        best_model = models[best_model_name]['model']
    
    logger.info(f"Best model: {best_model_name} with balanced accuracy: {best_score:.4f}")
    
    return {
        'models': models,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'best_score': best_score,
        'model_scores': model_scores
    }


def perform_cross_validation(
    data_folds: List[Dict],
    model_type: str = 'xgboost',
    class_weights: Dict = None,
    model_params: Dict = None
) -> Dict:
    """
    Perform time-based cross-validation with the specified model type.
    
    Parameters:
    -----------
    data_folds : List[Dict]
        List of dictionaries containing train and test data for each fold
    model_type : str, optional
        Type of model to train ('logistic_regression', 'random_forest', 
        'xgboost', 'lightgbm', 'neural_network')
    class_weights : Dict, optional
        Class weights for handling imbalanced classes
    model_params : Dict, optional
        Model hyperparameters
        
    Returns:
    --------
    Dict
        Dictionary with cross-validation results
    """
    logger.info(f"Performing cross-validation with {model_type} model...")
    
    # Mapping of model types to training functions
    model_trainers = {
        'logistic_regression': train_baseline_model,
        'random_forest': train_random_forest,
        'xgboost': train_xgboost,
        'lightgbm': train_lightgbm,
        'neural_network': train_neural_network
    }
    
    # Check if the requested model type is supported
    if model_type not in model_trainers:
        logger.error(f"Unsupported model type: {model_type}")
        return {}
    
    # Get the appropriate training function
    train_func = model_trainers[model_type]
    
    # Lists to store results for each fold
    fold_models = []
    fold_metrics = []
    fold_feature_importances = []
    fold_training_times = []
    
    # Perform cross-validation
    for i, fold in enumerate(data_folds):
        fold_num = fold['fold_num']
        logger.info(f"Training on fold {fold_num} ({i+1}/{len(data_folds)})")
        
        # Get training and testing data
        X_train = fold['train_features']
        y_train = fold['train_targets']
        X_test = fold['test_features']
        y_test = fold['test_targets']

        # Verify we have data
        if X_train is None or y_train is None or X_test is None or y_test is None:
            logger.error(f"Missing data in fold {fold_num}")
            continue
        
        # Check if we have target data
        if y_train is None or y_test is None:
            logger.error(f"Missing target data in fold {fold_num}")
            continue
        
        # Train the model
        if model_type == 'logistic_regression':
            model_result = train_func(X_train, y_train, class_weights=class_weights)
            model = model_result['best_model']
        elif model_type in ['xgboost', 'lightgbm']:
            # For these models, we'll use a portion of training data as validation
            # to enable early stopping
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=fold_num
            )
            model_result = train_func(X_tr, y_tr, X_val, y_val, class_weights=class_weights, params=model_params)
            model = model_result['model']
        else:
            model_result = train_func(X_train, y_train, class_weights=class_weights, params=model_params)
            model = model_result['model']
        
        # Evaluate the model
        eval_result = evaluate_model(model, X_test, y_test)
        metrics = eval_result['metrics']
        
        # Store training time
        training_time = model_result['training_time'] if 'training_time' in model_result else None
        
        # Store feature importances if available
        feature_importances = None
        if 'feature_importances' in model_result and model_result['feature_importances'] is not None:
            feature_importances = model_result['feature_importances']
        
        # Store results for this fold
        fold_result = {
            'fold_num': fold_num,
            'train_years': fold['train_years'],
            'test_years': fold['test_years'],
            'model': model,
            'metrics': metrics,
            'training_time': training_time,
            'feature_importances': feature_importances
        }
        
        fold_models.append(fold_result)
        fold_metrics.append(metrics)
        if feature_importances is not None:
            fold_feature_importances.append(feature_importances)
        if training_time is not None:
            fold_training_times.append(training_time)
        
        logger.info(f"Fold {fold_num} results: "
                    f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}, "
                    f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Calculate average metrics across folds
    avg_metrics = {}
    if fold_metrics:
        for metric in fold_metrics[0].keys():
            if metric != 'confusion_matrix' and metric != 'classification_report':
                avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
    
    # Calculate average training time
    avg_training_time = np.mean(fold_training_times) if fold_training_times else None
    
    # Analyze feature importance stability across folds
    feature_stability = None
    if fold_feature_importances:
        # Get all unique features
        all_features = set()
        for fi in fold_feature_importances:
            all_features.update(fi['features'])
        
        # Create a DataFrame to store feature importances across folds
        feature_df = pd.DataFrame(index=list(all_features), columns=[f"fold_{i+1}" for i in range(len(fold_feature_importances))])
        
        # Fill the DataFrame with feature importances
        for i, fi in enumerate(fold_feature_importances):
            for feature, importance in zip(fi['features'], fi['importances']):
                feature_df.loc[feature, f"fold_{i+1}"] = importance
        
        # Calculate mean and standard deviation of feature importances
        feature_df['mean'] = feature_df.mean(axis=1)
        feature_df['std'] = feature_df.std(axis=1)
        feature_df['cv'] = feature_df['std'] / feature_df['mean'].abs()  # Coefficient of variation
        
        # Sort by mean importance
        feature_df = feature_df.sort_values('mean', ascending=False)
        
        feature_stability = {
            'feature_importances_df': feature_df,
            'top_features': feature_df.head(20).index.tolist(),
            'stability_score': 1 - feature_df['cv'].mean()  # Higher is more stable
        }
    
    # Format average metrics nicely
    bal_acc = avg_metrics.get('balanced_accuracy', 0)
    roc_auc = avg_metrics.get('roc_auc', 0)
    logger.info(f"Cross-validation complete. "
                f"Average balanced accuracy: {bal_acc:.4f}, "
                f"Average ROC-AUC: {roc_auc:.4f}")
    
    return {
        'model_type': model_type,
        'fold_results': fold_models,
        'avg_metrics': avg_metrics,
        'avg_training_time': avg_training_time,
        'feature_stability': feature_stability
    }


def perform_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'model_based',
    model_type: str = 'random_forest',
    n_top_features: int = 20
) -> Dict:
    """
    Perform feature selection to identify the most important features.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    method : str, optional
        Feature selection method ('model_based', 'recursive', 'boruta')
    model_type : str, optional
        Type of model to use for model-based selection
    n_top_features : int, optional
        Number of top features to select
        
    Returns:
    --------
    Dict
        Dictionary with feature selection results
    """
    logger.info(f"Performing feature selection using {method} method...")
    
    # Available feature selection methods
    methods = ['model_based', 'recursive', 'boruta']
    
    if method not in methods:
        logger.error(f"Unsupported feature selection method: {method}")
        return {}
    
    # Model-based feature selection
    if method == 'model_based':
        # Train a model to get feature importances
        if model_type == 'random_forest':
            model_result = train_random_forest(X_train, y_train)
        elif model_type == 'xgboost':
            model_result = train_xgboost(X_train, y_train)
        elif model_type == 'lightgbm':
            model_result = train_lightgbm(X_train, y_train)
        else:
            logger.error(f"Unsupported model type for feature selection: {model_type}")
            return {}
        
        # Get feature importances
        if 'feature_importances' not in model_result or model_result['feature_importances'] is None:
            logger.error("No feature importances available from the model")
            return {}
        
        # Extract feature importances
        features = model_result['feature_importances']['features']
        importances = model_result['feature_importances']['importances']
        
        # Create a DataFrame for easier manipulation
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Select top N features
        top_features = importance_df.head(n_top_features)['feature'].tolist()
        
        # Create a reduced feature set
        X_reduced = X_train[top_features]
        
        return {
            'method': method,
            'model_type': model_type,
            'importance_df': importance_df,
            'top_features': top_features,
            'X_reduced': X_reduced
        }
    
    # Recursive feature elimination
    elif method == 'recursive':
        from sklearn.feature_selection import RFECV
        
        # Create a base model
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            base_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lightgbm':
            base_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        else:
            logger.error(f"Unsupported model type for feature selection: {model_type}")
            return {}
        
        # Create RFECV selector
        selector = RFECV(
            estimator=base_model,
            step=1,
            cv=5,
            scoring='balanced_accuracy',
            min_features_to_select=1
        )
        
        # Fit the selector
        selector.fit(X_train, y_train)
        
        # Get selected features
        selected_features = X_train.columns[selector.support_].tolist()
        
        # Create a reduced feature set
        X_reduced = X_train[selected_features]
        
        # Create importance ranking based on elimination order
        ranking = selector.ranking_
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'ranking': ranking
        })
        importance_df = importance_df.sort_values('ranking')
        
        return {
            'method': method,
            'model_type': model_type,
            'importance_df': importance_df,
            'selected_features': selected_features,
            'X_reduced': X_reduced,
            'grid_scores': selector.grid_scores_.tolist(),
            'n_features_selected': len(selected_features)
        }
    
    # Boruta feature selection
    elif method == 'boruta':
        try:
            from boruta import BorutaPy
        except ImportError:
            logger.error("Boruta package not installed. Please install 'boruta-py'.")
            return {}
        
        # Create a base model
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            logger.error(f"Unsupported model type for Boruta: {model_type}")
            return {}
        
        # Create Boruta selector
        selector = BorutaPy(
            estimator=base_model,
            n_estimators='auto',
            verbose=0,
            random_state=42
        )
        
        # Fit the selector
        # Boruta requires numpy arrays, not pandas
        selector.fit(X_train.values, y_train.values)
        
        # Get confirmed features
        confirmed_features = X_train.columns[selector.support_].tolist()
        
        # Get tentative features
        tentative_features = X_train.columns[selector.support_weak_].tolist()
        
        # Create a reduced feature set with confirmed features
        X_reduced = X_train[confirmed_features]
        
        # Create a ranking based on feature importances
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'rank': selector.ranking_,
            'importance': selector.importance_history_,
            'confirmed': selector.support_,
            'tentative': selector.support_weak_
        })
        importance_df = importance_df.sort_values('rank')
        
        return {
            'method': method,
            'model_type': model_type,
            'importance_df': importance_df,
            'confirmed_features': confirmed_features,
            'tentative_features': tentative_features,
            'X_reduced': X_reduced,
            'n_features_confirmed': len(confirmed_features)
        }
    
    return {}


class ModelTrainer:
    """
    Class to handle model training and evaluation for S&P500 prediction.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the ModelTrainer with optional configuration.
        
        Parameters:
        -----------
        config : Dict, optional
            Configuration dictionary with options for model training
        """
        self.config = config or {}
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importances = None
        
    def train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        model_type: str = 'xgboost',
        class_weights: Dict = None,
        params: Dict = None
    ) -> Dict:
        """
        Train a specific type of model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target values
        model_type : str, optional
            Type of model to train
        class_weights : Dict, optional
            Class weights for handling imbalanced classes
        params : Dict, optional
            Model hyperparameters
            
        Returns:
        --------
        Dict
            Dictionary with training results
        """
        # Mapping of model types to training functions
        model_trainers = {
            'logistic_regression': train_baseline_model,
            'random_forest': train_random_forest,
            'xgboost': train_xgboost,
            'lightgbm': train_lightgbm,
            'neural_network': train_neural_network
        }
        
        # Check if the requested model type is supported
        if model_type not in model_trainers:
            logger.error(f"Unsupported model type: {model_type}")
            return {}
        
        # Get the appropriate training function
        train_func = model_trainers[model_type]
        
        # Train the model
        if model_type == 'logistic_regression':
            model_result = train_func(X_train, y_train, class_weights=class_weights)
            model = model_result['best_model']
        elif model_type in ['xgboost', 'lightgbm'] and X_val is not None and y_val is not None:
            model_result = train_func(X_train, y_train, X_val, y_val, class_weights=class_weights, params=params)
            model = model_result['model']
        else:
            model_result = train_func(X_train, y_train, class_weights=class_weights, params=params)
            model = model_result['model']
        
        # Store the model and results
        self.models[model_type] = model_result
        
        # Update best model if this is the first or best model so far
        if self.best_model is None or (
            'val_scores' in model_result and model_result['val_scores'] is not None and
            model_result['val_scores']['balanced_accuracy'] > self.best_model_score
        ) or (
            ('val_scores' not in model_result or model_result['val_scores'] is None) and
            model_result['train_scores']['balanced_accuracy'] > self.best_model_score
        ):
            self.best_model = model
            self.best_model_name = model_type
            self.best_model_score = (
                model_result['val_scores']['balanced_accuracy'] 
                if 'val_scores' in model_result and model_result['val_scores'] is not None 
                else model_result['train_scores']['balanced_accuracy']
            )
            
            # Update feature importances if available
            if 'feature_importances' in model_result and model_result['feature_importances'] is not None:
                self.feature_importances = model_result['feature_importances']
        
        return model_result
    
    def train_multiple_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        class_weights: Dict = None,
        model_configs: Dict = None
    ) -> Dict:
        """
        Train multiple model types.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target values
        class_weights : Dict, optional
            Class weights for handling imbalanced classes
        model_configs : Dict, optional
            Configuration parameters for different models
            
        Returns:
        --------
        Dict
            Dictionary with training results for multiple models
        """
        return train_multiple_models(
            X_train, 
            y_train,
            X_val,
            y_val,
            class_weights,
            model_configs
        )
    
    def evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model: Any = None,
        threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate a model on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target values
        model : Any, optional
            Model to evaluate (if None, uses the best model)
        threshold : float, optional
            Probability threshold for positive class prediction
            
        Returns:
        --------
        Dict
            Dictionary with evaluation results
        """
        # Use the best model if none is provided
        if model is None:
            if self.best_model is None:
                logger.error("No model to evaluate. Train a model first.")
                return {}
            model = self.best_model
        
        return evaluate_model(model, X_test, y_test, threshold)
    
    def perform_cross_validation(
        self,
        data_folds: List[Dict],
        model_type: str = 'xgboost',
        class_weights: Dict = None,
        model_params: Dict = None
    ) -> Dict:
        """
        Perform time-based cross-validation.
        
        Parameters:
        -----------
        data_folds : List[Dict]
            List of dictionaries containing train and test data for each fold
        model_type : str, optional
            Type of model to train
        class_weights : Dict, optional
            Class weights for handling imbalanced classes
        model_params : Dict, optional
            Model hyperparameters
            
        Returns:
        --------
        Dict
            Dictionary with cross-validation results
        """
        return perform_cross_validation(
            data_folds,
            model_type,
            class_weights,
            model_params
        )
    
    def save_model(self, model: Any, output_path: str) -> bool:
        """
        Save a trained model to file.
        
        Parameters:
        -----------
        model : Any
            Trained model to save
        output_path : str
            Path to save the model
            
        Returns:
        --------
        bool
            True if save succeeded, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the model
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Saved model to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model to {output_path}: {str(e)}")
            return False
    
    def load_model(self, input_path: str) -> Any:
        """
        Load a trained model from file.
        
        Parameters:
        -----------
        input_path : str
            Path to the saved model
            
        Returns:
        --------
        Any
            Loaded model, or None if loading failed
        """
        try:
            # Load the model
            with open(input_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Loaded model from {input_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {input_path}: {str(e)}")
            return None
    
    def predict(self, X: pd.DataFrame, model: Any = None, threshold: float = 0.5) -> pd.DataFrame:
        """
        Make predictions with a trained model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on
        model : Any, optional
            Model to use for prediction (if None, uses the best model)
        threshold : float, optional
            Probability threshold for positive class prediction
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions
        """
        # Use the best model if none is provided
        if model is None:
            if self.best_model is None:
                logger.error("No model to predict with. Train a model first.")
                return pd.DataFrame()
            model = self.best_model
        
        # Get predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Apply custom threshold if different from default
        if threshold != 0.5:
            y_pred = (y_prob >= threshold).astype(int)
        
        # Create a DataFrame with predictions
        predictions = pd.DataFrame({
            'prediction': y_pred,
            'probability': y_prob
        }, index=X.index)
        
        return predictions
