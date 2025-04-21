"""
Model Evaluation Module

This module handles model evaluation and cross-validation functionality.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

from .baseline_models import train_baseline_model
from .tree_models import train_random_forest, train_xgboost, train_lightgbm

# Set up logging
logger = logging.getLogger(__name__)

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
    
    logger.info(f"Model evaluation results: "
                f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}, "
                f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return {
        'metrics': metrics,
        'results': results_df,
        'threshold': threshold
    }

def perform_cross_validation(
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
    logger.info(f"Performing cross-validation with {model_type} model...")
    
    # Mapping of model types to training functions
    model_trainers = {
        'logistic_regression': train_baseline_model,
        'random_forest': train_random_forest,
        'xgboost': train_xgboost,
        'lightgbm': train_lightgbm
    }
    
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
        
        # Train the model
        if model_type == 'logistic_regression':
            model_result = train_func(X_train, y_train, class_weights=class_weights)
            model = model_result['best_model']
        elif model_type in ['xgboost', 'lightgbm']:
            # Use a portion of training data as validation
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=fold_num
            )
            model_result = train_func(X_tr, y_tr, X_val, y_val, 
                                    class_weights=class_weights, 
                                    params=model_params)
            model = model_result['model']
        else:
            model_result = train_func(X_train, y_train, 
                                    class_weights=class_weights, 
                                    params=model_params)
            model = model_result['model']
        
        # Evaluate the model
        eval_result = evaluate_model(model, X_test, y_test)
        metrics = eval_result['metrics']
        
        # Store training time
        training_time = model_result.get('training_time')
        
        # Store feature importances if available
        feature_importances = model_result.get('feature_importances')
        
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
            if metric not in ['confusion_matrix', 'classification_report']:
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
        
        # Create DataFrame for feature importances across folds
        feature_df = pd.DataFrame(
            index=list(all_features), 
            columns=[f"fold_{i+1}" for i in range(len(fold_feature_importances))]
        )
        
        # Fill DataFrame with feature importances
        for i, fi in enumerate(fold_feature_importances):
            for feature, importance in zip(fi['features'], fi['importances']):
                feature_df.loc[feature, f"fold_{i+1}"] = importance
        
        # Calculate mean and standard deviation
        feature_df['mean'] = feature_df.mean(axis=1)
        feature_df['std'] = feature_df.std(axis=1)
        feature_df['cv'] = feature_df['std'] / feature_df['mean'].abs()
        
        # Sort by mean importance
        feature_df = feature_df.sort_values('mean', ascending=False)
        
        feature_stability = {
            'feature_importances_df': feature_df,
            'top_features': feature_df.head(20).index.tolist(),
            'stability_score': 1 - feature_df['cv'].mean()
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
