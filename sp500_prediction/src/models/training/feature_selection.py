"""
Feature Selection Module

This module handles feature selection using various methods including
model-based selection, recursive feature elimination, and Boruta.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import RFECV

from .tree_models import train_random_forest, train_xgboost, train_lightgbm
from .data_filtering import filter_ticker_out_with_nan

# Set up logging
logger = logging.getLogger(__name__)

def perform_model_based_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    n_top_features: int = 20,
    params: Dict = None
) -> Dict:
    """
    Perform feature selection using model-based importance scores.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    model_type : str, optional
        Type of model to use ('random_forest', 'xgboost', 'lightgbm')
    n_top_features : int, optional
        Number of top features to select
    params : Dict, optional
        Model hyperparameters
        
    Returns:
    --------
    Dict
        Dictionary with feature selection results
    """
    logger.info(f"Performing model-based feature selection using {model_type}...")
    
    # Train model to get feature importances
    logger.info(f"Training {model_type} model for feature selection...")
    
    # Ensure y_train is 1d array
    if isinstance(y_train, pd.DataFrame):
        # Take first target column if multiple targets
        y_train = y_train.iloc[:, 0]
    
    if model_type == 'random_forest':
        logger.info("Using RandomForest for feature importance calculation")
        model_result = train_random_forest(X_train, y_train, params=params)
    elif model_type == 'xgboost':
        logger.info("Using XGBoost for feature importance calculation")
        model_result = train_xgboost(X_train, y_train, params=params)
    elif model_type == 'lightgbm':
        logger.info("Using LightGBM for feature importance calculation")
        model_result = train_lightgbm(X_train, y_train, params=params)
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return {}
    
    # Get feature importances
    if 'feature_importances' not in model_result or model_result['feature_importances'] is None:
        logger.error("No feature importances available from the model")
        return {}
    
    # Extract feature importances
    features = model_result['feature_importances']['features']
    importances = model_result['feature_importances']['importances']
    
    # Create DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Select top N features
    top_features = importance_df.head(n_top_features)['feature'].tolist()
    
    # Create reduced feature set from filtered data
    X_reduced = X_train[top_features]  # Already using filtered X_train
    
    return {
        'method': 'model_based',
        'model_type': model_type,
        'importance_df': importance_df,
        'top_features': top_features,
        'X_reduced': X_reduced
    }

def perform_recursive_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    cv_folds: int = 5,
    scoring: str = 'balanced_accuracy',
    params: Dict = None
) -> Dict:
    """
    Perform recursive feature elimination with cross-validation.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    model_type : str, optional
        Type of model to use
    cv_folds : int, optional
        Number of cross-validation folds
    scoring : str, optional
        Scoring metric for feature selection
    params : Dict, optional
        Model hyperparameters
        
    Returns:
    --------
    Dict
        Dictionary with feature selection results
    """
    logger.info(f"Performing recursive feature elimination using {model_type}...")
    
    # Create base model
    if model_type == 'random_forest':
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        base_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    elif model_type == 'lightgbm':
        base_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return {}
    
    # Update model parameters if provided
    if params:
        base_model.set_params(**params)
    
    # Create RFECV selector
    selector = RFECV(
        estimator=base_model,
        step=1,
        cv=cv_folds,
        scoring=scoring,
        min_features_to_select=1
    )
    
    # Fit the selector
    selector.fit(X_train, y_train)
    
    # Get selected features
    selected_features = X_train.columns[selector.support_].tolist()
    
    # Create reduced feature set from filtered data
    X_reduced = X_train[selected_features]  # Already using filtered X_train
    
    # Create importance ranking based on elimination order
    ranking = selector.ranking_
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'ranking': ranking
    })
    importance_df = importance_df.sort_values('ranking')
    
    return {
        'method': 'recursive',
        'model_type': model_type,
        'importance_df': importance_df,
        'selected_features': selected_features,
        'X_reduced': X_reduced,
        'grid_scores': selector.grid_scores_.tolist(),
        'n_features_selected': len(selected_features)
    }

# The FeaturePreservingRandomForest class has been removed.
# We now track feature names explicitly and use standard RandomForestClassifier.

def perform_boruta_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict = None
) -> Dict:
    """
    Perform feature selection using the Boruta algorithm.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    params : Dict, optional
        Random Forest parameters for Boruta
        
    Returns:
    --------
    Dict
        Dictionary with feature selection results
    """
    try:
        from boruta import BorutaPy
    except ImportError:
        logger.error("Boruta package not installed. Please install 'boruta-py'.")
        return {}
    
    logger.info("Performing Boruta feature selection...")
    
    # Create base Random Forest model
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Update model parameters if provided
    if params:
        base_model.set_params(**params)
    
    # Store feature names before Boruta
    feature_names = X_train.columns.tolist()
    
    # Create Boruta selector
    selector = BorutaPy(
        estimator=base_model,
        n_estimators='auto',
        verbose=0,
        random_state=42
    )
    
    # Fit the selector
    # We'll provide DataFrame to our wrapper, which will handle the conversion
    selector.fit(X_train, y_train.values)
    
    # Set feature names on the underlying model (if it doesn't already have them)
    if hasattr(selector.estimator, 'estimator_') and not hasattr(selector.estimator.estimator_, 'feature_names_in_'):
        selector.estimator.estimator_.feature_names_in_ = np.array(feature_names)
    
    # Get confirmed features
    confirmed_features = X_train.columns[selector.support_].tolist()
    
    # Get tentative features
    tentative_features = X_train.columns[selector.support_weak_].tolist()
    
    # Create reduced feature set from filtered data with confirmed features
    X_reduced = X_train[confirmed_features]  # Already using filtered X_train
    
    # Create ranking based on feature importances
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'rank': selector.ranking_,
        'importance': selector.importance_history_,
        'confirmed': selector.support_,
        'tentative': selector.support_weak_
    })
    importance_df = importance_df.sort_values('rank')
    
    return {
        'method': 'boruta',
        'importance_df': importance_df,
        'confirmed_features': confirmed_features,
        'tentative_features': tentative_features,
        'X_reduced': X_reduced,
        'n_features_confirmed': len(confirmed_features)
    }

def perform_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'model_based',
    model_type: str = 'random_forest',
    n_top_features: int = 20,
    params: Dict = None
) -> Dict:
    """
    Perform feature selection after filtering out tickers with NaN values.
    
    This function:
    1. Filters out any tickers that have NaN values
    2. Performs feature selection on the clean data
    3. Returns selected features and the reduced dataset (using filtered data)
    
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
    params : Dict, optional
        Model hyperparameters
        
    Returns:
    --------
    Dict
        Dictionary containing:
        - method: Feature selection method used
        - importance_df: DataFrame with feature importance scores
        - top_features/selected_features: List of selected feature names
        - X_reduced: DataFrame with only selected features (using filtered data)
    """
    # Filter out tickers with NaN values
    logger.info("Filtering out tickers with NaN values before feature selection...")
    X_train_filtered, y_train_filtered, _, _ = filter_ticker_out_with_nan(
        X_train, y_train, X_train, y_train
    )
    
    # Check if we have any data left after filtering
    if len(X_train_filtered) == 0:
        logger.error("No data remaining after filtering out NaN values")
        return {}
    
    # Log data shape for debugging
    logger.info(f"Filtered data shape - X: {X_train_filtered.shape}, y: {len(y_train_filtered)}")
    logger.info(f"Features: {', '.join(X_train_filtered.columns)}")
    
    logger.info(f"Performing feature selection using {method} method...")
    
    # Available feature selection methods
    methods = {
        'model_based': perform_model_based_selection,
        'recursive': perform_recursive_selection,
        'boruta': perform_boruta_selection
    }
    
    if method not in methods:
        logger.error(f"Unsupported feature selection method: {method}")
        return {}
    
    # Call appropriate selection function with filtered data
    if method == 'model_based':
        result = methods[method](X_train_filtered, y_train_filtered, model_type, n_top_features, params)
    elif method == 'recursive':
        result = methods[method](X_train_filtered, y_train_filtered, model_type, params=params)
    else:  # boruta
        result = methods[method](X_train_filtered, y_train_filtered, params)
    
    # Create reduced feature set from filtered data using selected features
    if result and 'top_features' in result:
        selected_features = result['top_features']
        result['X_reduced'] = X_train_filtered[selected_features]
    elif result and 'selected_features' in result:
        selected_features = result['selected_features']
        result['X_reduced'] = X_train_filtered[selected_features]
    
    return result
