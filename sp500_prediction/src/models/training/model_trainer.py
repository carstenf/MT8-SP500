"""
Model Trainer Module

This module provides the main ModelTrainer class that coordinates all training components.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Any
import pandas as pd

from .baseline_models import train_baseline_model, train_neural_network
from .tree_models import train_random_forest, train_xgboost, train_lightgbm
from .feature_selection import perform_feature_selection
from .data_filtering import filter_ticker_out_with_nan

# Set up logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class to handle model training and evaluation for S&P500 prediction.
    
    This class provides methods to:
    - Train various types of models
    - Perform feature selection
    - Evaluate models
    - Perform cross-validation
    - Save and load trained models
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
        self.best_model_score = -1
        self.feature_importances = None
    
    def train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
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
        
        # Filter data to include only samples with complete features
        if X_val is not None and y_val is not None:
            X_train, y_train, X_val, y_val = self._filter_ticker_out_with_nan(X_train, y_train, X_val, y_val)
        else:
            # If no validation set, use the same data for both train and validation to filter
            X_train, y_train, _, _ = filter_ticker_out_with_nan(X_train, y_train, X_train, y_train)
        
        # Skip training if no valid data remains
        if len(X_train) == 0:
            logger.error("No valid samples remaining after filtering incomplete data")
            return {}
            
        # Train the model
        if model_type == 'logistic_regression':
            model_result = train_func(X_train, y_train, class_weights=class_weights)
            model = model_result['best_model']
        elif model_type in ['xgboost', 'lightgbm'] and X_val is not None and y_val is not None:
            model_result = train_func(X_train, y_train, X_val, y_val, 
                                    class_weights=class_weights, params=params)
            model = model_result['model']
        else:
            model_result = train_func(X_train, y_train, 
                                    class_weights=class_weights, params=params)
            model = model_result['model']
        
        # Store the model and results
        self.models[model_type] = model_result
        
        # Update best model if this is the first or best model so far
        current_score = (
            model_result['val_scores']['balanced_accuracy'] 
            if 'val_scores' in model_result and model_result['val_scores'] is not None 
            else model_result['train_scores']['balanced_accuracy']
        )
        
        if self.best_model is None or current_score > self.best_model_score:
            self.best_model = model
            self.best_model_name = model_type
            self.best_model_score = current_score
            
            # Update feature importances if available
            if 'feature_importances' in model_result and model_result['feature_importances'] is not None:
                self.feature_importances = model_result['feature_importances']
        
        return model_result
    
    def train_multiple_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
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
        # Default model configurations
        default_configs = {
            'logistic_regression': {},
            'random_forest': {},
            'xgboost': {},
            'lightgbm': {},
            'neural_network': {}
        }
        
        # Update with provided configurations
        if model_configs:
            for model_type, config in model_configs.items():
                if model_type in default_configs:
                    default_configs[model_type] = config
        
        # Train each model type
        results = {}
        for model_type, params in default_configs.items():
            logger.info(f"Training {model_type} model...")
            result = self.train_model(
                X_train, y_train,
                X_val=X_val,
                y_val=y_val,
                model_type=model_type,
                class_weights=class_weights,
                params=params
            )
            results[model_type] = result
        
        return {
            'models': results,
            'best_model_name': self.best_model_name,
            'best_model': self.best_model,
            'best_score': self.best_model_score
        }
    
    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = 'model_based',
        model_type: str = 'random_forest',
        n_top_features: int = 20,
        params: Dict = None
    ) -> Dict:
        """
        Perform feature selection.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        method : str, optional
            Feature selection method
        model_type : str, optional
            Type of model to use
        n_top_features : int, optional
            Number of top features to select
        params : Dict, optional
            Model hyperparameters
            
        Returns:
        --------
        Dict
            Dictionary with feature selection results
        """
        return perform_feature_selection(
            X_train, y_train,
            method=method,
            model_type=model_type,
            n_top_features=n_top_features,
            params=params
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
    
    def predict(
        self,
        model: Any,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.Series:
        """
        Make predictions with a trained model. Will filter out samples with NaN values.
        
        Parameters:
        -----------
        model : Any
            Model to use for predictions
        X : pd.DataFrame
            Features to predict on
        threshold : float, optional
            Probability threshold for positive class prediction
            
        Returns:
        --------
        pd.Series
            Series of predictions (NaN for samples that were filtered out)
        """
        # Create a Series to store predictions, initialized with NaN
        all_predictions = pd.Series(index=X.index, dtype=float)
        
        # Create dummy target variable for filtering
        dummy_y = pd.Series(0, index=X.index)
        
        # Use existing filter method to get complete data
        # Note: This preserves DataFrame structure and feature names
        X_valid, _, _, _ = filter_ticker_out_with_nan(X, dummy_y, X, dummy_y)
        
        if len(X_valid) > 0:
            # Make predictions on valid data using DataFrame directly
            # scikit-learn will handle feature name matching automatically
            if hasattr(model, 'predict_proba'):
                # Get probabilities as numpy array first
                probas = model.predict_proba(X_valid)
                # Convert the second column (positive class probabilities) to Series
                y_prob = pd.Series(probas[:, 1], index=X_valid.index)
                valid_predictions = (y_prob >= threshold).astype(int)
            else:
                valid_predictions = pd.Series(
                    model.predict(X_valid),
                    index=X_valid.index
                )
            
            # Update predictions for valid samples
            all_predictions.loc[X_valid.index] = valid_predictions
            
            n_tickers = len(X_valid.index.get_level_values('ticker').unique())
            logger.info(f"Made predictions for {n_tickers} tickers "
                       f"({len(X_valid)} samples, filtered from {len(X)} samples)")
        else:
            logger.warning("No valid samples for prediction after filtering NaN values")
        
        return all_predictions
