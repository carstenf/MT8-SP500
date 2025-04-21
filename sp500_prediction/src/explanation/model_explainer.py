"""
Model Explainer Module

This module provides the main ModelExplainer class that coordinates all explanation components.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from .components.shap_explainer import (
    generate_shap_values,
    create_shap_summary_plot
)
from .components.dependence_plots import (
    create_partial_dependence_plots,
    create_ice_plots
)
from .components.interaction_analyzer import (
    analyze_shap_interactions,
    create_interaction_report
)
from .components.error_analyzer import (
    analyze_correct_vs_incorrect,
    create_error_analysis_report
)
from .components.case_studies import (
    create_explanation_for_stock,
    create_case_study_report
)

# Set up logging
logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Class to handle model explainability for S&P500 prediction.
    
    This class provides methods to:
    - Generate SHAP values and visualizations
    - Create Partial Dependence Plots (PDPs)
    - Create Individual Conditional Expectation (ICE) plots
    - Analyze feature interactions
    - Create case studies for specific stock predictions
    - Compare correct and incorrect predictions
    - Generate comprehensive explanation reports
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the ModelExplainer with optional configuration.
        
        Parameters:
        -----------
        config : Dict, optional
            Configuration dictionary with options for model explanation
        """
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'results/explanation')
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.X_sample = None
        
    def explain_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series = None,
        model_type: str = 'auto',
        n_top_features: int = 20,
        sample_size: int = 100,
        show_plots: bool = False
    ) -> Dict:
        """
        Generate a comprehensive explanation for a trained model.
        
        Parameters:
        -----------
        model : Any
            Trained model to explain
        X : pd.DataFrame
            Feature matrix
        y : pd.Series, optional
            True target values (for error analysis)
        model_type : str, optional
            Type of model ('tree', 'linear', 'deep', or 'auto')
        n_top_features : int, optional
            Number of top features to include in explanations
        sample_size : int, optional
            Number of samples to use for calculations
        show_plots : bool, optional
            Whether to display plots
            
        Returns:
        --------
        Dict
            Dictionary with explanation results
        """
        logger.info("Generating comprehensive model explanation...")
        
        # Update parameters from config
        n_top_features = self.config.get('n_top_features', n_top_features)
        sample_size = self.config.get('sample_size', sample_size)
        
        try:
            # Step 1: Generate SHAP values
            shap_result = generate_shap_values(
                model, X, model_type=model_type, 
                sample_size=sample_size
            )
            
            if not shap_result:
                logger.error("Failed to generate SHAP values")
                return {}
            
            # Store SHAP values for later use
            self.explainer = shap_result['explainer']
            self.shap_values = shap_result['shap_values']
            self.feature_names = X.columns.tolist()
            self.X_sample = shap_result['X_sample']
            
            # Step 2: Create SHAP summary plots
            shap_plots = create_shap_summary_plot(
                self.shap_values, self.X_sample,
                output_dir=self.output_dir,
                max_display=n_top_features,
                show_plots=show_plots
            )
            
            # Step 3: Create PDP plots
            pdp_results = create_partial_dependence_plots(
                model, X,
                n_top_features=n_top_features,
                output_dir=self.output_dir,
                show_plots=show_plots
            )
            
            # Step 4: Create ICE plots
            ice_results = create_ice_plots(
                model, X,
                n_top_features=min(5, n_top_features),
                output_dir=self.output_dir,
                show_plots=show_plots
            )
            
            # Step 5: Analyze interactions
            interaction_results = analyze_shap_interactions(
                model, X,
                n_top_features=n_top_features,
                sample_size=min(50, sample_size),
                output_dir=self.output_dir,
                show_plots=show_plots
            )
            
            interaction_report = create_interaction_report(
                interaction_results,
                output_dir=self.output_dir
            )
            
            # Step 6: Error analysis if y is provided
            error_results = None
            if y is not None:
                error_results = analyze_correct_vs_incorrect(
                    model, X, y,
                    output_dir=os.path.join(self.output_dir, 'error_analysis'),
                    show_plots=show_plots
                )
                
                error_report = create_error_analysis_report(
                    error_results,
                    output_dir=os.path.join(self.output_dir, 'error_analysis')
                )
            
            # Compile results
            explanation = {
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'n_features': len(X.columns),
                'n_samples': len(X),
                'shap_summary': {
                    'plots': shap_plots,
                    'feature_importance': pd.DataFrame({
                        'feature': X.columns,
                        'importance': abs(self.shap_values).mean(axis=0)
                    }).sort_values('importance', ascending=False).to_dict('records')
                },
                'pdp_analysis': pdp_results,
                'ice_analysis': ice_results,
                'interaction_analysis': interaction_report,
                'error_analysis': error_report if y is not None else None
            }
            
            # Save explanation metadata
            metadata_path = os.path.join(self.output_dir, 'explanation_metadata.json')
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump({
                    'timestamp': explanation['timestamp'],
                    'model_type': model_type,
                    'n_features': len(X.columns),
                    'n_samples': len(X),
                    'output_dir': self.output_dir,
                    'parameters': {
                        'n_top_features': n_top_features,
                        'sample_size': sample_size
                    }
                }, f, indent=2)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating model explanation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def explain_stock_prediction(
        self,
        model: Any,
        X: pd.DataFrame,
        ticker: str,
        date: pd.Timestamp,
        show_plots: bool = False
    ) -> Dict:
        """
        Create detailed explanation for a specific stock's prediction.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X : pd.DataFrame
            Feature matrix with multi-index (ticker, date)
        ticker : str
            Stock ticker to explain
        date : pd.Timestamp
            Date for the prediction
        show_plots : bool, optional
            Whether to display plots
            
        Returns:
        --------
        Dict
            Dictionary with explanation for the stock
        """
        stock_dir = os.path.join(self.output_dir, 'stocks', ticker)
        
        return create_explanation_for_stock(
            model, X, ticker, date,
            output_dir=stock_dir,
            shap_values=self.shap_values,
            explainer=self.explainer,
            X_sample=self.X_sample,
            show_plots=show_plots
        )
    
    def create_case_studies(
        self,
        model: Any,
        X: pd.DataFrame,
        tickers: List[str],
        dates: List[pd.Timestamp],
        show_plots: bool = False
    ) -> Dict:
        """
        Create case studies for multiple stocks.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X : pd.DataFrame
            Feature matrix with multi-index (ticker, date)
        tickers : List[str]
            List of stock tickers to analyze
        dates : List[pd.Timestamp]
            List of dates to analyze
        show_plots : bool, optional
            Whether to display plots
            
        Returns:
        --------
        Dict
            Dictionary with case study results
        """
        return create_case_study_report(
            model, X, tickers, dates,
            output_dir=os.path.join(self.output_dir, 'case_studies'),
            show_plots=show_plots
        )
    
    def analyze_errors(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        show_plots: bool = False
    ) -> Dict:
        """
        Analyze model errors.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            True target values
        show_plots : bool, optional
            Whether to display plots
            
        Returns:
        --------
        Dict
            Dictionary with error analysis results
        """
        error_dir = os.path.join(self.output_dir, 'error_analysis')
        
        results = analyze_correct_vs_incorrect(
            model, X, y,
            output_dir=error_dir,
            show_plots=show_plots
        )
        
        return create_error_analysis_report(results, error_dir)
    
    def save_explainer(self, file_path: str) -> bool:
        """
        Save the SHAP explainer to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the explainer
            
        Returns:
        --------
        bool
            True if save succeeded
        """
        if self.explainer is None:
            logger.error("No explainer to save. Run explain_model() first.")
            return False
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(self.explainer, f)
            
            logger.info(f"Saved explainer to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving explainer: {str(e)}")
            return False
    
    def load_explainer(self, file_path: str) -> bool:
        """
        Load the SHAP explainer from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the saved explainer
            
        Returns:
        --------
        bool
            True if load succeeded
        """
        try:
            with open(file_path, 'rb') as f:
                self.explainer = pickle.load(f)
            
            logger.info(f"Loaded explainer from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading explainer: {str(e)}")
            return False
