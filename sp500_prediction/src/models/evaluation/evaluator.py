"""
Model Evaluator Module

This module provides a unified interface for model evaluation using the
metrics, analysis, visualization, and reporting components.
"""

import logging
from typing import Dict, Any, Optional

from .metrics_calculator import calculate_performance_metrics
from .analysis import analyze_predictions_by_ticker, analyze_predictions_by_time
from .visualization import (
    create_performance_visualizations,
    create_feature_importance_plot,
    create_time_series_performance_plot
)
from .reporting import generate_performance_report

# Set up logging
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class that coordinates the use of various
    evaluation components to assess model performance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ModelEvaluator with optional configuration.
        
        Parameters:
        -----------
        config : Dict, optional
            Configuration dictionary with options for model evaluation
        """
        self.config = self._load_default_config()
        if config is not None:
            self.config.update(config)
        
        self.results = {}
        self.output_dir = self.config['output_dirs']['base']
        self.plots_dir = self.config['output_dirs']['plots']
    
    def _load_default_config(self) -> Dict:
        """Load default configuration settings."""
        return {
            "output_dirs": {
                "base": "results",
                "plots": "results/plots"
            },
            "metrics": {
                "balanced_accuracy_threshold": 0.55,
                "zero_division": 0,
                "slope_threshold": 0.01
            },
            "visualization": {
                "plot_figsize": {
                    "default": [10, 6],
                    "feature_importance": [12, 8],
                    "confusion_matrix": [8, 6]
                },
                "plot_style": {
                    "linewidth": 2,
                    "alpha": 0.1
                }
            },
            "time_analysis": {
                "time_unit": "month",
                "min_periods_for_trend": 3
            }
        }
    
    def evaluate_model(
        self,
        model: Any,
        X_test: Any,
        y_test: Any,
        y_pred: Any = None,
        y_prob: Any = None
    ) -> Dict:
        """
        Evaluate a model's performance on test data.
        
        Parameters:
        -----------
        model : Any
            Trained model instance
        X_test : Any
            Test feature matrix
        y_test : Any
            True target values
        y_pred : Any, optional
            Pre-calculated predictions
        y_prob : Any, optional
            Pre-calculated prediction probabilities
            
        Returns:
        --------
        Dict
            Dictionary containing evaluation metrics
        """
        # Calculate metrics
        metrics = calculate_performance_metrics(
            y_test,
            y_pred if y_pred is not None else model.predict(X_test),
            y_prob if y_prob is not None else (
                model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            ),
            config=self.config.get('metrics')
        )
        
        # Store the results
        self.results['overall_metrics'] = metrics
        
        return metrics
    
    def analyze_by_ticker(
        self,
        predictions: Any,
        excess_returns: Any
    ) -> Dict:
        """
        Analyze prediction performance by ticker.
        
        Parameters:
        -----------
        predictions : Any
            DataFrame with predictions
        excess_returns : Any
            DataFrame with excess returns
            
        Returns:
        --------
        Dict
            Dictionary with ticker analysis results
        """
        ticker_analysis = analyze_predictions_by_ticker(
            predictions,
            excess_returns,
            config=self.config.get('metrics')
        )
        
        # Store the results
        self.results['ticker_analysis'] = ticker_analysis
        
        return ticker_analysis
    
    def analyze_by_time(
        self,
        predictions: Any,
        time_unit: Optional[str] = None
    ) -> Dict:
        """
        Analyze prediction performance over time.
        
        Parameters:
        -----------
        predictions : Any
            DataFrame with predictions
        time_unit : str, optional
            Time unit for analysis
            
        Returns:
        --------
        Dict
            Dictionary with time analysis results
        """
        if time_unit is None:
            time_unit = self.config.get('time_analysis', {}).get('time_unit', 'month')
            
        time_analysis = analyze_predictions_by_time(
            predictions,
            time_unit,
            config=self.config.get('metrics')
        )
        
        # Store the results
        self.results['time_analysis'] = time_analysis
        
        return time_analysis
    
    def create_visualizations(self) -> Dict:
        """
        Create all visualization plots.
        
        Returns:
        --------
        Dict
            Dictionary with paths to generated plots
        """
        viz_paths = {}
        
        # Create performance visualizations if metrics are available
        if 'overall_metrics' in self.results:
            perf_plots = create_performance_visualizations(
                self.results['overall_metrics'],
                self.plots_dir,
                self.config.get('visualization')
            )
            viz_paths.update(perf_plots)
        
        # Create feature importance plot if available
        if 'feature_importance' in self.results:
            fi_plot = create_feature_importance_plot(
                self.results['feature_importance'],
                self.plots_dir,
                viz_config=self.config.get('visualization')
            )
            if fi_plot:
                viz_paths['feature_importance'] = fi_plot
        
        # Create time series plot if available
        if 'time_analysis' in self.results:
            ts_plot = create_time_series_performance_plot(
                self.results['time_analysis'],
                self.plots_dir,
                viz_config=self.config.get('visualization')
            )
            if ts_plot:
                viz_paths['time_series'] = ts_plot
        
        # Store the visualization paths
        self.results['visualizations'] = viz_paths
        
        return viz_paths
    
    def generate_report(self, output_file: Optional[str] = None) -> bool:
        """
        Generate comprehensive performance report.
        
        Parameters:
        -----------
        output_file : str, optional
            Path to save the report
            
        Returns:
        --------
        bool
            True if report generation succeeded
        """
        if output_file is None:
            output_file = f"{self.output_dir}/performance_report.json"
            
        return generate_performance_report(self.results, output_file)
    
    def run_full_evaluation(
        self,
        model: Any,
        X_test: Any,
        y_test: Any,
        predictions: Any = None,
        excess_returns: Any = None,
        time_unit: Optional[str] = None
    ) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Parameters:
        -----------
        model : Any
            Trained model instance
        X_test : Any
            Test feature matrix
        y_test : Any
            True target values
        predictions : Any, optional
            DataFrame with predictions
        excess_returns : Any, optional
            DataFrame with excess returns
        time_unit : str, optional
            Time unit for temporal analysis
            
        Returns:
        --------
        Dict
            Dictionary with all evaluation results
        """
        logger.info("Running full evaluation pipeline...")
        
        # Basic model evaluation
        self.evaluate_model(model, X_test, y_test)
        
        # Additional analyses if predictions DataFrame is provided
        if predictions is not None:
            # Analyze by ticker if excess returns are provided
            if excess_returns is not None:
                self.analyze_by_ticker(predictions, excess_returns)
            
            # Analyze by time
            self.analyze_by_time(predictions, time_unit)
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        logger.info("Full evaluation pipeline completed")
        
        return self.results
