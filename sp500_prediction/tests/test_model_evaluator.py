"""
Unit tests for the model evaluator module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import json
import shutil
from datetime import datetime, timedelta

# Import the model evaluator module
import sys
sys.path.append('../src/models')  # Adjust path as needed
from model_evaluator import (
    calculate_performance_metrics,
    analyze_predictions_by_ticker,
    analyze_predictions_by_time,
    analyze_feature_importance,
    analyze_bias_variance_tradeoff,
    create_performance_visualizations,
    create_feature_importance_plot,
    create_bias_variance_plot,
    create_time_series_performance_plot,
    generate_performance_report,
    ModelEvaluator
)

class TestModelEvaluator(unittest.TestCase):
    """
    Test cases for model evaluator functions.
    """
    
    def setUp(self):
        """
        Set up test data for evaluation.
        """
        np.random.seed(42)
        
        # Create sample prediction data
        self.y_true = np.random.choice([0, 1], size=100)
        self.y_pred = np.random.choice([0, 1], size=100)
        self.y_prob = np.random.random(size=100)
        
        # Create a DataFrame with predictions (multi-index: ticker, date)
        dates = pd.date_range(start='2021-01-01', periods=20, freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
        
        index = pd.MultiIndex.from_product(
            [tickers, dates],
            names=['ticker', 'date']
        )
        
        self.predictions = pd.DataFrame(index=index)
        self.predictions['true'] = np.random.choice([0, 1], size=len(index))
        self.predictions['prediction'] = np.random.choice([0, 1], size=len(index))
        self.predictions['probability'] = np.random.random(size=len(index))
        
        # Create a DataFrame with excess returns
        self.excess_returns = pd.DataFrame(index=index)
        self.excess_returns['excess_return'] = np.random.normal(0, 0.02, size=len(index))
        
        # Create feature matrix and target
        n_samples = 100
        n_features = 20
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = pd.Series(np.random.choice([0, 1], size=n_samples))
        
        self.X_test = pd.DataFrame(
            np.random.randn(50, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_test = pd.Series(np.random.choice([0, 1], size=50))
        
        # Create a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Create a temporary test directory
        self.test_dir = 'test_results'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
    
    def tearDown(self):
        """
        Clean up after tests.
        """
        # Remove the test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_calculate_performance_metrics(self):
        """
        Test the calculation of performance metrics.
        """
        metrics = calculate_performance_metrics(self.y_true, self.y_pred, self.y_prob)
        
        # Check if metrics contain expected keys
        expected_keys = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 
                        'confusion_matrix', 'class_report', 'roc_auc', 'roc_curve', 
                        'pr_curve', 'pr_auc']
        
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check if metrics have valid values
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['balanced_accuracy'] <= 1)
        self.assertTrue(0 <= metrics['precision'] <= 1)
        self.assertTrue(0 <= metrics['recall'] <= 1)
        self.assertTrue(0 <= metrics['f1'] <= 1)
        self.assertTrue(0 <= metrics['roc_auc'] <= 1)
    
    def test_analyze_predictions_by_ticker(self):
        """
        Test the analysis of predictions by ticker.
        """
        ticker_analysis = analyze_predictions_by_ticker(self.predictions, self.excess_returns)
        
        # Check if analysis contains expected keys
        self.assertIn('ticker_metrics', ticker_analysis)
        self.assertIn('aggregate_metrics', ticker_analysis)
        
        # Check if all tickers are included
        tickers = self.predictions.index.get_level_values('ticker').unique()
        for ticker in tickers:
            self.assertIn(ticker, ticker_analysis['ticker_metrics'])
        
        # Check if aggregate metrics are calculated
        agg_metrics = ticker_analysis['aggregate_metrics']
        expected_agg_keys = ['n_tickers', 'avg_balanced_accuracy', 'avg_f1', 
                            'avg_pnl', 'avg_sharpe', 'top_tickers_by_accuracy', 
                            'top_tickers_by_pnl']
        
        for key in expected_agg_keys:
            self.assertIn(key, agg_metrics)
    
    def test_analyze_predictions_by_time(self):
        """
        Test the analysis of predictions by time.
        """
        time_analysis = analyze_predictions_by_time(self.predictions, time_unit='month')
        
        # Check if analysis contains expected keys
        self.assertIn('time_metrics', time_analysis)
        self.assertIn('time_series', time_analysis)
        self.assertIn('trend_direction', time_analysis)
        
        # Check if time_series contains expected keys
        time_series = time_analysis['time_series']
        expected_ts_keys = ['periods', 'balanced_accuracy', 'precision', 'recall', 'f1']
        
        for key in expected_ts_keys:
            self.assertIn(key, time_series)
        
        # Check if trend_direction contains expected keys
        trend_direction = time_analysis['trend_direction']
        expected_trend_keys = ['balanced_accuracy', 'precision', 'recall', 'f1']
        
        for key in expected_trend_keys:
            self.assertIn(key, trend_direction)
    
    def test_analyze_feature_importance(self):
        """
        Test the analysis of feature importance.
        """
        # Test model-based method
        feature_importance = analyze_feature_importance(
            self.model, self.X_test, self.y_test, method='model_based'
        )
        
        # Check if analysis contains expected keys
        expected_keys = ['importance_df', 'top_features', 'features_for_80_importance', 
                        'features_for_90_importance', 'method']
        
        for key in expected_keys:
            self.assertIn(key, feature_importance)
        
        # Check if importance DataFrame has the right structure
        importance_df = feature_importance['importance_df']
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        
        # Check if the number of features is correct
        self.assertEqual(len(importance_df), len(self.X_test.columns))
    
    def test_analyze_bias_variance_tradeoff(self):
        """
        Test the analysis of bias-variance tradeoff.
        """
        bias_variance = analyze_bias_variance_tradeoff(
            self.model, self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        # Check if analysis contains expected keys
        expected_keys = ['train_sizes', 'train_mean_scores', 'train_std_scores', 
                        'test_mean_scores', 'test_std_scores', 'bias', 'variance', 
                        'diagnosis', 'recommendations']
        
        for key in expected_keys:
            self.assertIn(key, bias_variance)
        
        # Check if bias and variance are valid values
        self.assertTrue(0 <= bias_variance['bias'] <= 1)
        self.assertTrue(0 <= bias_variance['variance'] <= 1)
        
        # Check if diagnosis is one of the expected values
        self.assertIn(bias_variance['diagnosis'], ['high_bias', 'high_variance', 'balanced'])
    
    def test_create_performance_visualizations(self):
        """
        Test the creation of performance visualizations.
        """
        # Calculate metrics first
        metrics = calculate_performance_metrics(self.y_true, self.y_pred, self.y_prob)
        
        # Create visualizations
        plot_paths = create_performance_visualizations(metrics, self.test_dir)
        
        # Check if visualization files were created
        expected_plots = ['confusion_matrix.png', 'roc_curve.png', 'pr_curve.png']
        
        for plot in expected_plots:
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, plot)))
        
        # Check if the return value includes the expected plots
        for plot in expected_plots:
            plot_name = plot.split('.')[0]
            self.assertIn(plot_name, plot_paths)
    
    def test_create_feature_importance_plot(self):
        """
        Test the creation of feature importance plot.
        """
        # Get feature importances first
        feature_importance = analyze_feature_importance(
            self.model, self.X_test, self.y_test, method='model_based'
        )
        
        # Create feature importance plot
        plot_path = create_feature_importance_plot(feature_importance, self.test_dir)
        
        # Check if the plot file was created
        self.assertTrue(os.path.exists(plot_path))
    
    def test_create_bias_variance_plot(self):
        """
        Test the creation of bias-variance plot.
        """
        # Get bias-variance analysis first
        bias_variance = analyze_bias_variance_tradeoff(
            self.model, self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        # Create bias-variance plot
        plot_path = create_bias_variance_plot(bias_variance, self.test_dir)
        
        # Check if the plot file was created
        self.assertTrue(os.path.exists(plot_path))
    
    def test_create_time_series_performance_plot(self):
        """
        Test the creation of time series performance plot.
        """
        # Get time analysis first
        time_analysis = analyze_predictions_by_time(self.predictions, time_unit='month')
        
        # Create time series plot
        plot_path = create_time_series_performance_plot(time_analysis, self.test_dir)
        
        # Check if the plot file was created
        self.assertTrue(os.path.exists(plot_path))
    
    def test_generate_performance_report(self):
        """
        Test the generation of performance report.
        """
        # Create a sample evaluation results dictionary
        evaluation_results = {
            'overall_metrics': calculate_performance_metrics(self.y_true, self.y_pred, self.y_prob),
            'ticker_analysis': analyze_predictions_by_ticker(self.predictions, self.excess_returns),
            'time_analysis': analyze_predictions_by_time(self.predictions, time_unit='month'),
            'feature_importance': analyze_feature_importance(self.model, self.X_test, self.y_test),
            'bias_variance': analyze_bias_variance_tradeoff(self.model, self.X_train, self.y_train, self.X_test, self.y_test),
            'visualizations': {
                'confusion_matrix': os.path.join(self.test_dir, 'confusion_matrix.png'),
                'roc_curve': os.path.join(self.test_dir, 'roc_curve.png')
            }
        }
        
        # Generate report
        report_path = os.path.join(self.test_dir, 'performance_report.json')
        result = generate_performance_report(evaluation_results, report_path)
        
        # Check if the function reported success
        self.assertTrue(result)
        
        # Check if the report file was created
        self.assertTrue(os.path.exists(report_path))
        
        # Check if the report is valid JSON
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        # Check if the report has the expected sections
        expected_sections = ['generated_at', 'overall_metrics', 'ticker_analysis', 
                            'time_analysis', 'feature_importance', 'bias_variance_analysis', 
                            'visualizations', 'success_criteria', 'overall_evaluation']
        
        for section in expected_sections:
            self.assertIn(section, report)
    
    def test_model_evaluator_class(self):
        """
        Test the ModelEvaluator class.
        """
        # Create an evaluator instance with custom output directory
        evaluator = ModelEvaluator({'output_dir': self.test_dir})
        
        # Test evaluate_model method
        metrics = evaluator.evaluate_model(self.model, self.X_test, self.y_test)
        self.assertIn('balanced_accuracy', metrics)
        
        # Test analyze_feature_importance method
        feature_importance = evaluator.analyze_feature_importance(self.model, self.X_test, self.y_test)
        self.assertIn('importance_df', feature_importance)
        
        # Test analyze_bias_variance method
        bias_variance = evaluator.analyze_bias_variance(self.model, self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertIn('bias', bias_variance)
        
        # Test analyze_by_ticker method
        ticker_analysis = evaluator.analyze_by_ticker(self.predictions, self.excess_returns)
        self.assertIn('aggregate_metrics', ticker_analysis)
        
        # Test analyze_by_time method
        time_analysis = evaluator.analyze_by_time(self.predictions, 'month')
        self.assertIn('time_series', time_analysis)
        
        # Test create_visualizations method
        visualizations = evaluator.create_visualizations()
        self.assertTrue(len(visualizations) > 0)
        
        # Test generate_report method
        report_result = evaluator.generate_report()
        self.assertTrue(report_result)
        
        # Test run_full_evaluation method
        results = evaluator.run_full_evaluation(
            self.model, self.X_train, self.y_train, self.X_test, self.y_test,
            self.predictions, self.excess_returns
        )
        
        # Check if all expected sections are in the results
        expected_sections = ['overall_metrics', 'ticker_analysis', 'time_analysis', 
                            'feature_importance', 'bias_variance', 'visualizations']
        
        for section in expected_sections:
            self.assertIn(section, results)


if __name__ == '__main__':
    unittest.main()