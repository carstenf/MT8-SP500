"""
Tests for the YellowbrickEvaluator class.
"""

import os
import shutil
import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.evaluation import YellowbrickEvaluator

class TestYellowbrickEvaluator(unittest.TestCase):
    """Test cases for YellowbrickEvaluator."""

    def setUp(self):
        """Set up test environment."""
        # Create test directory
        self.test_dir = 'test_results'
        os.makedirs(self.test_dir, exist_ok=True)

        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Create sample data with meaningful feature names
        feature_names = ['price', 'volume', 'momentum', 'volatility', 'trend']
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=feature_names
        )
        self.y = pd.Series(
            np.random.randint(0, 2, n_samples),
            name='target'
        )
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Create and train a model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        # Set feature names (for Yellowbrick)
        self.model.feature_names = feature_names

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_evaluate_model(self):
        """Test model evaluation."""
        evaluator = YellowbrickEvaluator({'output_dir': self.test_dir})
        
        # Evaluate model
        results = evaluator.evaluate_model(self.model, self.X_test, self.y_test)
        
        # Check results
        self.assertIn('metrics', results)
        self.assertIn('plots', results)
        
        # Check metrics
        metrics = results['metrics']
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('auc', metrics)
        
        # Check plots
        plots = results['plots']
        self.assertIn('classification_report', plots)
        self.assertIn('confusion_matrix', plots)
        self.assertIn('roc_curve', plots)
        self.assertIn('pr_curve', plots)
        self.assertIn('feature_importance', plots)
        
        # Check if plots were created
        plot_dir = os.path.join(self.test_dir, 'plots')
        self.assertTrue(os.path.exists(plot_dir))
        self.assertTrue(len(os.listdir(plot_dir)) > 0)

    def test_generate_report(self):
        """Test report generation."""
        evaluator = YellowbrickEvaluator({'output_dir': self.test_dir})
        
        # First evaluate model
        evaluator.evaluate_model(self.model, self.X_test, self.y_test)
        
        # Generate report
        success = evaluator.generate_report()
        self.assertTrue(success)
        
        # Check if report was created
        report_path = os.path.join(self.test_dir, 'performance_report.json')
        self.assertTrue(os.path.exists(report_path))

    def test_create_learning_curve(self):
        """Test learning curve creation."""
        evaluator = YellowbrickEvaluator({'output_dir': self.test_dir})
        
        # Create learning curve
        success = evaluator.create_learning_curve(
            self.model, self.X_train, self.y_train
        )
        self.assertTrue(success)
        
        # Check if learning curve plot was created
        plot_path = os.path.join(self.test_dir, 'plots', 'learning_curve.png')
        self.assertTrue(os.path.exists(plot_path))

if __name__ == '__main__':
    unittest.main()
