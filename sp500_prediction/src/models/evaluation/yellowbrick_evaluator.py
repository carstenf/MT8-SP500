"""
Model evaluator using Yellowbrick for visualization and evaluation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from yellowbrick.classifier import (
    ClassificationReport, ConfusionMatrix,
    ROCAUC, PrecisionRecallCurve
)
from yellowbrick.model_selection import LearningCurve
from yellowbrick.features import FeatureImportances
from .analysis import analyze_predictions_by_time

logger = logging.getLogger(__name__)

class YellowbrickEvaluator:
    """Class for model evaluation using Yellowbrick visualizations."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the evaluator."""
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'results')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.results = {}

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """Evaluate model performance using various metrics and visualizations."""
        try:
            # Generate predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Create visualizations
            plots = {}
            
            # Classification Report
            viz = ClassificationReport(model)
            viz.fit(X_test, y_test)
            viz.score(X_test, y_test)
            plot_path = os.path.join(self.plots_dir, 'classification_report.png')
            viz.show(outpath=plot_path)
            plots['classification_report'] = plot_path
            
            # Confusion Matrix
            viz = ConfusionMatrix(model)
            viz.fit(X_test, y_test)
            viz.score(X_test, y_test)
            plot_path = os.path.join(self.plots_dir, 'confusion_matrix.png')
            viz.show(outpath=plot_path)
            plots['confusion_matrix'] = plot_path
            
            # ROC Curve
            viz = ROCAUC(model)
            viz.fit(X_test, y_test)
            viz.score(X_test, y_test)
            plot_path = os.path.join(self.plots_dir, 'roc_curve.png')
            viz.show(outpath=plot_path)
            plots['roc_curve'] = plot_path
            
            # Precision-Recall Curve
            viz = PrecisionRecallCurve(model)
            viz.fit(X_test, y_test)
            viz.score(X_test, y_test)
            plot_path = os.path.join(self.plots_dir, 'pr_curve.png')
            viz.show(outpath=plot_path)
            plots['precision_recall_curve'] = plot_path
            
            # Feature Importance (if model supports it)
            # Create feature importance plot with adjusted figure size
            plt.figure(figsize=(12, 6))
            try:
                viz = FeatureImportances(model, fig=plt.gcf(), labels=X_test.columns)
                viz.fit(X_test, y_test)
                plot_path = os.path.join(self.plots_dir, 'feature_importance.png')
                viz.show(outpath=plot_path)
                plots['feature_importance'] = plot_path
            except Exception as e:
                logger.warning(f"Could not create feature importance plot: {str(e)}")
            plt.close()

            # Store results
            # Perform time-based analysis
            time_analysis_config = self.config.get('time_analysis', {})
            predictions_df = pd.DataFrame({
                'prediction': y_pred,
                'true': y_test,
                'probability': y_pred_proba
            }, index=X_test.index)
            
            time_analysis = analyze_predictions_by_time(
                predictions_df,
                time_unit=time_analysis_config.get('time_unit', 'month'),
                config=self.config
            )

            # Store results
            self.results = {
                'metrics': metrics,
                'plots': plots,
                'time_analysis': time_analysis
            }
            
            return self.results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {}

    def generate_report(self) -> bool:
        """Generate evaluation report."""
        if not self.results:
            logger.error("No results to report. Run evaluate_model first.")
            return False
        
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'metrics': self.results.get('metrics', {}),
                'plots': self.results.get('plots', {}),
                'time_analysis': self.results.get('time_analysis', {})
            }
            
            report_path = os.path.join(self.output_dir, 'performance_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            logger.info(f"Report saved to {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return False

    def create_learning_curve(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 3,
        max_samples: int = 2500
    ) -> bool:
        """Create a learning curve visualization with limited samples.
        
        Parameters:
        -----------
        model : Any
            The model to evaluate
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets
        cv : int, optional
            Number of cross-validation folds (default=3)
        max_samples : int, optional
            Maximum number of samples to use (default=2500)
        """
        logger.info("Create a learning curve visualization")
        
        try:
            # Limit the number of samples if needed
            if len(X_train) > max_samples:
                sample_indices = np.random.choice(len(X_train), max_samples, replace=False)
                X_train = X_train.iloc[sample_indices]
                y_train = y_train.iloc[sample_indices]
                logger.info(f"Using {max_samples} randomly sampled training examples")
            
            # Configure train sizes to use 5 points
            train_sizes = np.linspace(0.2, 1.0, 5)
            
            # Create progress bar wrapper for the fit process
            with tqdm(total=100, desc="Creating learning curve") as pbar:
                # Custom scorer that updates the progress bar
                def progress_scoring(estimator, X, y):
                    pbar.update(100 // (cv * len(train_sizes)))  # Update based on total steps
                    return model.score(X, y)

                viz = LearningCurve(
                    model,
                    cv=cv,
                    train_sizes=train_sizes,
                    scoring=progress_scoring,
                    n_jobs=1,  # Set to 1 to make progress bar work correctly
                    groups=None
                )
                viz.fit(X_train, y_train)
            plot_path = os.path.join(self.plots_dir, 'learning_curve.png')
            viz.show(outpath=plot_path)
            
            if 'plots' not in self.results:
                self.results['plots'] = {}
            self.results['plots']['learning_curve'] = plot_path
            
            return True
            
        except Exception as e:
            logger.error(f"Learning curve creation failed: {str(e)}")
            return False
