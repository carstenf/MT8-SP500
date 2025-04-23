"""
Model evaluator using Yellowbrick for visualization and evaluation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    ROCAUC, PrecisionRecallCurve,
    DiscriminationThreshold, ClassPredictionError
)
from yellowbrick.model_selection import LearningCurve
from yellowbrick.features import (
    FeatureImportances, Rank2D,
    JointPlotVisualizer, Manifold
)
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
            
            # Use a smaller subset for visualizations (max 10000 samples)
            if len(X_test) > 10000:
                viz_indices = np.random.choice(len(X_test), 10000, replace=False)
                X_test_viz = X_test.iloc[viz_indices]
                y_test_viz = y_test.iloc[viz_indices]
                y_pred_viz = y_pred[viz_indices]
                y_pred_proba_viz = y_pred_proba[viz_indices]
            else:
                X_test_viz = X_test
                y_test_viz = y_test
                y_pred_viz = y_pred
                y_pred_proba_viz = y_pred_proba

            # Calculate metrics using full dataset
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Set up default plot style
            plt.rcParams.update({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'figure.figsize': [10, 8],
                'figure.dpi': 300
            })

            # Create visualizations
            plots = {}
            
            logger.info("Creating Classification Report...")
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            viz = ClassificationReport(model, ax=ax, fontsize=12)
            viz.fit(X_test_viz, y_test_viz)
            viz.score(X_test_viz, y_test_viz)
            plot_path = os.path.join(self.plots_dir, 'classification_report.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['classification_report'] = plot_path
            
            logger.info("Creating Confusion Matrix...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            viz = ConfusionMatrix(model, ax=ax, fontsize=12)
            viz.fit(X_test, y_test)
            viz.score(X_test, y_test)
            plot_path = os.path.join(self.plots_dir, 'confusion_matrix.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['confusion_matrix'] = plot_path
            
            logger.info("Creating ROC Curve...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            viz = ROCAUC(model, ax=ax)
            viz.fit(X_test_viz, y_test_viz)
            viz.score(X_test_viz, y_test_viz)
            plot_path = os.path.join(self.plots_dir, 'roc_curve.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['roc_curve'] = plot_path
            
            logger.info("Creating Precision-Recall Curve...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            viz = PrecisionRecallCurve(model, ax=ax)
            viz.fit(X_test_viz, y_test_viz)
            viz.score(X_test_viz, y_test_viz)
            plot_path = os.path.join(self.plots_dir, 'pr_curve.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['precision_recall_curve'] = plot_path
            
            logger.info("Creating Discrimination Threshold...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            viz = DiscriminationThreshold(
                model, 
                ax=ax,
                n_trials=10,  # Reduce from default 50 to 10
                cv=None  # Skip cross-validation entirely for speed
            )
            viz.fit(X_test_viz, y_test_viz)
            viz.score(X_test_viz, y_test_viz)
            plot_path = os.path.join(self.plots_dir, 'discrimination_threshold.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['discrimination_threshold'] = plot_path
            
            logger.info("Creating Class Prediction Error...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            viz = ClassPredictionError(model, ax=ax)
            viz.fit(X_test_viz, y_test_viz)
            viz.score(X_test_viz, y_test_viz)
            plot_path = os.path.join(self.plots_dir, 'class_prediction_error.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['class_prediction_error'] = plot_path
            
            logger.info("Creating Feature Importance Plot...")
            try:
                # Create feature importance visualization using Yellowbrick
                fig, ax = plt.subplots(figsize=(12, 10))
                plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.05)
                plot_path = os.path.join(self.plots_dir, 'feature_importance.png')
                viz = FeatureImportances(
                    model,
                    ax=ax,
                    relative=True,
                    topn=40,  # Show only top 40 features
                    fontsize=10,
                    color='#2ecc71',
                    title='Top 40 Most Important Features'
                )
                viz.fit(X_test, y_test)
                viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
                plt.close()
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

            # Store results with converted metrics and additional data for report generation
            self.results = {
                'metrics': {k: float(v) if isinstance(v, (bool, np.bool_)) else v 
                          for k, v in metrics.items()},
                'plots': plots,
                'time_analysis': time_analysis,
                '_model': model,  # Store model for report generation
                '_data': {
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'X_test_viz': X_test_viz,
                    'feature_names': list(X_test.columns)
                }
            }

            # Create correlation visualization
            logger.info("Creating Feature Correlation Plots...")
            try:
                # Get top 40 features by importance
                importances = pd.Series(model.feature_importances_, index=X_test.columns)
                top_features = importances.nlargest(40)
                X_test_subset = X_test_viz[top_features.index]
                
                # Calculate correlation matrix
                corr_matrix = X_test_subset.corr()
                
                # Create correlation plot using seaborn
                fig, ax = plt.subplots(figsize=(15, 15))
                plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
                
                import seaborn as sns
                mask = np.triu(np.ones_like(corr_matrix), k=1)
                sns.heatmap(corr_matrix, 
                           mask=mask,
                           cmap='RdYlBu_r',
                           annot=True,
                           fmt='.2f',
                           square=True,
                           cbar_kws={'label': 'Correlation Coefficient'},
                           ax=ax)
                
                # Rotate x labels
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                
                # Add title
                plt.title("Feature Correlations (Top 40 Features by Importance)", pad=20)
                plot_path = os.path.join(self.plots_dir, 'feature_correlation.png')
                plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                self.results['plots']['feature_correlation'] = plot_path
            except Exception as e:
                logger.warning(f"Could not create correlation plot: {str(e)}")

            return self.results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {}

    def generate_report(self) -> bool:
        """Generate evaluation report in JSON and PDF formats."""
        if not self.results:
            logger.error("No results to report. Run evaluate_model first.")
            return False
        
        try:
            # Generate JSON report
            # Convert NumPy types to standard Python types
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, (np.float32, np.float64, np.bool_)):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                return obj

            # Format metrics to 4 decimal places
            metrics = {}
            for k, v in self.results.get('metrics', {}).items():
                if isinstance(v, (float, np.float32, np.float64)):
                    metrics[k] = float(f"{v:.4f}")
                else:
                    metrics[k] = convert_numpy_types(v)
            
            # Filter and format time analysis data
            time_analysis = {}
            raw_analysis = self.results.get('time_analysis', {})
            for k, v in raw_analysis.items():
                if isinstance(v, dict):
                    time_analysis[k] = {
                        metric: float(f"{value:.4f}") if isinstance(value, (float, np.float32, np.float64)) else value
                        for metric, value in v.items()
                        if not isinstance(value, np.ndarray)  # Skip large arrays
                    }
                elif isinstance(v, (float, np.float32, np.float64)):
                    time_analysis[k] = float(f"{v:.4f}")
                else:
                    time_analysis[k] = v

            # Get model and data from stored results
            stored_model = self.results.get('_model')
            data = self.results.get('_data', {})
            X_test = data.get('X_test')
            y_test = data.get('y_test')
            y_pred = data.get('y_pred')
            X_test_viz = data.get('X_test_viz')
            feature_names = data.get('feature_names', [])
            
            # Add feature importances if available
            feature_importances = {}
            if stored_model is not None and hasattr(stored_model, 'feature_importances_'):
                # Get and sort importances
                importances = pd.Series(stored_model.feature_importances_, index=feature_names)
                importances_sorted = importances.sort_values(ascending=False)
                
                # Convert to Python types for JSON serialization
                feature_importances = {
                    str(k): float(v) 
                    for k, v in importances_sorted.items()
                }
            
            # Get per-class metrics if data is available
            class_report = {}
            if all(x is not None for x in [y_test, y_pred]):
                class_report = {
                    "0": {
                        "precision": float(precision_score(y_test, y_pred, pos_label=0)),
                        "recall": float(recall_score(y_test, y_pred, pos_label=0)),
                        "f1": float(f1_score(y_test, y_pred, pos_label=0))
                    },
                    "1": {
                        "precision": float(precision_score(y_test, y_pred, pos_label=1)),
                        "recall": float(recall_score(y_test, y_pred, pos_label=1)),
                        "f1": float(f1_score(y_test, y_pred, pos_label=1))
                    }
                }
            
            # Create comprehensive report with safeguards
            report = {
                'generated_at': datetime.now().isoformat(),
                'model_info': {
                    'type': type(model).__name__ if model is not None else "Unknown",
                    'parameters': model.get_params() if (model is not None and hasattr(model, 'get_params')) else {},
                },
                'data_stats': {
                    'train_samples': len(X_test) if X_test is not None else 0,
                    'features_count': X_test.shape[1] if X_test is not None else 0,
                    'class_distribution': {
                        '0': int((y_test == 0).sum()) if y_test is not None else 0,
                        '1': int((y_test == 1).sum()) if y_test is not None else 0
                    },
                    'feature_names': feature_names
                },
                'metrics': {
                    'overall': metrics,
                    'per_class': class_report
                },
                'feature_importance': feature_importances,
                'plots': self.results.get('plots', {}),
                'time_analysis': time_analysis,
                'evaluation_settings': {
                    'visualization_sample_size': len(X_test_viz),
                    'threshold': 0.5  # Default classification threshold
                }
            }
            
            report_path = os.path.join(self.output_dir, 'performance_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            # Generate PDF report
            pdf_path = os.path.join(self.output_dir, 'evaluation_report.pdf')
            with PdfPages(pdf_path) as pdf:
                # Create title page
                plt.figure(figsize=(8.5, 11))
                plt.axis('off')
                plt.text(0.5, 0.8, 'Model Evaluation Report', 
                        horizontalalignment='center', fontsize=20)
                plt.text(0.5, 0.7, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                        horizontalalignment='center')
                plt.text(0.5, 0.6, f'Model Type: {self.config.get("model_type", "Not specified")}',
                        horizontalalignment='center')
                pdf.savefig()
                plt.close()
                
                # Add metrics summary
                plt.figure(figsize=(8.5, 11))
                plt.axis('off')
                plt.text(0.5, 0.95, 'Performance Metrics', horizontalalignment='center', fontsize=16)
                metrics = self.results.get('metrics', {})
                y_pos = 0.85
                for metric, value in metrics.items():
                    plt.text(0.5, y_pos, f'{metric.replace("_", " ").title()}: {value:.4f}',
                            horizontalalignment='center')
                    y_pos -= 0.05
                pdf.savefig()
                plt.close()
                
                # Add plots in groups of 6 (2x3 grid)
                plot_paths = list(self.results.get('plots', {}).values())
                for i in range(0, len(plot_paths), 6):
                    fig = plt.figure(figsize=(17, 11))  # Landscape orientation for better fit
                    plt.subplots_adjust(wspace=0.3, hspace=0.3)
                    
                    # Add up to 6 plots on this page
                    for j, plot_path in enumerate(plot_paths[i:i+6]):
                        if os.path.exists(plot_path):
                            img = plt.imread(plot_path)
                            ax = plt.subplot(2, 3, j+1)  # 2 rows, 3 columns
                            ax.imshow(img)
                            ax.axis('off')
                            # Add plot title
                            plot_name = os.path.splitext(os.path.basename(plot_path))[0]
                            ax.set_title(plot_name.replace('_', ' ').title(), pad=10)
                    
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
            
            logger.info(f"Reports saved to {report_path} and {pdf_path}")
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
        max_samples: int = 5000
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
            Maximum number of samples to use (default=5000)
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
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            plot_path = os.path.join(self.plots_dir, 'learning_curve.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            
            if 'plots' not in self.results:
                self.results['plots'] = {}
            self.results['plots']['learning_curve'] = plot_path
            
            return True
            
        except Exception as e:
            logger.error(f"Learning curve creation failed: {str(e)}")
            return False
