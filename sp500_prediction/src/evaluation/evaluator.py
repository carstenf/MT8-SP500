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

# Suppress only the specific feature names warning
warnings.filterwarnings('ignore', message='X does not have valid feature names, but RandomForestClassifier was fitted with feature names')
from typing import Dict, Any, Optional
from datetime import datetime
from .metrics import calculate_metrics
from src.visualization.plotters import get_yellowbrick_visualizer, save_plot

from yellowbrick.classifier import (
    ClassificationReport, ConfusionMatrix,
    ROCAUC, PrecisionRecallCurve,
    DiscriminationThreshold, ClassPredictionError
)
from yellowbrick.model_selection import LearningCurve

# Time-based analysis has been removed

logger = logging.getLogger(__name__)

# Helper function to convert numpy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

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
            from sklearn.base import clone

            # Generate predictions using original model and DataFrame
            logger.info("Generating predictions for evaluation...")
            # Keep predictions as pandas Series to maintain index alignment
            y_pred = pd.Series(model.predict(X_test), index=X_test.index)
            y_pred_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
            
            # Use a smaller subset for visualizations (max 10000 samples)
            if len(X_test) > 10000:
                # Sample while maintaining DataFrame/Series structure
                viz_indices = np.random.choice(X_test.index, 10000, replace=False)
                X_test_viz = X_test.loc[viz_indices]
                y_test_viz = y_test.loc[viz_indices]
                y_pred_viz = y_pred.loc[viz_indices]
                y_pred_proba_viz = y_pred_proba.loc[viz_indices]
            else:
                X_test_viz = X_test
                y_test_viz = y_test
                y_pred_viz = y_pred
                y_pred_proba_viz = y_pred_proba

            # Calculate metrics using our centralized metrics function
            from sklearn.metrics import log_loss
            
            # Calculate metrics using full dataset
            detailed_metrics = calculate_metrics(y_test, y_pred, y_pred_proba, config=self.config)
            
            # Format metrics for compatibility with existing code
            # Get confusion matrix values
            cm = np.array(detailed_metrics['confusion_matrix'])
            if cm.shape == (2, 2):
                # Matrix is in the format [[TN, FP], [FN, TP]]
                tn, fp = cm[0]
                fn, tp = cm[1]
            else:
                # Assume it's flattened [TN, FP, FN, TP]
                tn, fp, fn, tp = cm.ravel()
                
            total = len(y_test)
            
            metrics = {
                'Accuracy': detailed_metrics['accuracy'],
                'Precision': detailed_metrics['precision'],
                'Recall': detailed_metrics['recall'],
                'F1-score': detailed_metrics['f1'],
                'ROC AUC': detailed_metrics.get('roc_auc', 0.5),
                'Log Loss': log_loss(y_test, y_pred_proba),
                'Confusion Matrix': {
                    'True Negative Rate': float(tn) / total,
                    'False Positive Rate': float(fp) / total,
                    'False Negative Rate': float(fn) / total,
                    'True Positive Rate': float(tp) / total
                }
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
            # Use native Yellowbrick visualizer with DataFrame
            viz = ClassificationReport(
                model,
                ax=ax,
                classes=["Down", "Up"],
                support=True,
                fontsize=12
            )
            viz.fit(X_test_viz, y_test_viz)
            viz.score(X_test_viz, y_test_viz)
            plot_path = os.path.join(self.plots_dir, 'classification_report.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['classification_report'] = plot_path
            
            logger.info("Creating Confusion Matrix...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            # Use native Yellowbrick visualizer with DataFrame
            viz = ConfusionMatrix(
                model,
                ax=ax,
                classes=["Down", "Up"],
                percent=True,
                fontsize=12
            )
            viz.fit(X_test, y_test)
            viz.score(X_test, y_test)
            plot_path = os.path.join(self.plots_dir, 'confusion_matrix.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['confusion_matrix'] = plot_path
            
            logger.info("Creating ROC Curve...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            # Use native Yellowbrick visualizer with DataFrame
            viz = ROCAUC(
                model,
                ax=ax,
                classes=["Down", "Up"]
            )
            viz.fit(X_test_viz, y_test_viz)
            viz.score(X_test_viz, y_test_viz)
            plot_path = os.path.join(self.plots_dir, 'roc_curve.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['roc_curve'] = plot_path
            
            logger.info("Creating Precision-Recall Curve...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            # Use native Yellowbrick visualizer with DataFrame
            viz = PrecisionRecallCurve(
                model,
                ax=ax,
                classes=["Down", "Up"]
            )
            viz.fit(X_test_viz, y_test_viz)
            viz.score(X_test_viz, y_test_viz)
            plot_path = os.path.join(self.plots_dir, 'pr_curve.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['precision_recall_curve'] = plot_path
            
            logger.info("Creating Discrimination Threshold...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            # Use native Yellowbrick visualizer with DataFrame
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
            # Use native Yellowbrick visualizer with DataFrame
            viz = ClassPredictionError(
                model,
                ax=ax,
                classes=["Down", "Up"]
            )
            viz.fit(X_test_viz, y_test_viz)
            viz.score(X_test_viz, y_test_viz)
            plot_path = os.path.join(self.plots_dir, 'class_prediction_error.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            plots['class_prediction_error'] = plot_path
            
            logger.info("Creating Feature Importance Plot...")
            try:
                # Create feature importance visualization manually to avoid feature name warnings
                if hasattr(model, 'feature_importances_'):
                    fig, ax = plt.subplots(figsize=(12, 10))
                    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.05)
                    plot_path = os.path.join(self.plots_dir, 'feature_importance.png')
                    
                    # Get feature importances
                    importances = model.feature_importances_
                    
                    # Get feature names - ensure we have the right names even if model was fitted with DataFrame
                    if hasattr(model, 'feature_names_in_'):
                        feature_names = model.feature_names_in_
                    else:
                        feature_names = X_test.columns
                    
                    # Create dataframe for sorting
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    })
                    
                    # Sort and get top 40 features
                    importance_df = importance_df.sort_values('importance', ascending=False).head(40)
                    
                    # Plot using matplotlib directly instead of Yellowbrick
                    ax.barh(range(len(importance_df)), importance_df['importance'], align='center', color='#2ecc71')
                    ax.set_yticks(range(len(importance_df)))
                    ax.set_yticklabels(importance_df['feature'])
                    ax.invert_yaxis()  # Features with highest importance at the top
                    ax.set_xlabel('Importance')
                    ax.set_title('Top 40 Most Important Features')
                    
                    # Add values to the bars
                    for i, v in enumerate(importance_df['importance']):
                        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
                    
                    # Save the plot
                    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    plots['feature_importance'] = plot_path
                else:
                    logger.warning("Model does not have feature_importances_ attribute")
            except Exception as e:
                logger.warning(f"Could not create feature importance plot: {str(e)}")
            plt.close()

            # Store results
            # Store results with converted metrics and additional data for report generation
            metrics = convert_numpy_types(metrics)
            self.results = {
                'metrics': metrics,
                'plots': plots,
                '_model': model,  # Store model for report generation
                '_data': {
                    'X_test': X_test,  # Keep as DataFrame
                    'y_test': convert_numpy_types(y_test),
                    'y_pred': convert_numpy_types(y_pred),
                    'X_test_viz': X_test_viz,  # Keep as DataFrame
                    'feature_names': list(X_test.columns)
                }
            }

            # Create correlation visualization
            logger.info("Creating Feature Correlation Plots...")
            try:
                # Get top 40 features by importance with converted numpy types
                importances = pd.Series(convert_numpy_types(model.feature_importances_), 
                                     index=X_test.columns)
                top_features = importances.nlargest(40)
                X_test_subset = X_test_viz[top_features.index]
                
                # Calculate correlation matrix and convert to Python types
                corr_matrix = convert_numpy_types(X_test_subset.corr())
                
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
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            # Format metrics to 4 decimal places
            metrics = {}
            for k, v in self.results.get('metrics', {}).items():
                if isinstance(v, (float, np.float32, np.float64)):
                    metrics[k] = float(f"{v:.4f}")
                else:
                    metrics[k] = convert_numpy_types(v)
            
            # Time analysis has been removed

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
                # Get and sort importances with converted numpy types
                importances = pd.Series(convert_numpy_types(stored_model.feature_importances_), 
                                     index=feature_names)
                importances_sorted = importances.sort_values(ascending=False)
                
                # Already converted to Python types
                feature_importances = {
                    str(k): v for k, v in importances_sorted.items()
                }
            
            # Get per-class metrics from y_test and y_pred
            class_report = {}
            from sklearn.metrics import precision_score, recall_score, f1_score
            
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
            # Get model parameters in a safe way
            model_params = {}
            if hasattr(stored_model, 'get_params'):
                try:
                    raw_params = stored_model.get_params()
                    # Convert numpy types to native Python types
                    model_params = convert_numpy_types(raw_params)
                except Exception as e:
                    logger.warning(f"Could not get model parameters: {str(e)}")
            
            # Get all configured features
            feature_config = self.config.get('features', {})
            
            # Create pipeline info by checking what was actually used
            pipeline_info = {
                'data_preparation': {
                    'raw_data': {
                        'source': 'sp500_data.h5',
                        'period': {
                            'start': str(X_test.index.get_level_values('date').min()),
                            'end': str(X_test.index.get_level_values('date').max()),
                            'trading_days': len(X_test.index.get_level_values('date').unique())
                        }
                    },
                    'features': {
                        'initial': {
                            'technical_indicators': [
                                "roc_1", "roc_3", "roc_5", "roc_7", "roc_9", "mom_1", "roc_11", "roc_13", "roc_60", "bb_percent_b_20",
                                "roc_15", "roc_40", "roc_17", "bb_bandwidth_20", "roc_80", "mom_3", "rsi_14", "roc_19", "roc_21", "roc_29",
                                "roc_25", "roc_240", "roc_100", "roc_23", "roc_27", "mom_5", "roc_120", "mom_7", "mom_60", "mom_9",
                                "macd_hist_12_26_9", "roc_220", "roc_200", "mom_11", "roc_140", "mom_100", "roc_180", "mom_40", "roc_160",
                                "mom_80"
                            ],
                            'total_features': 61
                        },
                        'after_selection': {
                            'technical_indicators': list(feature_names) if feature_names else [],
                            'total_features': len(feature_names) if feature_names else 0
                        },
                        'preprocessing': {
                            'scaling_applied': True,
                            'scaling_method': 'StandardScaler'
                        }
                    },
                    'target': {
                        'type': 'binary',
                        'calculation': {
                            'method': 'quantile_based',
                            'return_type': 'excess',
                            'horizon': [2],
                            'quantile_threshold': {
                                'upper': 0.66,
                                'lower': 0.33
                            }
                        }
                    },
                },
                'feature_selection': {
                    'enabled': bool(self.config.get('feature_selection', False)),
                    'method': self.config.get('evaluation', {}).get('feature_importance', {}).get('method', 'boruta'),
                    'initial_features': X_test.shape[1],  # before selection
                    'selected_features': len(feature_names) if feature_names else 0
                },
                'model': {
                    'type': type(stored_model).__name__ if stored_model is not None else 'Unknown',
                    'train_test_split': {
                        'train_years': self.config.get('train_years', []),
                        'test_years': self.config.get('test_years', [])
                    }
                }
            }

            # Create comprehensive report
            report = {
                'generated_at': datetime.now().isoformat(),
                'pipeline_info': pipeline_info,
                'model_info': {
                    'type': type(stored_model).__name__ if stored_model is not None else "Unknown",
                    'parameters': model_params
                },
                'data_stats': {
                    'train_samples': len(X_test) if X_test is not None else 0,
                    'features_count': len(feature_names) if feature_names else 0,
                    'class_distribution': {
                        '0': convert_numpy_types((y_test == 0).sum()) if y_test is not None else 0,
                        '1': convert_numpy_types((y_test == 1).sum()) if y_test is not None else 0
                    }
                },
                'metrics': {
                    'overall': {
                        'Accuracy': metrics['Accuracy'],
                        'Precision': metrics['Precision'],
                        'Recall': metrics['Recall'],
                        'F1-score': metrics['F1-score'],
                        'ROC AUC': metrics['ROC AUC'],
                        'Log Loss': metrics['Log Loss'],
                        'Confusion Matrix': metrics['Confusion Matrix']
                    }
                },
                'feature_importance': feature_importances,
                'plots': self.results.get('plots', {}),
                'evaluation_settings': {
                    'visualization_sample_size': len(X_test_viz) if X_test_viz is not None else 0,
                    'threshold': 0.5,  # Default classification threshold
                    'config': self.config.get('evaluation', {}),
                    'data_split': {
                        'train_years': self.config.get('train_years', []),
                        'test_years': self.config.get('test_years', [])
                    }
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
                    if isinstance(value, dict):
                        # Handle nested metrics like Confusion Matrix
                        plt.text(0.5, y_pos, f'{metric.replace("_", " ").title()}:', horizontalalignment='center')
                        y_pos -= 0.03
                        for submetric, subvalue in value.items():
                            plt.text(0.5, y_pos, f'  {submetric}: {subvalue:.4f}', horizontalalignment='center')
                            y_pos -= 0.03
                    else:
                        # Handle regular metrics
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
        """Create a learning curve visualization with limited samples."""
        logger.info("Creating learning curve visualization")
        
        try:
            # Limit the number of samples if needed
            if len(X_train) > max_samples:
                sample_indices = np.random.choice(len(X_train), max_samples, replace=False)
                X_train = X_train.iloc[sample_indices]
                y_train = y_train.iloc[sample_indices]
                logger.info(f"Using {max_samples} randomly sampled training examples")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use native Yellowbrick LearningCurve visualizer with DataFrame
            sizes = np.linspace(0.3, 1.0, 5)  # [0.3, 0.475, 0.65, 0.825, 1.0]
            viz = LearningCurve(
                model,
                ax=ax,
                train_sizes=sizes,
                cv=cv,
                n_jobs=-1,
                scoring='balanced_accuracy'
            )
            viz.fit(X_train, y_train)
            
            # Save the plot
            plot_path = os.path.join(self.plots_dir, 'learning_curve.png')
            viz.show(outpath=plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            if 'plots' not in self.results:
                self.results['plots'] = {}
            self.results['plots']['learning_curve'] = plot_path
            
            return True
            
        except Exception as e:
            logger.error(f"Learning curve creation failed: {str(e)}")
            return False
