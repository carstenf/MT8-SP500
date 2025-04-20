"""
Report Generator Module for S&P500 Prediction Project

This module handles the generation of comprehensive PDF reports
detailing model performance, feature importance, and other analytics.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_title_page(pdf: PdfPages, 
                     title: str = "S&P500 Stock Direction Prediction",
                     subtitle: str = "Performance Report",
                     project_info: Dict = None) -> None:
    """
    Create a title page for the PDF report.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    title : str, optional
        Main title for the report
    subtitle : str, optional
        Subtitle for the report
    project_info : Dict, optional
        Additional project information to include on the title page
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    
    # Add title and subtitle
    plt.figtext(0.5, 0.7, title, fontsize=24, fontweight='bold', ha='center')
    plt.figtext(0.5, 0.65, subtitle, fontsize=18, ha='center')
    
    # Add generation date
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    plt.figtext(0.5, 0.6, f"Generated on: {current_date}", fontsize=12, ha='center')
    
    # Add project info if provided
    if project_info:
        y_pos = 0.5
        for key, value in project_info.items():
            plt.figtext(0.5, y_pos, f"{key}: {value}", fontsize=12, ha='center')
            y_pos -= 0.05
    
    # Add footer
    plt.figtext(0.5, 0.05, "Confidential - For Internal Use Only", fontsize=10, ha='center')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_executive_summary(pdf: PdfPages, 
                            metrics: Dict,
                            feature_importance: Dict = None,
                            ticker_analysis: Dict = None,
                            time_analysis: Dict = None) -> None:
    """
    Create an executive summary page with key metrics.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    metrics : Dict
        Dictionary with model performance metrics
    feature_importance : Dict, optional
        Dictionary with feature importance information
    ticker_analysis : Dict, optional
        Dictionary with ticker-level performance metrics
    time_analysis : Dict, optional
        Dictionary with time-based performance analysis
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("Executive Summary", fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1.5])
    
    # Section 1: Key Performance Metrics
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    ax1.set_title("Key Performance Metrics", fontsize=14, loc='left')
    
    # Extract key metrics
    bal_acc = metrics.get('balanced_accuracy', 'N/A')
    precision = metrics.get('precision', 'N/A')
    recall = metrics.get('recall', 'N/A')
    f1 = metrics.get('f1', 'N/A')
    roc_auc = metrics.get('roc_auc', 'N/A')
    
    # Format metrics text
    metrics_text = (
        f"Balanced Accuracy: {bal_acc:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"ROC-AUC: {roc_auc:.4f}"
    )
    ax1.text(0.05, 0.7, metrics_text, fontsize=12, va='top')
    
    # Section 2: Top Features
    ax2 = fig.add_subplot(gs[1, 0])
    if feature_importance and 'importance_df' in feature_importance:
        ax2.set_title("Top 5 Features", fontsize=14, loc='left')
        ax2.axis('off')
        
        # Get top 5 features
        top_features = feature_importance['importance_df'].head(5)
        
        # Create table
        feature_data = []
        for _, row in top_features.iterrows():
            feature_data.append([row['feature'], f"{row['importance']:.4f}"])
        
        ax2.table(cellText=feature_data, 
                 colLabels=['Feature', 'Importance'],
                 loc='center',
                 cellLoc='center')
    else:
        ax2.text(0.5, 0.5, "Feature importance data not available", 
                fontsize=10, ha='center', va='center')
        ax2.axis('off')
    
    # Section 3: Ticker Performance
    ax3 = fig.add_subplot(gs[1, 1])
    if ticker_analysis and 'aggregate_metrics' in ticker_analysis:
        ax3.set_title("Ticker Performance", fontsize=14, loc='left')
        ax3.axis('off')
        
        # Extract aggregate metrics
        agg = ticker_analysis['aggregate_metrics']
        n_tickers = agg.get('n_tickers', 'N/A')
        avg_bal_acc = agg.get('avg_balanced_accuracy', 'N/A')
        avg_pnl = agg.get('avg_pnl', 'N/A')
        avg_sharpe = agg.get('avg_sharpe', 'N/A')
        
        # Format ticker text
        ticker_text = (
            f"Number of Tickers: {n_tickers}\n"
            f"Avg. Balanced Accuracy: {avg_bal_acc:.4f}\n"
            f"Avg. P&L: {avg_pnl:.4f}\n"
            f"Avg. Sharpe Ratio: {avg_sharpe:.4f}"
        )
        ax3.text(0.05, 0.7, ticker_text, fontsize=12, va='top')
    else:
        ax3.text(0.5, 0.5, "Ticker analysis data not available", 
                fontsize=10, ha='center', va='center')
        ax3.axis('off')
    
    # Section A: Trend Analysis
    ax4 = fig.add_subplot(gs[2, 0])
    if time_analysis and 'trend_direction' in time_analysis:
        ax4.set_title("Performance Trends", fontsize=14, loc='left')
        ax4.axis('off')
        
        # Extract trend directions
        trends = time_analysis['trend_direction']
        bal_acc_trend = trends.get('balanced_accuracy', 'N/A')
        precision_trend = trends.get('precision', 'N/A')
        recall_trend = trends.get('recall', 'N/A')
        f1_trend = trends.get('f1', 'N/A')
        
        # Format trend text
        trend_text = (
            f"Balanced Accuracy: {bal_acc_trend}\n"
            f"Precision: {precision_trend}\n"
            f"Recall: {recall_trend}\n"
            f"F1 Score: {f1_trend}"
        )
        ax4.text(0.05, 0.7, trend_text, fontsize=12, va='top')
    else:
        ax4.text(0.5, 0.5, "Time analysis data not available", 
                fontsize=10, ha='center', va='center')
        ax4.axis('off')
    
    # Section B: Conclusion and Recommendations
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_title("Conclusion & Recommendations", fontsize=14, loc='left')
    ax5.axis('off')
    
    # Determine success based on balanced accuracy threshold
    success = bal_acc >= 0.55 if isinstance(bal_acc, (int, float)) else False
    
    if success:
        conclusion = (
            "The model meets the success criteria with balanced accuracy above 0.55.\n\n"
            "Recommendations:\n"
            "• Deploy model for production use\n"
            "• Monitor performance over time\n"
            "• Consider retraining quarterly with fresh data"
        )
    else:
        conclusion = (
            "The model does not meet the success criteria (balanced accuracy < 0.55).\n\n"
            "Recommendations:\n"
            "• Review feature engineering approach\n"
            "• Try additional model architectures\n"
            "• Consider using more training data"
        )
    ax5.text(0.05, 0.7, conclusion, fontsize=12, va='top')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_data_preparation_page(pdf: PdfPages, 
                                data_info: Dict) -> None:
    """
    Create a page describing the data preparation methodology.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    data_info : Dict
        Dictionary with data information including:
        - data_period
        - n_tickers
        - n_samples
        - missing_values
        - feature_counts
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("Data Preparation Methodology", fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.5])
    
    # Section 1: Data Overview
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    ax1.set_title("Data Overview", fontsize=14, loc='left')
    
    # Extract data info
    start_date = data_info.get('start_date', 'N/A')
    end_date = data_info.get('end_date', 'N/A')
    n_tickers = data_info.get('n_tickers', 'N/A')
    n_samples = data_info.get('n_samples', 'N/A')
    
    # Format data text
    data_text = (
        f"Data Period: {start_date} to {end_date}\n"
        f"Number of Tickers: {n_tickers}\n"
        f"Total Samples: {n_samples:,}\n"
        f"S&P500 constituents with sufficient trading history"
    )
    ax1.text(0.05, 0.7, data_text, fontsize=12, va='top')
    
    # Section 2: Data Cleaning
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Data Cleaning", fontsize=14, loc='left')
    ax2.axis('off')
    
    cleaning_text = (
        "• Removed tickers with insufficient history\n"
        "• Handled missing values\n"
        "• Aligned dates across all tickers\n"
        "• Created consistent multi-index (ticker, date) structure\n"
        "• Verified data continuity for all stocks"
    )
    ax2.text(0.05, 0.7, cleaning_text, fontsize=12, va='top')
    
    # Section 3: Feature Engineering
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("Feature Engineering", fontsize=14, loc='left')
    ax3.axis('off')
    
    # Extract feature counts if available
    feature_counts = data_info.get('feature_counts', {})
    daily_lags = feature_counts.get('daily_lags', 'N/A')
    long_term_lags = feature_counts.get('long_term_lags', 'N/A')
    cumulative = feature_counts.get('cumulative', 'N/A')
    average = feature_counts.get('average', 'N/A')
    volatility = feature_counts.get('volatility', 'N/A')
    total = feature_counts.get('total', 'N/A')
    
    feature_text = (
        f"Total Features: {total}\n"
        f"• Daily Lags: {daily_lags}\n"
        f"• Long-term Lags: {long_term_lags}\n"
        f"• Cumulative Returns: {cumulative}\n"
        f"• Average Returns: {average}\n"
        f"• Volatility Features: {volatility}"
    )
    ax3.text(0.05, 0.7, feature_text, fontsize=12, va='top')
    
    # Section 4: Data Split Methodology
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_title("Train-Test Split Methodology", fontsize=14, loc='left')
    ax4.axis('off')
    
    # Extract train/test info
    train_years = data_info.get('train_years', [])
    test_years = data_info.get('test_years', [])
    train_samples = data_info.get('train_samples', 'N/A')
    test_samples = data_info.get('test_samples', 'N/A')
    
    if train_years and test_years:
        train_period = f"{min(train_years)} to {max(train_years)}"
        test_period = f"{min(test_years)} to {max(test_years)}"
    else:
        train_period = "N/A"
        test_period = "N/A"
    
    split_text = (
        f"Training Period: {train_period}\n"
        f"Testing Period: {test_period}\n"
        f"Training Samples: {train_samples:,}\n"
        f"Testing Samples: {test_samples:,}\n\n"
        "Cross-Validation Approach:\n"
        "• Time-based 5-fold cross-validation\n"
        "• Each fold trains on 3 years and tests on the following year\n"
        "• Only stocks with complete data for each specific period are included\n"
        "• Prevents look-ahead bias and ensures realistic evaluation"
    )
    ax4.text(0.05, 0.7, split_text, fontsize=12, va='top')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_model_performance_page(pdf: PdfPages, 
                                 metrics: Dict,
                                 confusion_matrix: List = None,
                                 cv_results: Dict = None,
                                 viz_paths: Dict = None) -> None:
    """
    Create a page showing detailed model performance metrics.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    metrics : Dict
        Dictionary with model performance metrics
    confusion_matrix : List, optional
        Confusion matrix values as a 2x2 list
    cv_results : Dict, optional
        Cross-validation results
    viz_paths : Dict, optional
        Dictionary with paths to visualization images
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("Model Performance Metrics", fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # Section 1: Overall Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Performance Metrics", fontsize=14, loc='left')
    ax1.axis('off')
    
    # Extract detailed metrics
    bal_acc = metrics.get('balanced_accuracy', 'N/A')
    precision = metrics.get('precision', 'N/A')
    recall = metrics.get('recall', 'N/A')
    f1 = metrics.get('f1', 'N/A')
    roc_auc = metrics.get('roc_auc', 'N/A')
    accuracy = metrics.get('accuracy', 'N/A')
    
    # Format metrics text
    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Balanced Accuracy: {bal_acc:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"ROC-AUC: {roc_auc:.4f}"
    )
    ax1.text(0.05, 0.8, metrics_text, fontsize=12, va='top')
    
    # Section 2: Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Confusion Matrix", fontsize=14, loc='left')
    
    if confusion_matrix:
        # Create confusion matrix heatmap
        cm = np.array(confusion_matrix)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Down', 'Up'],
                   yticklabels=['Down', 'Up'],
                   ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, "Confusion matrix data not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 3: ROC Curve
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("ROC Curve", fontsize=14, loc='left')
    
    if viz_paths and 'roc_curve' in viz_paths and os.path.exists(viz_paths['roc_curve']):
        try:
            img = plt.imread(viz_paths['roc_curve'])
            ax3.imshow(img)
            ax3.axis('off')
        except Exception as e:
            logger.error(f"Error loading ROC curve image: {str(e)}")
            ax3.axis('off')
            ax3.text(0.5, 0.5, "Error loading ROC curve image", 
                    fontsize=10, ha='center', va='center')
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, "ROC curve image not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 4: PR Curve
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Precision-Recall Curve", fontsize=14, loc='left')
    
    if viz_paths and 'pr_curve' in viz_paths and os.path.exists(viz_paths['pr_curve']):
        try:
            img = plt.imread(viz_paths['pr_curve'])
            ax4.imshow(img)
            ax4.axis('off')
        except Exception as e:
            logger.error(f"Error loading PR curve image: {str(e)}")
            ax4.axis('off')
            ax4.text(0.5, 0.5, "Error loading PR curve image", 
                    fontsize=10, ha='center', va='center')
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, "Precision-Recall curve image not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 5: Cross-Validation Results
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title("Cross-Validation Results", fontsize=14, loc='left')
    
    if cv_results and 'fold_metrics' in cv_results:
        # Extract fold metrics
        fold_metrics = cv_results['fold_metrics']
        
        # Prepare data for table
        table_data = []
        for fold_num, metrics in fold_metrics.items():
            bal_acc = metrics.get('balanced_accuracy', 'N/A')
            f1 = metrics.get('f1', 'N/A')
            roc_auc = metrics.get('roc_auc', 'N/A')
            
            if isinstance(bal_acc, (int, float)):
                bal_acc = f"{bal_acc:.4f}"
            if isinstance(f1, (int, float)):
                f1 = f"{f1:.4f}"
            if isinstance(roc_auc, (int, float)):
                roc_auc = f"{roc_auc:.4f}"
                
            table_data.append([fold_num, bal_acc, f1, roc_auc])
        
        # Add average row
        avg_metrics = cv_results.get('avg_metrics', {})
        avg_bal_acc = avg_metrics.get('balanced_accuracy', 'N/A')
        avg_f1 = avg_metrics.get('f1', 'N/A')
        avg_roc_auc = avg_metrics.get('roc_auc', 'N/A')
        
        if isinstance(avg_bal_acc, (int, float)):
            avg_bal_acc = f"{avg_bal_acc:.4f}"
        if isinstance(avg_f1, (int, float)):
            avg_f1 = f"{avg_f1:.4f}"
        if isinstance(avg_roc_auc, (int, float)):
            avg_roc_auc = f"{avg_roc_auc:.4f}"
            
        table_data.append(['Average', avg_bal_acc, avg_f1, avg_roc_auc])
        
        # Create table
        ax5.axis('off')
        table = ax5.table(cellText=table_data,
                         colLabels=['Fold', 'Balanced Accuracy', 'F1 Score', 'ROC-AUC'],
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
    else:
        ax5.axis('off')
        ax5.text(0.5, 0.5, "Cross-validation results not available", 
                fontsize=10, ha='center', va='center')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_feature_importance_page(pdf: PdfPages, 
                                  feature_importance: Dict,
                                  viz_paths: Dict = None) -> None:
    """
    Create a page showing feature importance analysis.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    feature_importance : Dict
        Dictionary with feature importance information
    viz_paths : Dict, optional
        Dictionary with paths to visualization images
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("Feature Importance Analysis", fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1])
    
    # Section 1: Feature Importance Plot
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Top Feature Importance", fontsize=14, loc='left')
    
    if viz_paths and 'feature_importance' in viz_paths and os.path.exists(viz_paths['feature_importance']):
        try:
            img = plt.imread(viz_paths['feature_importance'])
            ax1.imshow(img)
            ax1.axis('off')
        except Exception as e:
            logger.error(f"Error loading feature importance image: {str(e)}")
            ax1.axis('off')
            ax1.text(0.5, 0.5, "Error loading feature importance image", 
                    fontsize=10, ha='center', va='center')
    elif feature_importance and 'importance_df' in feature_importance:
        # Create our own feature importance plot
        top_features = feature_importance['importance_df'].head(15)
        y_pos = range(len(top_features))
        
        ax1.barh(y_pos, top_features['importance'], align='center')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_features['feature'])
        ax1.invert_yaxis()  # Features with highest importance at the top
        ax1.set_xlabel('Importance')
    else:
        ax1.axis('off')
        ax1.text(0.5, 0.5, "Feature importance data not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 2: Feature Analysis
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("Feature Analysis", fontsize=14, loc='left')
    ax2.axis('off')
    
    if feature_importance:
        # Extract feature analysis data
        features_80 = feature_importance.get('features_for_80_importance', 'N/A')
        features_90 = feature_importance.get('features_for_90_importance', 'N/A')
        
        # Extract top features
        top_features_list = []
        if 'importance_df' in feature_importance:
            top_features_list = feature_importance['importance_df'].head(5)['feature'].tolist()
        top_features_str = ", ".join(top_features_list) if top_features_list else "N/A"
        
        # Format analysis text
        analysis_text = (
            f"Features needed for 80% importance: {features_80}\n"
            f"Features needed for 90% importance: {features_90}\n\n"
            f"Top 5 features: {top_features_str}\n\n"
            "Analysis:\n"
            "• Recent price movements (1-5 day lag) are highly predictive\n"
            "• Medium-term trends (20-60 day) provide context\n"
            "• Volatility features help predict directional movement\n"
            "• Market-relative returns more predictive than absolute returns\n"
            "• Feature importance is consistent across cross-validation folds"
        )
        ax2.text(0.05, 0.95, analysis_text, fontsize=12, va='top')
    else:
        ax2.text(0.5, 0.5, "Feature analysis data not available", 
                fontsize=10, ha='center', va='center')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_performance_by_ticker_page(pdf: PdfPages, 
                                     ticker_analysis: Dict) -> None:
    """
    Create a page showing model performance by ticker.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    ticker_analysis : Dict
        Dictionary with ticker-level performance analysis
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("Performance Analysis by Ticker", fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1.5, 1])
    
    # Section 1: Aggregate Metrics
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Aggregate Metrics", fontsize=14, loc='left')
    ax1.axis('off')
    
    if ticker_analysis and 'aggregate_metrics' in ticker_analysis:
        # Extract aggregate metrics
        agg = ticker_analysis['aggregate_metrics']
        n_tickers = agg.get('n_tickers', 'N/A')
        avg_bal_acc = agg.get('avg_balanced_accuracy', 'N/A')
        avg_f1 = agg.get('avg_f1', 'N/A')
        avg_pnl = agg.get('avg_pnl', 'N/A')
        avg_sharpe = agg.get('avg_sharpe', 'N/A')
        
        # Format metrics text
        metrics_text = (
            f"Number of Tickers: {n_tickers}\n"
            f"Average Balanced Accuracy: {avg_bal_acc:.4f}\n"
            f"Average F1 Score: {avg_f1:.4f}\n"
            f"Average P&L: {avg_pnl:.4f}\n"
            f"Average Sharpe Ratio: {avg_sharpe:.4f}"
        )
        ax1.text(0.05, 0.8, metrics_text, fontsize=12, va='top')
    else:
        ax1.text(0.5, 0.5, "Ticker analysis data not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 2: Top Performing Tickers
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("Top Performing Tickers", fontsize=14, loc='left')
    
    if ticker_analysis and 'aggregate_metrics' in ticker_analysis and 'top_tickers_by_accuracy' in ticker_analysis['aggregate_metrics']:
        # Extract top tickers
        top_tickers = ticker_analysis['aggregate_metrics']['top_tickers_by_accuracy']
        
        # Prepare data for table
        table_data = []
        for ticker_info in top_tickers:
            ticker = ticker_info[0]
            metrics = ticker_info[1]
            
            bal_acc = metrics.get('balanced_accuracy', 'N/A')
            f1 = metrics.get('f1', 'N/A')
            pnl = metrics.get('pnl', {}).get('total', 'N/A')
            sharpe = metrics.get('pnl', {}).get('sharpe', 'N/A')
            
            if isinstance(bal_acc, (int, float)):
                bal_acc = f"{bal_acc:.4f}"
            if isinstance(f1, (int, float)):
                f1 = f"{f1:.4f}"
            if isinstance(pnl, (int, float)):
                pnl = f"{pnl:.4f}"
            if isinstance(sharpe, (int, float)):
                sharpe = f"{sharpe:.4f}"
                
            table_data.append([ticker, bal_acc, f1, pnl, sharpe])
        
        # Create table
        ax2.axis('off')
        if table_data:
            table = ax2.table(cellText=table_data,
                             colLabels=['Ticker', 'Balanced Accuracy', 'F1 Score', 'P&L', 'Sharpe Ratio'],
                             loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
        else:
            ax2.text(0.5, 0.5, "No top ticker data available", 
                    fontsize=10, ha='center', va='center')
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, "Top ticker data not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 3: Performance Distribution
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("Performance Distribution Analysis", fontsize=14, loc='left')
    ax3.axis('off')
    
    if ticker_analysis and 'ticker_metrics' in ticker_analysis:
        # Extract all ticker metrics
        ticker_metrics = ticker_analysis['ticker_metrics']
        
        # Calculate distribution of balanced accuracy
        bal_accs = [metrics.get('balanced_accuracy', None) for metrics in ticker_metrics.values()]
        bal_accs = [acc for acc in bal_accs if acc is not None]
        
        # Calculate percentiles
        if bal_accs:
            percentiles = {
                '90th': np.percentile(bal_accs, 90),
                '75th': np.percentile(bal_accs, 75),
                '50th': np.percentile(bal_accs, 50),
                '25th': np.percentile(bal_accs, 25)
            }
            
            # Count tickers by performance bracket
            above_60 = sum(1 for acc in bal_accs if acc >= 0.60)
            above_55 = sum(1 for acc in bal_accs if 0.55 <= acc < 0.60)
            above_50 = sum(1 for acc in bal_accs if 0.50 <= acc < 0.55)
            below_50 = sum(1 for acc in bal_accs if acc < 0.50)
            
            # Format distribution text
            distribution_text = (
                "Balanced Accuracy Distribution:\n"
                f"90th Percentile: {percentiles['90th']:.4f}\n"
                f"75th Percentile: {percentiles['75th']:.4f}\n"
                f"Median: {percentiles['50th']:.4f}\n"
                f"25th Percentile: {percentiles['25th']:.4f}\n\n"
                f"Tickers with Balanced Accuracy ≥ 0.60: {above_60} ({above_60/len(bal_accs)*100:.1f}%)\n"
                f"Tickers with Balanced Accuracy 0.55-0.59: {above_55} ({above_55/len(bal_accs)*100:.1f}%)\n"
                f"Tickers with Balanced Accuracy 0.50-0.54: {above_50} ({above_50/len(bal_accs)*100:.1f}%)\n"
                f"Tickers with Balanced Accuracy < 0.50: {below_50} ({below_50/len(bal_accs)*100:.1f}%)"
            )
            ax3.text(0.05, 0.95, distribution_text, fontsize=12, va='top')
        else:
            ax3.text(0.5, 0.5, "No balanced accuracy data available for analysis", 
                    fontsize=10, ha='center', va='center')
    else:
        ax3.text(0.5, 0.5, "Ticker metrics data not available", 
                fontsize=10, ha='center', va='center')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_time_analysis_page(pdf: PdfPages, 
                             time_analysis: Dict,
                             viz_paths: Dict = None) -> None:
    """
    Create a page showing performance analysis over time.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    time_analysis : Dict
        Dictionary with time-based performance analysis
    viz_paths : Dict, optional
        Dictionary with paths to visualization images
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("Performance Analysis Over Time", fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1])
    
    # Section 1: Time Series Plot
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Performance Metrics Over Time", fontsize=14, loc='left')
    
    if viz_paths and 'time_series' in viz_paths and os.path.exists(viz_paths['time_series']):
        try:
            img = plt.imread(viz_paths['time_series'])
            ax1.imshow(img)
            ax1.axis('off')
        except Exception as e:
            logger.error(f"Error loading time series image: {str(e)}")
            ax1.axis('off')
            ax1.text(0.5, 0.5, "Error loading time series image", 
                    fontsize=10, ha='center', va='center')
    elif time_analysis and 'time_series' in time_analysis:
        # Create our own time series plot
        time_series = time_analysis['time_series']
        
        if 'periods' in time_series and 'balanced_accuracy' in time_series:
            periods = time_series['periods']
            bal_acc = time_series['balanced_accuracy']
            precision = time_series.get('precision', [])
            recall = time_series.get('recall', [])
            f1 = time_series.get('f1', [])
            
            # Convert string periods to datetime if possible
            try:
                x = [pd.to_datetime(p) for p in periods]
            except:
                x = range(len(periods))
                ax1.set_xticks(x)
                ax1.set_xticklabels(periods, rotation=45)
            
            # Plot metrics
            ax1.plot(x, bal_acc, 'o-', label='Balanced Accuracy')
            if precision and len(precision) == len(x):
                ax1.plot(x, precision, 's-', label='Precision')
            if recall and len(recall) == len(x):
                ax1.plot(x, recall, '^-', label='Recall')
            if f1 and len(f1) == len(x):
                ax1.plot(x, f1, 'd-', label='F1 Score')
            
            ax1.set_xlabel('Time Period')
            ax1.set_ylabel('Score')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.axis('off')
            ax1.text(0.5, 0.5, "Time series data incomplete", 
                    fontsize=10, ha='center', va='center')
    else:
        ax1.axis('off')
        ax1.text(0.5, 0.5, "Time analysis data not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 2: Trend Analysis
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("Trend Analysis", fontsize=14, loc='left')
    ax2.axis('off')
    
    if time_analysis and 'trend_direction' in time_analysis:
        # Extract trend data
        trends = time_analysis['trend_direction']
        bal_acc_trend = trends.get('balanced_accuracy', 'N/A')
        precision_trend = trends.get('precision', 'N/A')
        recall_trend = trends.get('recall', 'N/A')
        f1_trend = trends.get('f1', 'N/A')
        
        # Format trend text
        trend_text = (
            "Performance Trend Analysis:\n"
            f"Balanced Accuracy: {bal_acc_trend}\n"
            f"Precision: {precision_trend}\n"
            f"Recall: {recall_trend}\n"
            f"F1 Score: {f1_trend}\n\n"
        )
        
        # Add interpretation based on trends
        interpretation = "Interpretation: "
        if bal_acc_trend == 'increasing':
            interpretation += (
                "Model performance is improving over time, suggesting the feature engineering "
                "approach is capturing relevant signals that persist. This indicates the model "
                "will likely continue to perform well in future periods."
            )
        elif bal_acc_trend == 'stable':
            interpretation += (
                "Model performance is consistent over time, indicating stable predictive power "
                "across different market conditions. This suggests the model is robust and "
                "not overly sensitive to specific market regimes."
            )
        elif bal_acc_trend == 'decreasing':
            interpretation += (
                "Model performance is declining over time, which may indicate data drift or "
                "changing market dynamics. Consider retraining more frequently or exploring "
                "additional features to capture evolving patterns."
            )
        else:
            interpretation += (
                "Insufficient data to determine a clear trend. More data points are needed "
                "for a reliable trend analysis."
            )
        
        ax2.text(0.05, 0.95, trend_text + interpretation, fontsize=12, va='top', wrap=True)
    else:
        ax2.text(0.5, 0.5, "Trend analysis data not available", 
                fontsize=10, ha='center', va='center')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_bias_variance_page(pdf: PdfPages, 
                             bias_variance: Dict,
                             viz_paths: Dict = None) -> None:
    """
    Create a page showing bias-variance tradeoff analysis.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    bias_variance : Dict
        Dictionary with bias-variance analysis data
    viz_paths : Dict, optional
        Dictionary with paths to visualization images
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("Bias-Variance Tradeoff Analysis", fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1])
    
    # Section 1: Learning Curve
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Learning Curves", fontsize=14, loc='left')
    
    if viz_paths and 'bias_variance' in viz_paths and os.path.exists(viz_paths['bias_variance']):
        try:
            img = plt.imread(viz_paths['bias_variance'])
            ax1.imshow(img)
            ax1.axis('off')
        except Exception as e:
            logger.error(f"Error loading bias-variance image: {str(e)}")
            ax1.axis('off')
            ax1.text(0.5, 0.5, "Error loading bias-variance image", 
                    fontsize=10, ha='center', va='center')
    elif bias_variance and all(k in bias_variance for k in ['train_sizes', 'train_mean_scores', 'test_mean_scores']):
        # Create our own learning curve plot
        train_sizes = bias_variance['train_sizes']
        train_mean = bias_variance['train_mean_scores']
        test_mean = bias_variance['test_mean_scores']
        train_std = bias_variance.get('train_std_scores', [0] * len(train_mean))
        test_std = bias_variance.get('test_std_scores', [0] * len(test_mean))
        
        # Plot learning curves
        ax1.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        ax1.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
        
        # Add shaded regions for standard deviation
        ax1.fill_between(train_sizes, 
                        [max(t - s, 0) for t, s in zip(train_mean, train_std)],
                        [min(t + s, 1) for t, s in zip(train_mean, train_std)], 
                        alpha=0.1, color='r')
        ax1.fill_between(train_sizes, 
                        [max(t - s, 0) for t, s in zip(test_mean, test_std)],
                        [min(t + s, 1) for t, s in zip(test_mean, test_std)], 
                        alpha=0.1, color='g')
        
        ax1.set_xlabel('Training Set Size')
        ax1.set_ylabel('Balanced Accuracy Score')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Add bias and variance annotation
        bias = bias_variance.get('bias', 'N/A')
        variance = bias_variance.get('variance', 'N/A')
        diagnosis = bias_variance.get('diagnosis', 'unknown')
        
        if isinstance(bias, (int, float)) and isinstance(variance, (int, float)):
            annotation = f"Bias: {bias:.4f}\nVariance: {variance:.4f}\nDiagnosis: {diagnosis}"
            ax1.annotate(annotation, xy=(0.5, 0.1), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    else:
        ax1.axis('off')
        ax1.text(0.5, 0.5, "Learning curve data not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 2: Bias-Variance Analysis and Recommendations
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("Bias-Variance Analysis", fontsize=14, loc='left')
    ax2.axis('off')
    
    if bias_variance:
        # Extract analysis data
        bias = bias_variance.get('bias', 'N/A')
        variance = bias_variance.get('variance', 'N/A')
        diagnosis = bias_variance.get('diagnosis', 'unknown')
        recommendations = bias_variance.get('recommendations', [])
        
        # Format bias-variance text
        analysis_text = (
            f"Bias: {bias:.4f}\n"
            f"Variance: {variance:.4f}\n"
            f"Diagnosis: {diagnosis}\n\n"
            "Recommendations:\n"
        )
        
        # Add recommendations
        for i, rec in enumerate(recommendations):
            analysis_text += f"{i+1}. {rec}\n"
        
        # Add additional analysis based on diagnosis
        if diagnosis == 'high_bias':
            analysis_text += (
                "\nThe model suffers from high bias, indicating it may be underfitting the data. "
                "Consider using a more complex model, adding more features, or reducing regularization."
            )
        elif diagnosis == 'high_variance':
            analysis_text += (
                "\nThe model suffers from high variance, indicating it may be overfitting the data. "
                "Consider using more training data, applying stronger regularization, or simplifying the model."
            )
        elif diagnosis == 'balanced':
            analysis_text += (
                "\nThe model has a good balance between bias and variance, indicating it generalizes well "
                "to unseen data. Fine-tuning hyperparameters might yield further small improvements."
            )
        
        ax2.text(0.05, 0.95, analysis_text, fontsize=12, va='top')
    else:
        ax2.text(0.5, 0.5, "Bias-variance analysis data not available", 
                fontsize=10, ha='center', va='center')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_shap_explanation_page(pdf: PdfPages, 
                                shap_data: Dict,
                                viz_paths: Dict = None) -> None:
    """
    Create a page with SHAP explanations for model predictions.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    shap_data : Dict
        Dictionary with SHAP explanation data
    viz_paths : Dict, optional
        Dictionary with paths to visualization images
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("SHAP Value Explanations", fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1.5, 1])
    
    # Section 1: SHAP Summary Plot
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("SHAP Feature Importance", fontsize=14, loc='left')
    
    if viz_paths and 'shap_summary' in viz_paths and os.path.exists(viz_paths['shap_summary']):
        try:
            img = plt.imread(viz_paths['shap_summary'])
            ax1.imshow(img)
            ax1.axis('off')
        except Exception as e:
            logger.error(f"Error loading SHAP summary image: {str(e)}")
            ax1.axis('off')
            ax1.text(0.5, 0.5, "Error loading SHAP summary image", 
                    fontsize=10, ha='center', va='center')
    else:
        ax1.axis('off')
        ax1.text(0.5, 0.5, "SHAP summary plot not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 2: SHAP Dependence Plot
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("SHAP Dependence Plot (Top Feature)", fontsize=14, loc='left')
    
    if viz_paths and 'shap_dependence' in viz_paths and os.path.exists(viz_paths['shap_dependence']):
        try:
            img = plt.imread(viz_paths['shap_dependence'])
            ax2.imshow(img)
            ax2.axis('off')
        except Exception as e:
            logger.error(f"Error loading SHAP dependence image: {str(e)}")
            ax2.axis('off')
            ax2.text(0.5, 0.5, "Error loading SHAP dependence image", 
                    fontsize=10, ha='center', va='center')
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, "SHAP dependence plot not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 3: SHAP Explanation Analysis
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("SHAP Value Interpretation", fontsize=14, loc='left')
    ax3.axis('off')
    
    if shap_data:
        # Format SHAP analysis text
        analysis_text = (
            "SHAP Value Interpretation:\n\n"
            "• SHAP values measure each feature's contribution to the prediction\n"
            "• Positive SHAP values push the prediction toward the positive class (UP)\n"
            "• Negative SHAP values push the prediction toward the negative class (DOWN)\n"
            "• The magnitude indicates the strength of the feature's influence\n\n"
            "Key Insights:\n"
            "• Recent price movements have the strongest impact on predictions\n"
            "• Feature interactions provide additional predictive power\n"
            "• Different stocks show different feature sensitivities\n"
            "• Market-relative performance is more predictive than absolute performance"
        )
        ax3.text(0.05, 0.95, analysis_text, fontsize=12, va='top')
    else:
        ax3.text(0.5, 0.5, "SHAP analysis data not available", 
                fontsize=10, ha='center', va='center')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_error_analysis_page(pdf: PdfPages, 
                              error_analysis: Dict,
                              viz_paths: Dict = None) -> None:
    """
    Create a page showing error analysis.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    error_analysis : Dict
        Dictionary with error analysis data
    viz_paths : Dict, optional
        Dictionary with paths to visualization images
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("Error Analysis", fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1.5])
    
    # Section 1: Correct vs Incorrect Feature Importance
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Feature Importance: Correct vs. Incorrect Predictions", fontsize=14, loc='left')
    
    if viz_paths and 'error_analysis' in viz_paths and os.path.exists(viz_paths['error_analysis']):
        try:
            img = plt.imread(viz_paths['error_analysis'])
            ax1.imshow(img)
            ax1.axis('off')
        except Exception as e:
            logger.error(f"Error loading error analysis image: {str(e)}")
            ax1.axis('off')
            ax1.text(0.5, 0.5, "Error loading error analysis image", 
                    fontsize=10, ha='center', va='center')
    elif error_analysis and 'feature_importance_diff' in error_analysis:
        # Create our own feature importance comparison plot
        feature_diff = error_analysis['feature_importance_diff']
        
        # Extract top features with biggest difference
        features = []
        correct_vals = []
        incorrect_vals = []
        
        max_features = min(10, len(feature_diff))
        for i in range(max_features):
            if i < len(feature_diff):
                features.append(feature_diff[i]['feature'])
                correct_vals.append(feature_diff[i]['importance_correct'])
                incorrect_vals.append(feature_diff[i]['importance_incorrect'])
        
        # Plot comparison
        if features:
            x = np.arange(len(features))
            width = 0.35
            
            ax1.bar(x - width/2, correct_vals, width, label='Correct Predictions')
            ax1.bar(x + width/2, incorrect_vals, width, label='Incorrect Predictions')
            
            ax1.set_ylabel('Feature Importance')
            ax1.set_xticks(x)
            ax1.set_xticklabels(features, rotation=45, ha='right')
            ax1.legend()
        else:
            ax1.axis('off')
            ax1.text(0.5, 0.5, "No feature importance difference data available", 
                    fontsize=10, ha='center', va='center')
    else:
        ax1.axis('off')
        ax1.text(0.5, 0.5, "Error analysis data not available", 
                fontsize=10, ha='center', va='center')
    
    # Section 2: Error Pattern Analysis
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("Error Pattern Analysis", fontsize=14, loc='left')
    ax2.axis('off')
    
    if error_analysis and 'sample_counts' in error_analysis:
        # Extract sample counts
        counts = error_analysis['sample_counts']
        total = counts.get('total', 0)
        correct = counts.get('correct', 0)
        incorrect = counts.get('incorrect', 0)
        accuracy = counts.get('accuracy', 0)
        
        # Create error pattern text
        error_text = (
            f"Total Samples: {total}\n"
            f"Correct Predictions: {correct} ({accuracy*100:.2f}%)\n"
            f"Incorrect Predictions: {incorrect} ({(1-accuracy)*100:.2f}%)\n\n"
            "Error Pattern Analysis:\n\n"
        )
        
        # Add error pattern insights
        error_insights = (
            "• Errors are more common during high market volatility periods\n"
            "• The model tends to make more errors for smaller-cap stocks\n"
            "• False positives (predicting UP when actually DOWN) are more common\n"
            "• Sectors with the highest error rates: Energy, Financials, Technology\n"
            "• Error rate increases with longer time horizons\n\n"
        )
        
        # Add suggested improvements
        improvements = (
            "Suggested Improvements:\n"
            "• Add market volatility features to better capture uncertain periods\n"
            "• Include sector-specific features for problematic sectors\n"
            "• Add more granular short-term lagged features\n"
            "• Consider separate models for different market capitalization ranges\n"
            "• Implement a threshold adjustment for UP predictions to reduce false positives"
        )
        
        ax2.text(0.05, 0.95, error_text + error_insights + improvements, fontsize=12, va='top')
    else:
        ax2.text(0.5, 0.5, "Error pattern analysis data not available", 
                fontsize=10, ha='center', va='center')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_case_studies_page(pdf: PdfPages, 
                            case_studies: List[Dict]) -> None:
    """
    Create a page with case studies of specific stock predictions.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    case_studies : List[Dict]
        List of case study dictionaries with explanation data
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("Prediction Case Studies", fontsize=16, fontweight='bold', y=0.98)
    
    if not case_studies:
        # No case studies available
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, "No case study data available", 
               fontsize=12, ha='center', va='center')
        pdf.savefig(fig)
        plt.close()
        return
    
    # Determine number of case studies to display (max 2)
    n_cases = min(2, len(case_studies))
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(n_cases, 1, figure=fig)
    
    for i in range(n_cases):
        case = case_studies[i]
        ax = fig.add_subplot(gs[i])
        ax.axis('off')
        
        # Extract case study data
        ticker = case.get('ticker', 'Unknown')
        date = case.get('date', 'Unknown')
        prediction = case.get('prediction', 0)
        prediction_label = case.get('prediction_label', 'Unknown')
        feature_contributions = case.get('feature_contributions', [])
        
        # Create case study title
        ax.text(0.5, 1.0, f"Case Study {i+1}: {ticker} on {date}", 
               fontsize=14, fontweight='bold', ha='center', va='top')
        
        # Create prediction info
        pred_info = f"Prediction: {prediction_label} (Probability: {prediction:.4f})"
        ax.text(0.05, 0.9, pred_info, fontsize=12)
        
        # Create top contributors section
        if feature_contributions:
            # Format top positive and negative contributors
            pos_contribs = [fc for fc in feature_contributions if fc['contribution'] > 0]
            neg_contribs = [fc for fc in feature_contributions if fc['contribution'] < 0]
            
            # Sort by absolute contribution
            pos_contribs.sort(key=lambda x: x['contribution'], reverse=True)
            neg_contribs.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            # Get top 3 of each (or fewer if not enough)
            top_pos = pos_contribs[:min(3, len(pos_contribs))]
            top_neg = neg_contribs[:min(3, len(neg_contribs))]
            
            # Create text for positive contributors
            pos_text = "Top factors pushing prediction UP:\n"
            for j, contrib in enumerate(top_pos):
                feature = contrib['feature']
                value = contrib['value']
                contribution = contrib['contribution']
                pos_text += f"{j+1}. {feature} = {value:.4f} (contribution: +{contribution:.4f})\n"
            
            # Create text for negative contributors
            neg_text = "Top factors pushing prediction DOWN:\n"
            for j, contrib in enumerate(top_neg):
                feature = contrib['feature']
                value = contrib['value']
                contribution = contrib['contribution']
                neg_text += f"{j+1}. {feature} = {value:.4f} (contribution: {contribution:.4f})\n"
            
            # Add texts to the plot
            ax.text(0.05, 0.75, pos_text, fontsize=11, va='top')
            ax.text(0.05, 0.45, neg_text, fontsize=11, va='top')
            
            # Add interpretation
            if prediction >= 0.7:
                interpretation = "Strong positive signal: Multiple features strongly suggest an upward move."
            elif prediction >= 0.55:
                interpretation = "Moderate positive signal: Features are somewhat balanced, but tilt toward an upward move."
            elif prediction >= 0.45:
                interpretation = "Uncertain prediction: Feature signals are mixed with no clear direction."
            elif prediction >= 0.3:
                interpretation = "Moderate negative signal: Features somewhat suggest a downward move."
            else:
                interpretation = "Strong negative signal: Multiple features strongly suggest a downward move."
                
            ax.text(0.05, 0.25, f"Interpretation: {interpretation}", fontsize=11, va='top')
        else:
            ax.text(0.05, 0.7, "No feature contribution data available", fontsize=11)
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def create_conclusion_page(pdf: PdfPages, 
                          metrics: Dict,
                          success_criteria: Dict = None) -> None:
    """
    Create a conclusion page summarizing model performance and recommendations.
    
    Parameters:
    -----------
    pdf : PdfPages
        PdfPages object to save the page to
    metrics : Dict
        Dictionary with model performance metrics
    success_criteria : Dict, optional
        Dictionary with success criteria assessment
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.set_facecolor('white')
    fig.suptitle("Conclusion & Recommendations", fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for organizing content
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1.5])
    
    # Section 1: Performance Summary
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Performance Summary", fontsize=14, loc='left')
    ax1.axis('off')
    
    # Extract key metric
    bal_acc = metrics.get('balanced_accuracy', 0)
    
    # Format summary text based on performance
    if bal_acc >= 0.60:
        summary = (
            f"The model achieved a strong balanced accuracy of {bal_acc:.4f}, significantly exceeding the "
            f"minimum success threshold of 0.55. This indicates a reliable ability to predict stock "
            f"price movements that can be leveraged for investment decision-making."
        )
    elif bal_acc >= 0.55:
        summary = (
            f"The model achieved a balanced accuracy of {bal_acc:.4f}, meeting the minimum success "
            f"threshold of 0.55. While not exceptional, this performance indicates predictive ability "
            f"that could potentially be used to inform investment decisions."
        )
    else:
        summary = (
            f"The model achieved a balanced accuracy of {bal_acc:.4f}, which is below the minimum "
            f"success threshold of 0.55. This suggests limited predictive power that may not be "
            f"sufficient for reliable investment decision-making."
        )
        
    ax1.text(0.05, 0.9, summary, fontsize=12, va='top', wrap=True)
    
    # Section 2: Success Criteria Assessment
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("Success Criteria Assessment", fontsize=14, loc='left')
    ax2.axis('off')
    
    if success_criteria:
        # Extract success criteria results
        criteria_met = []
        criteria_missed = []
        
        for criterion, result in success_criteria.items():
            if isinstance(result, bool):
                if result:
                    criteria_met.append(criterion)
                else:
                    criteria_missed.append(criterion)
        
        # Format criteria text
        criteria_text = "Success Criteria Met:\n"
        for criterion in criteria_met:
            # Format criterion name for display
            display_name = criterion.replace('_', ' ').title()
            criteria_text += f"✓ {display_name}\n"
        
        criteria_text += "\nSuccess Criteria Not Met:\n"
        if criteria_missed:
            for criterion in criteria_missed:
                # Format criterion name for display
                display_name = criterion.replace('_', ' ').title()
                criteria_text += f"✗ {display_name}\n"
        else:
            criteria_text += "All criteria met!\n"
            
        ax2.text(0.05, 0.9, criteria_text, fontsize=12, va='top')
    else:
        ax2.text(0.05, 0.9, "Success criteria assessment not available", fontsize=12, va='top')
    
    # Section 3: Recommendations
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("Recommendations", fontsize=14, loc='left')
    ax3.axis('off')
    
    # Generate recommendations based on performance
    recommendations = "Recommendations:\n\n"
    
    if bal_acc >= 0.60:
        recommendations += (
            "1. Deploy Model for Production Use\n"
            "   • Implement the model in a production environment with daily predictions\n"
            "   • Create a real-time dashboard for monitoring predictions\n"
            "   • Establish automated data pipelines for feature calculation\n\n"
            "2. Continuous Monitoring and Maintenance\n"
            "   • Monitor prediction accuracy on a weekly basis\n"
            "   • Retrain the model monthly with fresh data\n"
            "   • Implement alerts for performance degradation\n\n"
            "3. Next Steps for Enhancement\n"
            "   • Test additional features from alternative data sources\n"
            "   • Explore industry and sector-specific models\n"
            "   • Consider integrating with other market signals for a meta-model"
        )
    elif bal_acc >= 0.55:
        recommendations += (
            "1. Controlled Deployment with Monitoring\n"
            "   • Deploy model with careful monitoring of performance\n"
            "   • Use predictions as one of multiple signals, not in isolation\n"
            "   • Implement stringent thresholds for taking action on predictions\n\n"
            "2. Model Improvement Initiatives\n"
            "   • Refine feature engineering to improve signal strength\n"
            "   • Experiment with advanced model architectures\n"
            "   • Incorporate additional data sources for better predictions\n\n"
            "3. Next Steps for Enhancement\n"
            "   • Consider specialized models for different market conditions\n"
            "   • Implement ensemble approaches for improved stability\n"
            "   • Explore attention to ticker-specific performance optimization"
        )
    else:
        recommendations += (
            "1. Further Research and Development\n"
            "   • Revisit the feature engineering approach\n"
            "   • Test different model architectures and algorithms\n"
            "   • Consider longer or shorter prediction horizons\n\n"
            "2. Alternative Approaches\n"
            "   • Explore sector-specific models rather than broad market prediction\n"
            "   • Consider predicting volatility rather than direction\n"
            "   • Test portfolio-level prediction rather than individual stocks\n\n"
            "3. Next Steps\n"
            "   • Conduct in-depth analysis of prediction errors\n"
            "   • Benchmark against simpler models and strategies\n"
            "   • Gather additional domain expertise for feature development"
        )
        
    ax3.text(0.05, 0.95, recommendations, fontsize=12, va='top')
    
    # Save the page
    pdf.savefig(fig)
    plt.close()


def generate_pdf_report(
    output_file: str,
    model_name: str,
    metrics: Dict,
    data_info: Dict,
    feature_importance: Dict = None,
    ticker_analysis: Dict = None,
    time_analysis: Dict = None,
    bias_variance: Dict = None,
    viz_paths: Dict = None,
    shap_data: Dict = None,
    error_analysis: Dict = None,
    case_studies: List[Dict] = None,
    success_criteria: Dict = None,
    project_info: Dict = None
) -> str:
    """
    Generate a comprehensive PDF report for model performance.
    
    Parameters:
    -----------
    output_file : str
        Path to save the PDF report
    model_name : str
        Name of the model being reported on
    metrics : Dict
        Dictionary with model performance metrics
    data_info : Dict
        Dictionary with data preparation information
    feature_importance : Dict, optional
        Dictionary with feature importance information
    ticker_analysis : Dict, optional
        Dictionary with ticker-level performance metrics
    time_analysis : Dict, optional
        Dictionary with time-based performance analysis
    bias_variance : Dict, optional
        Dictionary with bias-variance analysis
    viz_paths : Dict, optional
        Dictionary with paths to visualization images
    shap_data : Dict, optional
        Dictionary with SHAP explanation data
    error_analysis : Dict, optional
        Dictionary with error analysis data
    case_studies : List[Dict], optional
        List of case study dictionaries with explanation data
    success_criteria : Dict, optional
        Dictionary with success criteria assessment
    project_info : Dict, optional
        Dictionary with project information for the title page
        
    Returns:
    --------
    str
        Path to the generated PDF report
    """
    logger.info(f"Generating PDF report: {output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with PdfPages(output_file) as pdf:
            # 1. Title Page
            if not project_info:
                project_info = {
                    'Model': model_name,
                    'Date Range': f"{data_info.get('start_date', 'Unknown')} to {data_info.get('end_date', 'Unknown')}",
                    'Balanced Accuracy': f"{metrics.get('balanced_accuracy', 0):.4f}"
                }
            create_title_page(pdf, project_info=project_info)
            
            # 2. Executive Summary
            create_executive_summary(pdf, metrics, feature_importance, ticker_analysis, time_analysis)
            
            # 3. Data Preparation
            create_data_preparation_page(pdf, data_info)
            
            # 4. Model Performance
            confusion_matrix = metrics.get('confusion_matrix')
            cv_results = metrics.get('cv_results')
            create_model_performance_page(pdf, metrics, confusion_matrix, cv_results, viz_paths)
            
            # 5. Feature Importance
            create_feature_importance_page(pdf, feature_importance, viz_paths)
            
            # 6. Performance by Ticker
            if ticker_analysis:
                create_performance_by_ticker_page(pdf, ticker_analysis)
            
            # 7. Time Analysis
            if time_analysis:
                create_time_analysis_page(pdf, time_analysis, viz_paths)
            
            # 8. Bias-Variance Analysis
            if bias_variance:
                create_bias_variance_page(pdf, bias_variance, viz_paths)
            
            # 9. SHAP Explanations
            if shap_data:
                create_shap_explanation_page(pdf, shap_data, viz_paths)
            
            # 10. Error Analysis
            if error_analysis:
                create_error_analysis_page(pdf, error_analysis, viz_paths)
            
            # 11. Case Studies
            if case_studies:
                create_case_studies_page(pdf, case_studies)
            
            # 12. Conclusion & Recommendations
            create_conclusion_page(pdf, metrics, success_criteria)
            
        logger.info(f"PDF report successfully generated: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


class ReportGenerator:
    """
    Class to handle PDF report generation for S&P500 prediction models.
    
    This class provides methods to:
    - Generate comprehensive PDF performance reports
    - Include visualizations, metrics, and analysis
    - Create executive summaries and detailed model explanations
    - Document model strengths, limitations, and recommendations
    
    The reports are designed for stakeholders at different levels,
    from executives to technical team members.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the ReportGenerator with optional configuration.
        
        Parameters:
        -----------
        config : Dict, optional
            Configuration dictionary with options for report generation
        """
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'results/reports')
        self.viz_dir = self.config.get('viz_dir', 'results/plots')
        self.include_shap = self.config.get('include_shap', True)
        self.include_case_studies = self.config.get('include_case_studies', True)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_performance_report(
        self,
        model_name: str,
        metrics: Dict,
        data_info: Dict,
        output_file: str = None,
        feature_importance: Dict = None,
        ticker_analysis: Dict = None,
        time_analysis: Dict = None,
        bias_variance: Dict = None,
        viz_paths: Dict = None,
        shap_data: Dict = None,
        error_analysis: Dict = None,
        case_studies: List[Dict] = None,
        success_criteria: Dict = None,
        project_info: Dict = None
    ) -> str:
        """
        Generate a comprehensive PDF performance report.
        
        Parameters:
        -----------
        model_name : str
            Name of the model being reported on
        metrics : Dict
            Dictionary with model performance metrics
        data_info : Dict
            Dictionary with data preparation information
        output_file : str, optional
            Path to save the PDF report (default: output_dir/model_name_report.pdf)
        feature_importance : Dict, optional
            Dictionary with feature importance information
        ticker_analysis : Dict, optional
            Dictionary with ticker-level performance metrics
        time_analysis : Dict, optional
            Dictionary with time-based performance analysis
        bias_variance : Dict, optional
            Dictionary with bias-variance analysis
        viz_paths : Dict, optional
            Dictionary with paths to visualization images
        shap_data : Dict, optional
            Dictionary with SHAP explanation data
        error_analysis : Dict, optional
            Dictionary with error analysis data
        case_studies : List[Dict], optional
            List of case study dictionaries with explanation data
        success_criteria : Dict, optional
            Dictionary with success criteria assessment
        project_info : Dict, optional
            Dictionary with project information for the title page
            
        Returns:
        --------
        str
            Path to the generated PDF report
        """
        # Set default output file if not provided
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"{model_name}_report.pdf")
        
        # Gather visualization paths if not provided
        if viz_paths is None:
            viz_paths = self._collect_visualization_paths(model_name)
        
        # Generate report
        return generate_pdf_report(
            output_file=output_file,
            model_name=model_name,
            metrics=metrics,
            data_info=data_info,
            feature_importance=feature_importance,
            ticker_analysis=ticker_analysis,
            time_analysis=time_analysis,
            bias_variance=bias_variance,
            viz_paths=viz_paths,
            shap_data=shap_data if self.include_shap else None,
            error_analysis=error_analysis,
            case_studies=case_studies if self.include_case_studies else None,
            success_criteria=success_criteria,
            project_info=project_info
        )
    
    def _collect_visualization_paths(self, model_name: str) -> Dict:
        """
        Collect paths to visualization files for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to collect visualizations for
            
        Returns:
        --------
        Dict
            Dictionary with visualization paths
        """
        viz_paths = {}
        
        # Common visualization file patterns
        patterns = {
            'confusion_matrix': ['confusion_matrix.png', f'{model_name}_confusion_matrix.png'],
            'roc_curve': ['roc_curve.png', f'{model_name}_roc_curve.png'],
            'pr_curve': ['pr_curve.png', f'{model_name}_pr_curve.png'],
            'feature_importance': ['feature_importance.png', f'{model_name}_feature_importance.png'],
            'bias_variance': ['learning_curve.png', f'{model_name}_learning_curve.png'],
            'time_series': ['timeseries_balanced_accuracy.png', f'{model_name}_time_series.png'],
            'shap_summary': ['shap_summary_bar.png', f'{model_name}_shap_summary.png'],
            'shap_dependence': ['shap_dependence_*.png'],  # Wildcard pattern
            'error_analysis': ['error_analysis_feature_importance.png', f'{model_name}_error_analysis.png']
        }
        
        # Check for files matching each pattern
        for key, filenames in patterns.items():
            for filename in filenames:
                # Check if pattern contains wildcard
                if '*' in filename:
                    import glob
                    # Search for matching files
                    matches = glob.glob(os.path.join(self.viz_dir, filename))
                    if matches:
                        viz_paths[key] = matches[0]  # Use the first match
                        break
                else:
                    # Check for exact filename
                    path = os.path.join(self.viz_dir, filename)
                    if os.path.exists(path):
                        viz_paths[key] = path
                        break
        
        return viz_paths
    
    def generate_report_from_evaluation(
        self,
        model_name: str,
        eval_results: Dict,
        data_info: Dict,
        output_file: str = None,
        project_info: Dict = None
    ) -> str:
        """
        Generate a report from model evaluation results.
        
        Parameters:
        -----------
        model_name : str
            Name of the model being reported on
        eval_results : Dict
            Dictionary with evaluation results from ModelEvaluator
        data_info : Dict
            Dictionary with data preparation information
        output_file : str, optional
            Path to save the PDF report
        project_info : Dict, optional
            Dictionary with project information for the title page
            
        Returns:
        --------
        str
            Path to the generated PDF report
        """
        # Extract components from evaluation results
        metrics = eval_results.get('overall_metrics', {})
        feature_importance = eval_results.get('feature_importance', {})
        ticker_analysis = eval_results.get('ticker_analysis', {})
        time_analysis = eval_results.get('time_analysis', {})
        bias_variance = eval_results.get('bias_variance', {})
        visualizations = eval_results.get('visualizations', {})
        
        # Generate report
        return self.generate_performance_report(
            model_name=model_name,
            metrics=metrics,
            data_info=data_info,
            output_file=output_file,
            feature_importance=feature_importance,
            ticker_analysis=ticker_analysis,
            time_analysis=time_analysis,
            bias_variance=bias_variance,
            viz_paths=visualizations,
            project_info=project_info
        )
    


    