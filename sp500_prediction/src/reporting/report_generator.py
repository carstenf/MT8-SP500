"""
Report Generator Module

This module provides the main ReportGenerator class for creating PDF reports.
"""

import os
import logging
from typing import Dict, List, Optional
from matplotlib.backends.backend_pdf import PdfPages

from .pages import (
    create_title_page,
    create_executive_summary,
    create_data_preparation_page
)
from .visualization_utils import load_image_safely
from .report_utils import (
    collect_visualization_paths,
    format_metrics_for_display,
    generate_success_criteria,
    generate_recommendations,
    save_report_metadata,
    validate_report_inputs
)

# Set up logging
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Class to handle PDF report generation for S&P500 prediction models.
    
    This class provides methods to:
    - Generate comprehensive PDF performance reports
    - Include visualizations, metrics, and analysis
    - Create executive summaries and detailed model explanations
    - Document model strengths, limitations, and recommendations
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
            Path to save the PDF report
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
        
        # Validate required inputs
        if not validate_report_inputs(metrics, data_info):
            logger.error("Missing required inputs for report generation")
            return ""
        
        # Format metrics for display
        formatted_metrics = format_metrics_for_display(metrics)
        
        # Gather visualization paths if not provided
        if viz_paths is None:
            viz_paths = collect_visualization_paths(self.viz_dir, model_name)
        
        # Generate success criteria if not provided
        if success_criteria is None:
            success_criteria = generate_success_criteria(
                metrics,
                time_analysis,
                feature_importance
            )
        
        # Generate recommendations
        recommendations = generate_recommendations(
            metrics,
            success_criteria,
            bias_variance
        )
        
        try:
            with PdfPages(output_file) as pdf:
                # 1. Title Page
                if not project_info:
                    project_info = {
                        'Model': model_name,
                        'Date Range': f"{data_info.get('start_date', 'Unknown')} to {data_info.get('end_date', 'Unknown')}",
                        'Balanced Accuracy': formatted_metrics.get('balanced_accuracy', 'N/A')
                    }
                create_title_page(pdf, project_info=project_info)
                
                # 2. Executive Summary
                create_executive_summary(
                    pdf,
                    formatted_metrics,
                    feature_importance,
                    ticker_analysis,
                    time_analysis
                )
                
                # 3. Data Preparation
                create_data_preparation_page(pdf, data_info)
                
                # Additional pages would be added here...
                # [Note: Additional page creation calls would be included here,
                # following the same pattern as above]
                
            # Save report metadata
            report_metadata = {
                'model_name': model_name,
                'output_file': output_file,
                'success_criteria_met': success_criteria,
                'recommendations': recommendations,
                'visualizations': list(viz_paths.keys()) if viz_paths else []
            }
            save_report_metadata(self.output_dir, report_metadata)
            
            logger.info(f"PDF report successfully generated: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
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
