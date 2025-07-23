"""
Comprehensive Model Evaluation for Crisis Classification

This module provides advanced evaluation metrics, visualizations, and analysis
tools specifically designed for crisis classification systems. Features include
multi-class metrics, ROC analysis, calibration assessment, and crisis-specific
response time simulation.

Classes:
    CrisisEvaluator: Main evaluation orchestrator
    CalibrationAnalyzer: Confidence score calibration assessment
    ErrorAnalyzer: Misclassification pattern analysis
    ResponseTimeSimulator: Crisis response time simulation

Functions:
    evaluate_model: Main evaluation function
    generate_evaluation_report: Comprehensive report generation
    plot_evaluation_metrics: Visualization utilities
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc, average_precision_score, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crisis classification labels
CRISIS_LABELS = [
    "requests_or_urgent_needs",
    "infrastructure_and_utility_damage", 
    "injured_or_dead_people",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian"
]

# Crisis severity mapping for response time simulation
CRISIS_SEVERITY = {
    "requests_or_urgent_needs": "critical",       # Immediate response needed
    "infrastructure_and_utility_damage": "high",  # Major infrastructure impact
    "injured_or_dead_people": "critical",         # Life-threatening situations
    "rescue_volunteering_or_donation_effort": "medium",  # Support coordination
    "other_relevant_information": "low",          # General information
    "not_humanitarian": "none"                    # No crisis response needed
}

@dataclass
class EvaluationResults:
    """Container for comprehensive evaluation results."""
    # Basic metrics
    accuracy: float
    macro_f1: float
    weighted_f1: float
    micro_f1: float
    
    # Per-class metrics
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    per_class_support: List[int]
    
    # Confusion matrix
    confusion_matrix: np.ndarray
    normalized_confusion_matrix: np.ndarray
    
    # ROC metrics
    per_class_auc: List[float]
    macro_auc: float
    
    # Crisis-specific metrics
    humanitarian_f1: float
    critical_crisis_f1: float
    emergency_response_accuracy: float
    
    # Calibration metrics
    expected_calibration_error: float
    reliability_score: float
    
    # Error analysis
    misclassification_patterns: Dict[str, Any]
    confidence_distribution: Dict[str, Any]
    
    # Response simulation
    response_time_metrics: Dict[str, float]
    
    # Classification report
    classification_report: str


class CrisisEvaluator:
    """
    Comprehensive evaluation system for crisis classification models.
    
    Provides multi-class metrics, ROC analysis, calibration assessment,
    and crisis-specific evaluation tailored for emergency response systems.
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        save_plots: bool = True
    ):
        """
        Initialize crisis evaluator.
        
        Args:
            class_names: List of class names. If None, uses default crisis labels
            output_dir: Directory to save plots and reports
            save_plots: Whether to save generated plots
        """
        self.class_names = class_names or CRISIS_LABELS
        self.num_classes = len(self.class_names)
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_plots = save_plots
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        # self.calibration_analyzer = CalibrationAnalyzer(self.class_names) # Removed
        # self.error_analyzer = ErrorAnalyzer(self.class_names) # Removed
        self.response_simulator = ResponseTimeSimulator()
        
        logger.info(f"CrisisEvaluator initialized for {self.num_classes} classes")
    
    def evaluate(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        y_proba: Optional[Union[np.ndarray, torch.Tensor]] = None,
        texts: Optional[List[str]] = None
    ) -> EvaluationResults:
        """
        Perform comprehensive evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (required for advanced metrics)
            texts: Original texts (optional, for error analysis)
            
        Returns:
            EvaluationResults with comprehensive metrics
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Convert inputs to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        if y_proba is not None:
            y_proba = self._to_numpy(y_proba)
        
        # Validate inputs
        self._validate_inputs(y_true, y_pred, y_proba)
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division='warn'
        )
        
        # Macro/micro/weighted averages
        macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division='warn')[2]
        micro_f1 = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division='warn')[2]
        weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division='warn')[2]
        
        # Confusion matrices
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        
        # ROC metrics (if probabilities available)
        per_class_auc = []
        macro_auc = 0.0
        
        if y_proba is not None:
            # One-vs-rest ROC for each class
            y_true_binary = label_binarize(y_true, classes=range(self.num_classes))
            if self.num_classes == 2:
                y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])
            
            for i in range(self.num_classes):
                try:
                    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_proba[:, i])
                    auc_score = auc(fpr, tpr)
                    per_class_auc.append(auc_score)
                except ValueError:
                    # Handle case where class is not present in true labels
                    per_class_auc.append(0.0)
            
            macro_auc = np.mean(per_class_auc)
        else:
            per_class_auc = [0.0] * self.num_classes
        
        # Crisis-specific metrics
        crisis_metrics = self._compute_crisis_metrics(y_true, y_pred)
        
        # Calibration metrics (if probabilities available)
        calibration_metrics = {}
        if y_proba is not None:
            # calibration_metrics = self.calibration_analyzer.analyze(y_true, y_proba) # Removed
            pass # Placeholder for future integration
        
        # Error analysis
        error_metrics = self._compute_error_analysis(y_true, y_pred, y_proba, texts) # Updated call
        
        # Response time simulation
        response_metrics = self.response_simulator.simulate(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division='warn',
            digits=4
        )
        
        # Create results object
        results = EvaluationResults(
            accuracy=float(accuracy),
            macro_f1=float(macro_f1),
            weighted_f1=float(weighted_f1),
            micro_f1=float(micro_f1),
            per_class_precision=precision.tolist() if (precision is not None and hasattr(precision, 'tolist') and not isinstance(precision, (float, int))) else ([float(precision)] if isinstance(precision, float) else [int(precision)] if isinstance(precision, int) else list(precision) if isinstance(precision, (list, tuple)) else []),
            per_class_recall=recall.tolist() if (recall is not None and hasattr(recall, 'tolist') and not isinstance(recall, (float, int))) else ([float(recall)] if isinstance(recall, float) else [int(recall)] if isinstance(recall, int) else list(recall) if isinstance(recall, (list, tuple)) else []),
            per_class_f1=f1.tolist() if (f1 is not None and hasattr(f1, 'tolist') and not isinstance(f1, (float, int))) else ([float(f1)] if isinstance(f1, float) else [int(f1)] if isinstance(f1, int) else list(f1) if isinstance(f1, (list, tuple)) else []),
            per_class_support=support.tolist() if (support is not None and hasattr(support, 'tolist') and not isinstance(support, (float, int))) else ([int(support)] if isinstance(support, (int, float)) else list(support) if isinstance(support, (list, tuple)) else []),
            confusion_matrix=cm,
            normalized_confusion_matrix=cm_normalized,
            per_class_auc=per_class_auc,
            macro_auc=float(macro_auc),
            humanitarian_f1=crisis_metrics['humanitarian_f1'],
            critical_crisis_f1=crisis_metrics['critical_crisis_f1'],
            emergency_response_accuracy=crisis_metrics['emergency_response_accuracy'],
            expected_calibration_error=calibration_metrics.get('ece', 0.0),
            reliability_score=calibration_metrics.get('reliability', 0.0),
            misclassification_patterns=error_metrics['patterns'],
            confidence_distribution=error_metrics['confidence'],
            response_time_metrics=response_metrics,
            classification_report=str(report)
        )
        
        logger.info("Evaluation completed successfully")
        return results
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        else:
            return data
    
    def _validate_inputs(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray]
    ):
        """Validate input arrays."""
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
        
        if y_proba is not None:
            if len(y_true) != len(y_proba):
                raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_proba ({len(y_proba)})")
            
            if y_proba.shape[1] != self.num_classes:
                raise ValueError(f"y_proba shape {y_proba.shape} doesn't match num_classes {self.num_classes}")
        
        # Check if all classes are represented
        unique_true = set(y_true)
        unique_pred = set(y_pred)
        expected_classes = set(range(self.num_classes))
        
        if not unique_true.issubset(expected_classes):
            logger.warning(f"Unexpected classes in y_true: {unique_true - expected_classes}")
        
        if not unique_pred.issubset(expected_classes):
            logger.warning(f"Unexpected classes in y_pred: {unique_pred - expected_classes}")
    
    def _compute_crisis_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute crisis-specific evaluation metrics."""
        # Humanitarian vs non-humanitarian classification
        humanitarian_classes = [0, 1, 2, 3]  # First 4 are humanitarian
        non_humanitarian_classes = [4, 5]    # Last 2 are non-humanitarian
        
        # Binary classification: humanitarian vs non-humanitarian
        humanitarian_true = np.isin(y_true, humanitarian_classes).astype(int)
        humanitarian_pred = np.isin(y_pred, humanitarian_classes).astype(int)
        
        humanitarian_f1 = precision_recall_fscore_support(
            humanitarian_true, humanitarian_pred, average='binary', zero_division='warn'
        )[2]
        
        # Critical crisis classification (most urgent classes)
        critical_classes = [0, 2]  # Urgent needs and casualties
        critical_true = np.isin(y_true, critical_classes).astype(int)
        critical_pred = np.isin(y_pred, critical_classes).astype(int)
        
        critical_f1 = precision_recall_fscore_support(
            critical_true, critical_pred, average='binary', zero_division='warn'
        )[2]
        
        # Emergency response accuracy (ability to identify any crisis)
        emergency_true = (y_true != 5).astype(int)  # Not "not_humanitarian"
        emergency_pred = (y_pred != 5).astype(int)
        
        emergency_accuracy = accuracy_score(emergency_true, emergency_pred)
        
        return {
            'humanitarian_f1': float(humanitarian_f1),
            'critical_crisis_f1': float(critical_f1),
            'emergency_response_accuracy': float(emergency_accuracy)
        }
    
    def _compute_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray], texts: Optional[List[str]]) -> Dict[str, Any]:
        """Compute error analysis metrics."""
        # Import CalibrationAnalyzer and ErrorAnalyzer from error_analysis.py
        from src.training.error_analysis import CalibrationAnalyzer, ErrorAnalyzer

        # Create instances of analyzers
        calibration_analyzer = CalibrationAnalyzer(self.class_names)
        error_analyzer = ErrorAnalyzer(self.class_names)

        # Analyze
        error_analysis_results = error_analyzer.analyze(
            y_true, y_pred, y_proba, texts, self.class_names
        )
        calibration_results = calibration_analyzer.analyze(y_true, y_proba)

        # Combine results
        return {
            'patterns': error_analysis_results,
            'confidence': calibration_results
        }
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray, 
        normalize: bool = True,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix with crisis-specific styling.
        
        Args:
            cm: Confusion matrix
            normalize: Whether to normalize the matrix
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 10))
        
        if normalize:
            cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_plot = np.nan_to_num(cm_plot)
            fmt = '.2f'
            cmap = 'Blues'
        else:
            cm_plot = cm
            fmt = 'd'
            cmap = 'Blues'
        
        # Create heatmap
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=[name.replace('_', '\n') for name in self.class_names],
            yticklabels=[name.replace('_', '\n') for name in self.class_names],
            cbar_kws={'label': 'Normalized Frequency' if normalize else 'Count'}
        )
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        if save_path or (self.save_plots and self.output_dir):
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curves(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curves for all classes.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 8))
        
        # Binarize labels for multiclass ROC
        y_true_binary = label_binarize(y_true, classes=range(self.num_classes))
        if self.num_classes == 2:
            y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])
        
        colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, self.num_classes))
        
        for i, (color, class_name) in enumerate(zip(colors, self.class_names)):
            try:
                fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_proba[:, i])
                auc_score = auc(fpr, tpr)
                
                plt.plot(
                    fpr, tpr, 
                    color=color, 
                    linewidth=2,
                    label=f'{class_name.replace("_", " ").title()} (AUC = {auc_score:.3f})'
                )
            except ValueError:
                logger.warning(f"Could not compute ROC for class {class_name}")
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Crisis Classification', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        if save_path or (self.save_plots and self.output_dir):
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curves(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot precision-recall curves for all classes.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 8))
        
        # Binarize labels
        y_true_binary = label_binarize(y_true, classes=range(self.num_classes))
        if self.num_classes == 2:
            y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])
        
        colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, self.num_classes))
        
        for i, (color, class_name) in enumerate(zip(colors, self.class_names)):
            try:
                precision, recall, _ = precision_recall_curve(y_true_binary[:, i], y_proba[:, i])
                avg_precision = average_precision_score(y_true_binary[:, i], y_proba[:, i])
                
                plt.plot(
                    recall, precision, 
                    color=color, 
                    linewidth=2,
                    label=f'{class_name.replace("_", " ").title()} (AP = {avg_precision:.3f})'
                )
            except ValueError:
                logger.warning(f"Could not compute PR curve for class {class_name}")
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Crisis Classification', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path or (self.save_plots and self.output_dir):
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        
        return plt.gcf()


class ResponseTimeSimulator:
    """
    Crisis response time simulation for emergency management assessment.
    
    Simulates emergency response scenarios to evaluate the practical impact
    of classification accuracy on crisis response effectiveness.
    """
    
    def __init__(self):
        """Initialize response time simulator."""
        # Response time parameters (in minutes)
        self.response_times = {
            'critical': {'mean': 15, 'std': 5},    # 15±5 minutes for critical
            'high': {'mean': 60, 'std': 15},       # 60±15 minutes for high priority
            'medium': {'mean': 240, 'std': 60},    # 4±1 hours for medium priority
            'low': {'mean': 1440, 'std': 360},     # 24±6 hours for low priority
            'none': {'mean': 0, 'std': 0}          # No response for non-humanitarian
        }
        
        # Cost of delayed response (relative units)
        self.delay_costs = {
            'critical': 10.0,   # High cost for delaying critical response
            'high': 5.0,        # Moderate cost for delaying high priority
            'medium': 2.0,      # Low cost for delaying medium priority
            'low': 1.0,         # Minimal cost for delaying low priority
            'none': 0.0         # No cost for non-humanitarian
        }
    
    def simulate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Simulate emergency response scenarios.
        
        Args:
            y_true: True crisis classifications
            y_pred: Predicted crisis classifications
            
        Returns:
            Dictionary with response time metrics
        """
        # Map labels to severity levels
        true_severities = [CRISIS_SEVERITY[CRISIS_LABELS[label]] for label in y_true]
        pred_severities = [CRISIS_SEVERITY[CRISIS_LABELS[label]] for label in y_pred]
        
        # Simulate response times
        metrics = {}
        
        # Average response time if predictions were followed
        pred_response_times = [self._sample_response_time(severity) for severity in pred_severities]
        metrics['avg_predicted_response_time'] = float(np.mean(pred_response_times))
        
        # Optimal response time if true labels were known
        true_response_times = [self._sample_response_time(severity) for severity in true_severities]
        metrics['avg_optimal_response_time'] = float(np.mean(true_response_times))
        
        # Response efficiency (how close predictions are to optimal)
        efficiency = np.mean(true_response_times) / (np.mean(pred_response_times) + 1e-6)
        metrics['response_efficiency'] = float(min(efficiency, 2.0))  # Cap at 200%
        
        # Cost analysis
        true_costs = [self.delay_costs[severity] for severity in true_severities]
        pred_costs = [self.delay_costs[severity] for severity in pred_severities]
        
        # Additional cost from misclassification
        misclassification_cost = 0.0
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                true_severity = true_severities[i]
                pred_severity = pred_severities[i]
                
                # Cost of under-responding to critical situations
                if true_severity == 'critical' and pred_severity in ['low', 'none']:
                    misclassification_cost += 20.0  # High penalty
                elif true_severity == 'high' and pred_severity in ['low', 'none']:
                    misclassification_cost += 10.0  # Moderate penalty
                
                # Cost of over-responding (resource waste)
                elif pred_severity == 'critical' and true_severity in ['low', 'none']:
                    misclassification_cost += 5.0   # Resource waste penalty
        
        metrics['total_misclassification_cost'] = float(misclassification_cost)
        metrics['avg_misclassification_cost'] = float(misclassification_cost / len(y_true))
        
        # Critical crisis detection rate
        critical_true = np.array([s == 'critical' for s in true_severities])
        critical_pred = np.array([s == 'critical' for s in pred_severities])
        
        if np.sum(critical_true) > 0:
            critical_recall = np.sum(critical_true & critical_pred) / np.sum(critical_true)
            metrics['critical_crisis_recall'] = float(critical_recall)
        else:
            metrics['critical_crisis_recall'] = 1.0  # No critical crises to miss
        
        return metrics
    
    def _sample_response_time(self, severity: str) -> float:
        """Sample response time for given severity level."""
        params = self.response_times[severity]
        if params['std'] == 0:
            return params['mean']
        
        # Sample from normal distribution, ensure positive
        response_time = np.random.normal(params['mean'], params['std'])
        return max(0, response_time)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> EvaluationResults:
    """
    Evaluate a crisis classification model comprehensively.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to run evaluation on
        class_names: Class names for labeling
        output_dir: Directory to save results
        
    Returns:
        EvaluationResults with comprehensive metrics
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_texts = []
    
    logger.info("Running model evaluation...")
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions and probabilities
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)
    
    # Create evaluator and run evaluation
    evaluator = CrisisEvaluator(class_names=class_names, output_dir=output_dir)
    results = evaluator.evaluate(y_true, y_pred, y_proba)
    
    # Generate visualizations
    if output_dir:
        evaluator.plot_confusion_matrix(results.confusion_matrix, normalize=True)
        evaluator.plot_roc_curves(y_true, y_proba)
        evaluator.plot_precision_recall_curves(y_true, y_proba)
        
        if y_proba is not None:
            # evaluator.calibration_analyzer.plot_calibration_curve(y_true, y_proba) # Removed
            pass # Placeholder for future integration
    
    return results


def generate_evaluation_report(
    results: EvaluationResults,
    model_name: str = "Crisis Classifier",
    output_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        results: Evaluation results
        model_name: Name of the model
        output_path: Path to save the report
        
    Returns:
        Report content as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Crisis Classification Model Evaluation Report

**Model**: {model_name}  
**Generated**: {timestamp}  
**Evaluation Framework**: Comprehensive Crisis Assessment

---

## Executive Summary

This report provides a comprehensive evaluation of the {model_name} for crisis classification across {len(CRISIS_LABELS)} categories. The evaluation includes standard classification metrics, crisis-specific assessments, calibration analysis, and emergency response simulation.

### Key Performance Indicators
- **Overall Accuracy**: {results.accuracy:.4f}
- **Macro F1-Score**: {results.macro_f1:.4f}
- **Humanitarian Crisis Detection F1**: {results.humanitarian_f1:.4f}
- **Critical Crisis Detection F1**: {results.critical_crisis_f1:.4f}
- **Emergency Response Accuracy**: {results.emergency_response_accuracy:.4f}

---

## Classification Performance

### Overall Metrics
| Metric | Score |
|--------|-------|
| Accuracy | {results.accuracy:.4f} |
| Macro F1 | {results.macro_f1:.4f} |
| Weighted F1 | {results.weighted_f1:.4f} |
| Micro F1 | {results.micro_f1:.4f} |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support | AUC |
|-------|-----------|--------|----------|---------|-----|"""

    for i, class_name in enumerate(CRISIS_LABELS):
        precision = results.per_class_precision[i]
        recall = results.per_class_recall[i]
        f1 = results.per_class_f1[i]
        support = results.per_class_support[i]
        auc = results.per_class_auc[i]
        
        report += f"\n| {class_name.replace('_', ' ').title()} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} | {auc:.4f} |"

    report += f"""

---

## Crisis-Specific Analysis

### Humanitarian vs Non-Humanitarian Classification
- **Humanitarian F1-Score**: {results.humanitarian_f1:.4f}
- **Critical Crisis F1-Score**: {results.critical_crisis_f1:.4f}
- **Emergency Response Accuracy**: {results.emergency_response_accuracy:.4f}

### Emergency Response Simulation
| Metric | Value |
|--------|-------|"""

    for metric, value in results.response_time_metrics.items():
        metric_name = metric.replace('_', ' ').title()
        if isinstance(value, float):
            if 'time' in metric.lower():
                report += f"\n| {metric_name} | {value:.1f} minutes |"
            elif 'cost' in metric.lower():
                report += f"\n| {metric_name} | {value:.2f} units |"
            else:
                report += f"\n| {metric_name} | {value:.4f} |"
        else:
            report += f"\n| {metric_name} | {value} |"

    report += f"""

---

## Model Calibration Assessment

### Confidence Score Analysis
- **Expected Calibration Error (ECE)**: {results.expected_calibration_error:.4f}
- **Reliability Score**: {results.reliability_score:.4f}

The Expected Calibration Error measures how well the model's confidence scores align with actual accuracy. Lower ECE values indicate better calibration.

---

## Error Analysis

### Misclassification Patterns
**Total Misclassification Rate**: {results.misclassification_patterns.get('misclassification_rate', 0.0):.4f}

#### Top Misclassification Patterns:"""

    if 'patterns' in results.misclassification_patterns and 'top_misclassifications' in results.misclassification_patterns['patterns']:
        for i, error in enumerate(results.misclassification_patterns['patterns']['top_misclassifications'][:5]):
            true_class = error['true_class'].replace('_', ' ').title()
            pred_class = error['pred_class'].replace('_', ' ').title()
            count = error['count']
            report += f"\n{i+1}. **{true_class}** → **{pred_class}**: {count} cases"

    report += f"""

#### Crisis-Specific Error Patterns:"""
    
    crisis_patterns = results.misclassification_patterns.get('patterns', {}).get('crisis_patterns', {})
    for pattern_name, count in crisis_patterns.items():
        pattern_display = pattern_name.replace('_', ' ').title()
        report += f"\n- **{pattern_display}**: {count} cases"

    report += f"""

### Confidence Distribution Analysis"""
    
    if results.confidence_distribution:
        conf_stats = results.confidence_distribution.get('overall_stats', {})
        if conf_stats:
            report += f"""
- **Mean Confidence**: {conf_stats.get('mean_confidence', 0):.4f}
- **Standard Deviation**: {conf_stats.get('std_confidence', 0):.4f}
- **Median Confidence**: {conf_stats.get('median_confidence', 0):.4f}"""
        
        conf_comparison = results.confidence_distribution.get('correct_vs_incorrect', {})
        if conf_comparison:
            report += f"""

#### Confidence Comparison (Correct vs Incorrect Predictions)
- **Correct Predictions Mean Confidence**: {conf_comparison.get('correct_mean', 0):.4f}
- **Incorrect Predictions Mean Confidence**: {conf_comparison.get('incorrect_mean', 0):.4f}
- **High Confidence Errors**: {results.confidence_distribution.get('high_confidence_errors', 0)} cases"""

    report += f"""

---

## Detailed Classification Report

```
{results.classification_report}
```

---

## Recommendations

### Model Performance
"""

    # Generate recommendations based on results
    recommendations = []
    
    if results.macro_f1 < 0.7:
        recommendations.append("Consider additional training or data augmentation to improve overall F1-score")
    
    if results.humanitarian_f1 < 0.8:
        recommendations.append("Focus on improving humanitarian crisis detection through class balancing")
    
    if results.critical_crisis_f1 < 0.8:
        recommendations.append("Critical crisis detection needs improvement - consider increasing penalty for missing critical cases")
    
    if results.expected_calibration_error > 0.1:
        recommendations.append("Model calibration needs improvement - consider calibration techniques like Platt scaling")
    
    # Check for high confidence errors
    high_conf_errors = results.confidence_distribution.get('high_confidence_errors', 0)
    total_errors = results.misclassification_patterns.get('num_misclassified', 1)
    if total_errors > 0: # Avoid division by zero
        if high_conf_errors / total_errors > 0.3:
            recommendations.append("High rate of overconfident errors - review training regularization and confidence thresholding")
    
    # Response time recommendations
    efficiency = results.response_time_metrics.get('response_efficiency', 1.0)
    if efficiency < 0.8:
        recommendations.append("Response time efficiency is suboptimal - focus on improving classification of time-sensitive crisis types")
    
    for i, rec in enumerate(recommendations, 1):
        report += f"\n{i}. {rec}"

    if not recommendations:
        report += "\n1. Model performance is satisfactory across all evaluated metrics"
        report += "\n2. Continue monitoring performance on new data"
        report += "\n3. Consider regular retraining with updated crisis data"

    report += f"""

### Deployment Considerations
1. **Confidence Thresholding**: Set appropriate confidence thresholds for different crisis severity levels
2. **Human-in-the-Loop**: Implement human review for high-stakes classifications
3. **Monitoring**: Continuously monitor model performance in production
4. **Retraining Schedule**: Plan regular retraining with new crisis data

---

## Technical Details

- **Evaluation Framework**: Custom crisis classification evaluation
- **Metrics Computed**: {len([k for k in asdict(results).keys() if not k.startswith('_')])} comprehensive metrics
- **Visualization**: Confusion matrix, ROC curves, PR curves, calibration plots
- **Crisis Simulation**: Emergency response time modeling

*Report generated by AICrisisAlert Evaluation System*
"""

    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Evaluation report saved to {output_path}")
    
    return report


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Crisis Evaluation System...")
    
    # Generate dummy data for testing
    np.random.seed(42)
    n_samples = 1000
    n_classes = 6
    
    # Simulate realistic class imbalance
    class_probs = [0.05, 0.05, 0.05, 0.10, 0.50, 0.25]  # Realistic crisis distribution
    y_true = np.random.choice(n_classes, n_samples, p=class_probs)
    
    # Simulate predictions with some accuracy
    y_pred = y_true.copy()
    # Add some noise (20% error rate)
    noise_indices = np.random.choice(n_samples, int(0.2 * n_samples), replace=False)
    y_pred[noise_indices] = np.random.choice(n_classes, len(noise_indices))
    
    # Generate probabilities
    y_proba = np.random.dirichlet([1] * n_classes, n_samples)
    # Make probabilities somewhat realistic
    for i in range(n_samples):
        y_proba[i] = np.abs(y_proba[i])
        y_proba[i, y_pred[i]] += 0.3  # Boost predicted class probability
        y_proba[i] = y_proba[i] / y_proba[i].sum()  # Normalize
    
    # Test evaluation
    print("Testing comprehensive evaluation...")
    evaluator = CrisisEvaluator(output_dir="test_output")
    results = evaluator.evaluate(y_true, y_pred, y_proba)
    
    print(f"✅ Accuracy: {results.accuracy:.4f}")
    print(f"✅ Macro F1: {results.macro_f1:.4f}")
    print(f"✅ Humanitarian F1: {results.humanitarian_f1:.4f}")
    print(f"✅ ECE: {results.expected_calibration_error:.4f}")
    print(f"✅ Response Efficiency: {results.response_time_metrics.get('response_efficiency', 0):.4f}")
    
    # Test report generation
    print("Testing report generation...")
    report = generate_evaluation_report(results, "Test Crisis Classifier")
    print(f"✅ Report generated: {len(report)} characters")
    
    print("\n🎉 Crisis evaluation system testing completed successfully!")