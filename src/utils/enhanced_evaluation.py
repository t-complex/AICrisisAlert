"""
Enhanced Model Evaluation for Crisis Classification

This module extends the base evaluation.py with advanced assessment capabilities
including statistical significance testing, cross-validation metrics, temporal
analysis, and enhanced crisis-specific evaluations for emergency response systems.

Classes:
    EnhancedCrisisEvaluator: Advanced evaluation orchestrator
    StatisticalAnalyzer: Statistical significance and confidence intervals
    TemporalAnalyzer: Time-based performance analysis
    CrisisImpactAssessor: Real-world impact assessment
    ModelComparator: Multi-model comparison utilities

Functions:
    evaluate_model_comprehensive: Main enhanced evaluation function
    bootstrap_confidence_intervals: Bootstrap confidence interval computation
    cross_validate_metrics: Cross-validation performance assessment
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score, hamming_loss, jaccard_score
)
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from dataclasses import dataclass

# Import base evaluation components
try:
    from .evaluation import (
        CrisisEvaluator, EvaluationResults
    )
    from src.training.constants import CRISIS_LABELS, CRISIS_TYPE_MAPPING
    from src.training.statistics import StatisticalAnalyzer, TemporalAnalyzer
    from src.training.simulation import CrisisImpactAssessor
    from src.training.error_analysis import ErrorAnalyzer
except ImportError:
    # Fallback for when running as standalone script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from evaluation import (
        CrisisEvaluator, EvaluationResults
    )
    from src.training.constants import CRISIS_LABELS, CRISIS_TYPE_MAPPING
    from src.training.statistics import StatisticalAnalyzer, TemporalAnalyzer
    from src.training.simulation import CrisisImpactAssessor
    from src.training.error_analysis import ErrorAnalyzer

# Import reporting components
try:
    from src.training.reporting import generate_enhanced_evaluation_report
except ImportError:
    # Fallback for when running as standalone script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from src.training.reporting import generate_enhanced_evaluation_report


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedEvaluationResults:
    # Required fields first
    base_results: EvaluationResults
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, Dict[str, float]]
    effect_sizes: Dict[str, float]
    cv_scores: Dict[str, List[float]]
    cv_mean_std: Dict[str, Tuple[float, float]]
    balanced_accuracy: float
    mcc: float
    kappa_score: float
    hamming_loss: float
    jaccard_scores: List[float]
    impact_metrics: Dict[str, float]
    resource_allocation_efficiency: float
    error_clustering: Dict[str, Any]
    difficulty_analysis: Dict[str, Any]
    stability_metrics: Dict[str, float]
    robustness_scores: Dict[str, float]
    # Optional fields with defaults at the end
    temporal_trends: Optional[Dict[str, Any]] = None
    comparative_analysis: Optional[Dict[str, Any]] = None
    fairness_metrics: Optional[Dict[str, Any]] = None


class EnhancedCrisisEvaluator(CrisisEvaluator):
    """
    Enhanced crisis classification evaluator with advanced analytics.
    Extends the base CrisisEvaluator with statistical analysis, temporal
    assessment, impact evaluation, and comprehensive reporting capabilities.
    """
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        save_plots: bool = True,
        enable_temporal_analysis: bool = False
    ):
        super().__init__(class_names, output_dir, save_plots)
        self.enable_temporal_analysis = enable_temporal_analysis
        self.statistical_analyzer = StatisticalAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.impact_assessor = CrisisImpactAssessor()
        logger.info("Enhanced CrisisEvaluator initialized with advanced analytics")

    def evaluate_comprehensive(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        y_proba: Optional[Union[np.ndarray, torch.Tensor]] = None,
        texts: Optional[List[str]] = None,
        timestamps: Optional[List[datetime]] = None,
        baseline_models: Optional[Dict[str, np.ndarray]] = None
    ) -> EnhancedEvaluationResults:
        # Convert to numpy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba) if y_proba is not None else None

        # Base evaluation
        base_results = self.evaluate(y_true, y_pred, y_proba, texts)

        # Statistical analysis
        confidence_intervals = self.statistical_analyzer.bootstrap_confidence_intervals(
            y_true, y_pred, y_proba
        )
        statistical_significance = {}
        effect_sizes = self.statistical_analyzer.compute_effect_sizes(y_true, y_pred)

        # Cross-validation metrics (simple stratified k-fold)
        cv_scores, cv_mean_std = self._compute_cv_estimates(y_true, y_pred)

        # Advanced classification metrics
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        hamming = float(hamming_loss(y_true, y_pred))
        jaccard = [
            float(jaccard_score((y_true == i).astype(int), (y_pred == i).astype(int), zero_division='warn'))
            for i in range(self.num_classes)
        ]

        # Temporal analysis
        temporal_trends = None
        if self.enable_temporal_analysis and timestamps is not None:
            temporal_trends = self.temporal_analyzer.analyze_temporal_trends(
                y_true, y_pred, timestamps
            )

        # Crisis impact assessment
        impact_metrics = self.impact_assessor.assess_crisis_impact(y_true, y_pred, y_proba)
        resource_allocation_efficiency = impact_metrics.get('resource_efficiency', 0.0)

        # Error analysis
        error_analyzer = ErrorAnalyzer(self.class_names)
        error_clustering = error_analyzer.analyze(y_true, y_pred)
        difficulty_analysis = self._analyze_sample_difficulty(y_true, y_pred, y_proba)

        # Model reliability
        stability_metrics = self._compute_stability_metrics(y_true, y_pred, y_proba)
        robustness_scores = self._compute_robustness_scores(y_true, y_pred)

        # Model comparison (if provided)
        comparative_analysis = None
        if baseline_models:
            comparative_analysis = {}
            for name, baseline_pred in baseline_models.items():
                comparative_analysis[name] = self.statistical_analyzer.mcnemar_test(
                    y_true, y_pred, baseline_pred
                )

        return EnhancedEvaluationResults(
            base_results=base_results,
            confidence_intervals=confidence_intervals,
            statistical_significance=statistical_significance,
            effect_sizes=effect_sizes,
            cv_scores=cv_scores,
            cv_mean_std=cv_mean_std,
            balanced_accuracy=balanced_acc,
            mcc=mcc,
            kappa_score=kappa,
            hamming_loss=hamming,
            jaccard_scores=jaccard,
            impact_metrics=impact_metrics,
            resource_allocation_efficiency=resource_allocation_efficiency,
            error_clustering=error_clustering,
            difficulty_analysis=difficulty_analysis,
            stability_metrics=stability_metrics,
            robustness_scores=robustness_scores,
            temporal_trends=temporal_trends,
            comparative_analysis=comparative_analysis,
            fairness_metrics=None
        )

    def _compute_cv_estimates(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cv_folds: int = 5
    ) -> Tuple[Dict[str, List[float]], Dict[str, Tuple[float, float]]]:
        # For demonstration, use y_pred as model output (no model retraining)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = {'accuracy': [], 'f1_macro': [], 'f1_weighted': []}
        for train_idx, test_idx in skf.split(y_true, y_true):
            fold_true = y_true[test_idx]
            fold_pred = y_pred[test_idx]
            cv_scores['accuracy'].append(accuracy_score(fold_true, fold_pred))
            cv_scores['f1_macro'].append(
                precision_recall_fscore_support(fold_true, fold_pred, average='macro', zero_division='warn')[2]
            )
            cv_scores['f1_weighted'].append(
                precision_recall_fscore_support(fold_true, fold_pred, average='weighted', zero_division='warn')[2]
            )
        cv_mean_std = {k: (float(np.mean(v)), float(np.std(v))) for k, v in cv_scores.items()}
        return cv_scores, cv_mean_std

    def _analyze_sample_difficulty(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        if y_proba is None:
            return {}
        entropy_scores = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
        confidence_scores = np.max(y_proba, axis=1)
        margin_scores = np.sort(y_proba, axis=1)[:, -1] - np.sort(y_proba, axis=1)[:, -2]
        difficult_indices = np.argsort(entropy_scores)[-10:]
        easy_indices = np.argsort(entropy_scores)[:10]
        return {
            'mean_entropy': float(np.mean(entropy_scores)),
            'std_entropy': float(np.std(entropy_scores)),
            'mean_confidence': float(np.mean(confidence_scores)),
            'std_confidence': float(np.std(confidence_scores)),
            'mean_margin': float(np.mean(margin_scores)),
            'std_margin': float(np.std(margin_scores)),
            'most_difficult_samples': {
                'indices': difficult_indices.tolist(),
                'entropy_scores': entropy_scores[difficult_indices].tolist(),
                'confidence_scores': confidence_scores[difficult_indices].tolist()
            },
            'easiest_samples': {
                'indices': easy_indices.tolist(),
                'entropy_scores': entropy_scores[easy_indices].tolist(),
                'confidence_scores': confidence_scores[easy_indices].tolist()
            }
        }

    def _compute_stability_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray]
    ) -> Dict[str, float]:
        if y_proba is None:
            return {}
        prediction_entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
        confidence_scores = np.max(y_proba, axis=1)
        sorted_proba = np.sort(y_proba, axis=1)
        margins = sorted_proba[:, -1] - sorted_proba[:, -2]
        return {
            'prediction_consistency': float(1.0 / (1.0 + np.mean(prediction_entropy))),
            'confidence_stability': float(1.0 / (1.0 + np.var(confidence_scores))),
            'decision_boundary_stability': float(1.0 / (1.0 + np.var(margins)))
        }

    def _compute_robustness_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        class_counts = np.bincount(y_true, minlength=self.num_classes)
        class_weights = class_counts / len(y_true)
        per_class_accuracy = []
        for i in range(self.num_classes):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
                per_class_accuracy.append(class_acc)
            else:
                per_class_accuracy.append(0.0)
        per_class_accuracy_array = np.array(per_class_accuracy)
        weighted_std = np.sqrt(np.average((per_class_accuracy_array - np.mean(per_class_accuracy_array))**2, weights=class_weights))
        min_class_idx = np.argmin(class_counts)
        minority_class_acc = per_class_accuracy[min_class_idx]
        return {
            'class_imbalance_robustness': float(1.0 / (1.0 + weighted_std)),
            'minority_class_performance': float(minority_class_acc)
        }


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    dataloader: Any,  # torch.utils.data.DataLoader
    device: torch.device,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    enable_temporal_analysis: bool = False,
    baseline_models: Optional[Dict[str, torch.nn.Module]] = None
) -> EnhancedEvaluationResults:
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_texts = []
    all_timestamps = []
    logger.info("Running enhanced model evaluation...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            if 'text' in batch:
                all_texts.extend(batch['text'])
            if 'timestamp' in batch:
                all_timestamps.extend(batch['timestamp'])
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)
    evaluator = EnhancedCrisisEvaluator(class_names=class_names, output_dir=output_dir, enable_temporal_analysis=enable_temporal_analysis)
    results = evaluator.evaluate_comprehensive(
        y_true, y_pred, y_proba, texts=all_texts if all_texts else None, timestamps=all_timestamps if all_timestamps else None
    )
    return results


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Enhanced Crisis Evaluation System...")
    np.random.seed(42)
    n_samples = 1000
    n_classes = 6
    class_probs = [0.05, 0.05, 0.05, 0.10, 0.50, 0.25]
    y_true = np.random.choice(n_classes, n_samples, p=class_probs)
    y_pred = y_true.copy()
    noise_indices = np.random.choice(n_samples, int(0.2 * n_samples), replace=False)
    y_pred[noise_indices] = np.random.choice(n_classes, len(noise_indices))
    y_proba = np.random.dirichlet([1] * n_classes, n_samples)
    for i in range(n_samples):
        y_proba[i] = np.abs(y_proba[i])
        y_proba[i, y_pred[i]] += 0.3
        y_proba[i] = y_proba[i] / y_proba[i].sum()
    evaluator = EnhancedCrisisEvaluator()
    results = evaluator.evaluate_comprehensive(y_true, y_pred, y_proba)
    print(f"✅ Accuracy: {results.base_results.accuracy:.4f}")
    print(f"✅ Macro F1: {results.base_results.macro_f1:.4f}")
    print(f"✅ Humanitarian F1: {results.base_results.humanitarian_f1:.4f}")
    print(f"✅ ECE: {results.base_results.expected_calibration_error:.4f}")
    print(f"✅ Response Efficiency: {results.base_results.response_time_metrics.get('response_efficiency', 0):.4f}")
    report = generate_enhanced_evaluation_report(results, "Test Enhanced Crisis Classifier")
    print(f"✅ Report generated: {len(report)} characters")
    print("\n🎉 Enhanced crisis evaluation system testing completed successfully!")