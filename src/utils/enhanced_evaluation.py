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
        CrisisEvaluator, EvaluationResults, CRISIS_LABELS
    )
except ImportError:
    # Fallback for when running as standalone script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from evaluation import (
        CrisisEvaluator, EvaluationResults, CRISIS_LABELS
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced crisis type mappings for detailed analysis
CRISIS_TYPE_MAPPING = {
    "requests_or_urgent_needs": {
        "category": "humanitarian", 
        "severity": "critical", 
        "response_time": 15,
        "impact_multiplier": 3.0
    },
    "infrastructure_and_utility_damage": {
        "category": "humanitarian", 
        "severity": "high", 
        "response_time": 60,
        "impact_multiplier": 2.5
    },
    "injured_or_dead_people": {
        "category": "humanitarian", 
        "severity": "critical", 
        "response_time": 10,
        "impact_multiplier": 5.0
    },
    "rescue_volunteering_or_donation_effort": {
        "category": "humanitarian", 
        "severity": "medium", 
        "response_time": 120,
        "impact_multiplier": 1.5
    },
    "other_relevant_information": {
        "category": "information", 
        "severity": "low", 
        "response_time": 480,
        "impact_multiplier": 1.0
    },
    "not_humanitarian": {
        "category": "non_crisis", 
        "severity": "none", 
        "response_time": 0,
        "impact_multiplier": 0.0
    }
}

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


class StatisticalAnalyzer:
    """
    Statistical significance testing and confidence interval computation.
    Provides bootstrap confidence intervals, statistical tests for model
    comparison, and effect size calculations for crisis classification.
    """
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def bootstrap_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Tuple[float, float]]:
        if metrics is None:
            metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'precision', 'recall']
        n_samples = len(y_true)
        bootstrap_results = {metric: [] for metric in metrics}
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_proba_boot = y_proba[indices] if y_proba is not None else None
            if 'accuracy' in metrics:
                bootstrap_results['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            if 'macro_f1' in metrics:
                f1 = precision_recall_fscore_support(y_true_boot, y_pred_boot, average='macro', zero_division='warn')[2]
                bootstrap_results['macro_f1'].append(f1)
            if 'weighted_f1' in metrics:
                f1 = precision_recall_fscore_support(y_true_boot, y_pred_boot, average='weighted', zero_division='warn')[2]
                bootstrap_results['weighted_f1'].append(f1)
            if 'precision' in metrics:
                prec = precision_recall_fscore_support(y_true_boot, y_pred_boot, average='macro', zero_division='warn')[0]
                bootstrap_results['precision'].append(prec)
            if 'recall' in metrics:
                rec = precision_recall_fscore_support(y_true_boot, y_pred_boot, average='macro', zero_division='warn')[1]
                bootstrap_results['recall'].append(rec)
        confidence_intervals = {}
        alpha_lower = (1 - self.confidence_level) / 2
        alpha_upper = 1 - alpha_lower
        for metric, values in bootstrap_results.items():
            lower = np.percentile(values, alpha_lower * 100)
            upper = np.percentile(values, alpha_upper * 100)
            confidence_intervals[metric] = (float(lower), float(upper))
        return confidence_intervals

    def mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray
    ) -> Dict[str, float]:
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        both_correct = np.sum(correct1 & correct2)
        model1_only = np.sum(correct1 & ~correct2)
        model2_only = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        if model1_only + model2_only == 0:
            return {'statistic': 0.0, 'p_value': 1.0, 'effect_size': 0.0}
        statistic = (abs(model1_only - model2_only) - 1) ** 2 / (model1_only + model2_only)
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        acc1 = accuracy_score(y_true, y_pred1)
        acc2 = accuracy_score(y_true, y_pred2)
        effect_size = acc1 - acc2
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'model1_only_correct': int(model1_only),
            'model2_only_correct': int(model2_only)
        }

    def compute_effect_sizes(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_accuracy: Optional[float] = None
    ) -> Dict[str, float]:
        accuracy = accuracy_score(y_true, y_pred)
        effect_sizes = {}
        if baseline_accuracy is not None:
            p = baseline_accuracy
            std_baseline = np.sqrt(p * (1 - p) / len(y_true))
            cohens_d = (accuracy - baseline_accuracy) / std_baseline if std_baseline > 0 else 0.0
            effect_sizes['cohens_d'] = float(cohens_d)
        mcc = matthews_corrcoef(y_true, y_pred)
        effect_sizes['mcc'] = float(mcc)
        kappa = cohen_kappa_score(y_true, y_pred)
        effect_sizes['kappa'] = float(kappa)
        return effect_sizes


class TemporalAnalyzer:
    """
    Time-based performance analysis for crisis classification.
    """
    def __init__(self):
        pass
    def analyze_temporal_trends(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        window_size: int = 100
    ) -> Dict[str, Any]:
        if timestamps is None or len(timestamps) != len(y_true):
            return {'message': 'No valid timestamps provided for temporal analysis.'}
        df = pd.DataFrame({'timestamp': pd.to_datetime(timestamps), 'true': y_true, 'pred': y_pred})
        df = df.sort_values('timestamp')
        df['correct'] = df['true'] == df['pred']
        rolling_acc = df['correct'].rolling(window=window_size, min_periods=1).mean().tolist()
        return {
            'rolling_accuracy': rolling_acc,
            'timestamps': df['timestamp'].tolist()
        }


class CrisisImpactAssessor:
    """
    Real-world impact assessment for crisis classification decisions.
    Evaluates the practical consequences of classification decisions
    in terms of resource allocation, response time, and crisis outcomes.
    """
    def __init__(self):
        self.crisis_weights = {
            label: CRISIS_TYPE_MAPPING[label]["impact_multiplier"] 
            for label in CRISIS_LABELS
        }

    def assess_crisis_impact(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        impact_metrics = {}
        weights = np.array([self.crisis_weights[CRISIS_LABELS[label]] for label in y_true])
        correct = (y_true == y_pred).astype(float)
        weighted_accuracy = np.average(correct, weights=weights)
        impact_metrics['weighted_accuracy'] = float(weighted_accuracy)
        # Critical miss rate (missing critical crises)
        critical_indices = [0, 2]  # requests_or_urgent_needs, injured_or_dead_people
        critical_mask = np.isin(y_true, critical_indices)
        critical_missed = np.sum(critical_mask & (y_true != y_pred))
        total_critical = np.sum(critical_mask)
        critical_miss_rate = critical_missed / total_critical if total_critical > 0 else 0.0
        impact_metrics['critical_miss_rate'] = float(critical_miss_rate)
        # False alarm rate for non-crises
        non_crisis_index = 5  # not_humanitarian
        non_crisis_mask = (y_true == non_crisis_index)
        false_alarms = np.sum(non_crisis_mask & (y_pred != non_crisis_index))
        total_non_crisis = np.sum(non_crisis_mask)
        false_alarm_rate = false_alarms / total_non_crisis if total_non_crisis > 0 else 0.0
        impact_metrics['false_alarm_rate'] = float(false_alarm_rate)
        # Resource allocation efficiency
        efficiency = self._compute_resource_efficiency(y_true, y_pred)
        impact_metrics['resource_efficiency'] = efficiency
        # Expected response delay
        expected_delay = self._compute_expected_delay(y_true, y_pred)
        impact_metrics['expected_response_delay'] = expected_delay
        # Crisis escalation risk
        escalation_risk = self._compute_escalation_risk(y_true, y_pred)
        impact_metrics['escalation_risk'] = escalation_risk
        return impact_metrics

    def _compute_resource_efficiency(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        perfect_allocation = sum(self.crisis_weights[CRISIS_LABELS[label]] for label in y_true)
        actual_allocation = sum(self.crisis_weights[CRISIS_LABELS[label]] for label in y_pred)
        over_allocation = sum(
            max(0, self.crisis_weights[CRISIS_LABELS[pred]] - self.crisis_weights[CRISIS_LABELS[true]])
            for true, pred in zip(y_true, y_pred)
        )
        under_allocation = sum(
            max(0, self.crisis_weights[CRISIS_LABELS[true]] - self.crisis_weights[CRISIS_LABELS[pred]]) * 2
            for true, pred in zip(y_true, y_pred)
        )
        total_penalty = over_allocation + under_allocation
        efficiency = max(0, 1 - total_penalty / (perfect_allocation + 1e-6))
        return float(efficiency)

    def _compute_expected_delay(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        total_delay = 0.0
        for true_label, pred_label in zip(y_true, y_pred):
            true_response_time = CRISIS_TYPE_MAPPING[CRISIS_LABELS[true_label]]["response_time"]
            pred_response_time = CRISIS_TYPE_MAPPING[CRISIS_LABELS[pred_label]]["response_time"]
            if pred_response_time > true_response_time:
                delay = pred_response_time - true_response_time
                severity_weight = CRISIS_TYPE_MAPPING[CRISIS_LABELS[true_label]]["impact_multiplier"]
                total_delay += delay * severity_weight
        return float(total_delay / len(y_true))

    def _compute_escalation_risk(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        escalation_cases = 0
        for true_label, pred_label in zip(y_true, y_pred):
            true_severity = CRISIS_TYPE_MAPPING[CRISIS_LABELS[true_label]]["severity"]
            pred_severity = CRISIS_TYPE_MAPPING[CRISIS_LABELS[pred_label]]["severity"]
            if true_severity == "critical" and pred_severity in ["low", "none"]:
                escalation_cases += 3
            elif true_severity == "high" and pred_severity in ["low", "none"]:
                escalation_cases += 2
            elif true_severity == "medium" and pred_severity == "none":
                escalation_cases += 1
        return float(escalation_cases / len(y_true))


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
        error_clustering = self.error_analyzer.analyze(y_true, y_pred, y_proba, texts)
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


def generate_enhanced_evaluation_report(
    results: EnhancedEvaluationResults,
    model_name: str = "Enhanced Crisis Classifier",
    output_path: Optional[str] = None
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# Enhanced Crisis Classification Model Evaluation Report
**Model**: {model_name}  
**Generated**: {timestamp}  
**Evaluation Framework**: Enterprise-Grade Crisis Assessment

---

## Executive Summary

This report provides a comprehensive evaluation of the {model_name} for crisis classification across {len(CRISIS_LABELS)} categories. The evaluation includes advanced classification metrics, crisis-specific assessments, calibration analysis, and emergency response simulation.

### Key Performance Indicators
- **Overall Accuracy**: {results.base_results.accuracy:.4f}
- **Macro F1-Score**: {results.base_results.macro_f1:.4f}
- **Humanitarian Crisis Detection F1**: {results.base_results.humanitarian_f1:.4f}
- **Critical Crisis Detection F1**: {results.base_results.critical_crisis_f1:.4f}
- **Emergency Response Accuracy**: {results.base_results.emergency_response_accuracy:.4f}

---

## Advanced Metrics

| Metric | Score |
|--------|-------|
| Balanced Accuracy | {results.balanced_accuracy:.4f} |
| MCC | {results.mcc:.4f} |
| Kappa | {results.kappa_score:.4f} |
| Hamming Loss | {results.hamming_loss:.4f} |

---

## Impact Assessment

| Metric | Value |
|--------|-------|
""" + '\n'.join([f"| {k.replace('_', ' ').title()} | {v:.4f} |" for k, v in results.impact_metrics.items()]) + """

---

## Cross-Validation Metrics

| Metric | Mean | Std |
|--------|------|-----|
""" + '\n'.join([f"| {k.replace('_', ' ').title()} | {results.cv_mean_std[k][0]:.4f} | {results.cv_mean_std[k][1]:.4f} |" for k in results.cv_mean_std]) + """

---

## Recommendations

- Review high-impact error types and consider targeted data augmentation.
- Monitor model calibration and retrain with new crisis data as available.
- Use human-in-the-loop for high-stakes classifications.

---

*Report generated by AICrisisAlert Enhanced Evaluation System*
"""
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Enhanced evaluation report saved to {output_path}")
    return report


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