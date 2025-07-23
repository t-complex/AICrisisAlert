import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report, confusion_matrix
from typing import Dict, List, Any

def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division='warn'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division='warn'
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

def get_classification_report(y_true: List[int], y_pred: List[int], target_names: List[str]) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, zero_division='warn')

def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred) 