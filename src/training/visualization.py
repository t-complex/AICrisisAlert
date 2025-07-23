import logging
from typing import Any, Dict, List, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TrainingVisualizer:
    def __init__(self, save_dir: str, log_level: str = 'INFO'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('TrainingVisualizer')
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.history: List[Dict[str, Any]] = []

    def log_epoch(self, epoch: int, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any], learning_rate: float, epoch_time: float):
        entry = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': learning_rate,
            'epoch_time': epoch_time
        }
        self.history.append(entry)
        self.logger.info(f"Epoch {epoch}: {entry}")

    def save_training_history(self):
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved to {history_path}")

def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None, title: str = 'Confusion Matrix', save_path: Optional[str] = None, show: bool = False):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def plot_roc_curves(y_true, y_proba, class_names: Optional[List[str]] = None, save_path: Optional[str] = None, show: bool = False):
    from sklearn.metrics import roc_curve, auc
    n_classes = y_proba.shape[1]
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        label = class_names[i] if class_names else f'Class {i}'
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def plot_precision_recall_curves(y_true, y_proba, class_names: Optional[List[str]] = None, save_path: Optional[str] = None, show: bool = False):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    n_classes = y_proba.shape[1]
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_proba[:, i])
        ap = average_precision_score(y_true == i, y_proba[:, i])
        label = class_names[i] if class_names else f'Class {i}'
        plt.plot(recall, precision, label=f'{label} (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def plot_calibration_curve(y_true, y_proba, n_bins: int = 10, save_path: Optional[str] = None, show: bool = False):
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close() 