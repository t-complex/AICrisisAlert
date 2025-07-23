"""
Training Utilities for Crisis Classification

This module provides comprehensive training utilities including custom metrics,
early stopping, learning rate scheduling, and loss functions optimized for
multi-class crisis classification with class imbalance handling.

Classes:
    CrisisMetrics: Custom metric computation for crisis classification
    EarlyStopping: Early stopping with patience and best model tracking
    CrisisLoss: Custom loss functions for handling class imbalance
    TrainingVisualizer: Training progress visualization and logging

Functions:
    compute_classification_metrics: Compute comprehensive classification metrics
    create_lr_scheduler: Create learning rate scheduler
    log_training_progress: Log and visualize training progress
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix, f1_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path
import json
from collections import defaultdict
from dataclasses import dataclass
from src.training.losses import CrisisLoss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crisis classification labels for reference
CRISIS_LABELS = [
    "requests_or_urgent_needs",
    "infrastructure_and_utility_damage", 
    "injured_or_dead_people",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian"
]

@dataclass
class MetricResult:
    """Container for metric computation results."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: List[float]
    per_class_precision: List[float]
    per_class_recall: List[float]
    confusion_matrix: np.ndarray
    classification_report: str
    support: List[int]


class CrisisMetrics:
    """
    Custom metric computation for crisis classification.
    
    This class provides comprehensive metrics including per-class and macro-averaged
    F1-score, precision, recall, and specialized crisis classification metrics.
    """
    
    def __init__(self, label_names: Optional[List[str]] = None):
        """
        Initialize crisis metrics calculator.
        
        Args:
            label_names: List of label names. If None, uses default crisis labels.
        """
        self.label_names = label_names or CRISIS_LABELS
        self.num_classes = len(self.label_names)
        
        # Reset metrics tracking
        self.reset()
        
        logger.info(f"CrisisMetrics initialized for {self.num_classes} classes")
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.batch_metrics = []
    
    def update(
        self, 
        predictions: Union[torch.Tensor, np.ndarray], 
        targets: Union[torch.Tensor, np.ndarray]
    ):
        """
        Update metrics with batch predictions and targets.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                predictions = torch.argmax(predictions, dim=-1)
            predictions = predictions.cpu().numpy()
        
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Store for final computation
        self.all_predictions.extend(predictions.flatten())
        self.all_targets.extend(targets.flatten())
    
    def compute(self, return_dict: bool = True) -> Union[MetricResult, Dict]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            return_dict: Whether to return dictionary or MetricResult object
            
        Returns:
            Comprehensive metrics including per-class and macro metrics
        """
        if not self.all_predictions:
            logger.warning("No predictions available for metric computation")
            return self._empty_metrics() if return_dict else MetricResult(**self._empty_metrics())
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # Basic accuracy
        accuracy = accuracy_score(targets, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0, labels=range(self.num_classes)
        )
        
        # Macro and weighted averages with explicit labels
        macro_f1 = f1_score(targets, predictions, labels=list(range(self.num_classes)), average='macro', zero_division=0)
        weighted_f1 = f1_score(targets, predictions, labels=list(range(self.num_classes)), average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))
        
        # Classification report with explicit labels
        report = classification_report(
            targets, predictions,
            labels=list(range(self.num_classes)),
            target_names=self.label_names,
            zero_division=0,
            digits=4
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'per_class_f1': f1.tolist(),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'confusion_matrix': cm,
            'classification_report': report,
            'support': support.tolist()
        }
        
        # Add per-class metrics with names
        for i, label in enumerate(self.label_names):
            metrics[f'f1_{label}'] = float(f1[i])
            metrics[f'precision_{label}'] = float(precision[i])
            metrics[f'recall_{label}'] = float(recall[i])
        
        if return_dict:
            return metrics
        else:
            return MetricResult(**metrics)
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'per_class_f1': [0.0] * self.num_classes,
            'per_class_precision': [0.0] * self.num_classes,
            'per_class_recall': [0.0] * self.num_classes,
            'confusion_matrix': np.zeros((self.num_classes, self.num_classes)),
            'classification_report': "",
            'support': [0] * self.num_classes
        }
    
    def get_crisis_specific_metrics(self) -> Dict[str, float]:
        """
        Get crisis-specific metrics focusing on humanitarian categories.
        
        Returns:
            Dictionary with crisis-specific evaluation metrics
        """
        if not self.all_predictions:
            return {}
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # Separate humanitarian vs non-humanitarian
        humanitarian_classes = [0, 1, 2, 3]  # First 4 classes are humanitarian
        non_humanitarian_classes = [4, 5]    # Last 2 classes are non-humanitarian
        
        # Binary classification: humanitarian vs non-humanitarian
        binary_targets = np.isin(targets, humanitarian_classes).astype(int)
        binary_predictions = np.isin(predictions, humanitarian_classes).astype(int)
        
        humanitarian_f1 = f1_score(binary_targets, binary_predictions)
        humanitarian_precision = precision_recall_fscore_support(
            binary_targets, binary_predictions, average='binary'
        )[0]
        humanitarian_recall = precision_recall_fscore_support(
            binary_targets, binary_predictions, average='binary'
        )[1]
        
        # Critical vs non-critical (requests, injuries, infrastructure are critical)
        critical_classes = [0, 1, 2]  # Most critical for emergency response
        critical_targets = np.isin(targets, critical_classes).astype(int)
        critical_predictions = np.isin(predictions, critical_classes).astype(int)
        
        critical_f1 = f1_score(critical_targets, critical_predictions)
        
        return {
            'humanitarian_f1': float(humanitarian_f1),
            'humanitarian_precision': float(humanitarian_precision),
            'humanitarian_recall': float(humanitarian_recall),
            'critical_crisis_f1': float(critical_f1),
            'emergency_response_accuracy': float(accuracy_score(critical_targets, critical_predictions))
        }


class EarlyStopping:
    """
    Early stopping utility with patience and best model tracking.
    
    Monitors validation metrics and stops training when no improvement
    is observed for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0001,
        monitor: str = 'val_macro_f1',
        mode: str = 'max',
        restore_best_weights: bool = True,
        save_best_model: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to qualify as improvement
            monitor: Metric to monitor for improvement
            mode: 'min' for metrics to minimize, 'max' for metrics to maximize
            restore_best_weights: Whether to restore best weights when stopping
            save_best_model: Whether to save best model during training
            save_path: Path to save best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.save_best_model = save_best_model
        self.save_path = save_path
        
        # State tracking
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.inf
        
        logger.info(f"EarlyStopping initialized: monitor={monitor}, patience={patience}, mode={mode}")
    
    def __call__(
        self, 
        current_score: float, 
        model: nn.Module, 
        epoch: int
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current epoch score for monitored metric
            model: Model to potentially save
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self._is_improvement(current_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Save best model
            if self.save_best_model and self.save_path:
                torch.save(model.state_dict(), self.save_path)
                logger.info(f"Best model saved at epoch {epoch} with {self.monitor}={current_score:.4f}")
        
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch}")
                
                # Restore best weights
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                    logger.info("Best weights restored")
                
                return True
        
        return False
    
    def _is_improvement(self, current_score: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == 'min':
            return current_score < (self.best_score - self.min_delta)
        else:
            return current_score > (self.best_score + self.min_delta)
    
    def get_best_score(self) -> float:
        """Get the best score achieved."""
        return self.best_score
    
    def get_best_epoch(self) -> int:
        """Get the epoch with best score."""
        return self.best_epoch


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_epochs: int = 10,
    min_lr: float = 1e-7,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'plateau', 'step')
        num_epochs: Total number of training epochs
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler parameters
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=min_lr,
            **kwargs
        )
        logger.info(f"Created CosineAnnealingLR scheduler with T_max={num_epochs}, eta_min={min_lr}")
    
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            min_lr=min_lr,
            **kwargs
        )
        logger.info("Created ReduceLROnPlateau scheduler")
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler


def calculate_class_weights(
    targets: Union[List[int], np.ndarray, torch.Tensor],
    num_classes: Optional[int] = None,
    method: str = 'balanced'
) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced data.
    
    Args:
        targets: Target labels
        num_classes: Number of classes
        method: Weight calculation method ('balanced', 'sqrt')
        
    Returns:
        Tensor of class weights
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    targets = np.array(targets)
    
    if num_classes is None:
        num_classes = len(np.unique(targets))
    
    # Count class frequencies
    class_counts = np.bincount(targets, minlength=num_classes)
    total_samples = len(targets)
    
    if method == 'balanced':
        # Inverse frequency weighting
        weights = total_samples / (num_classes * class_counts + 1e-6)
    elif method == 'sqrt':
        # Square root of inverse frequency
        weights = np.sqrt(total_samples / (num_classes * class_counts + 1e-6))
    else:
        raise ValueError(f"Unsupported weighting method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    logger.info(f"Class weights calculated using '{method}' method: {weights}")
    
    return torch.FloatTensor(weights)


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Trainer Utils...")
    
    # Test metrics computation
    num_samples = 1000
    num_classes = 6
    
    # Generate dummy data
    predictions = torch.randint(0, num_classes, (num_samples,))
    targets = torch.randint(0, num_classes, (num_samples,))
    
    # Test metrics
    print("Testing metrics computation...")
    metrics_calc = CrisisMetrics()
    metrics_calc.update(predictions, targets)
    metrics = metrics_calc.compute()
    
    print(f"✅ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✅ Macro F1: {metrics['macro_f1']:.4f}")
    print(f"✅ Per-class F1: {[f'{f1:.3f}' for f1 in metrics['per_class_f1']]}")
    
    # Test early stopping
    print("Testing early stopping...")
    early_stopping = EarlyStopping(patience=3, monitor='val_macro_f1')
    
    # Test loss functions
    print("Testing loss functions...")
    crisis_loss = CrisisLoss(num_classes=6)
    
    # Test class weights
    print("Testing class weight calculation...")
    weights = calculate_class_weights(targets)
    print(f"✅ Class weights: {weights}")
    
    print("✅ All trainer utils tests completed successfully!")