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


class CrisisLoss:
    """
    Custom loss functions for handling class imbalance in crisis classification.
    
    Provides various loss functions including weighted cross-entropy, focal loss,
    and label smoothing optimized for crisis classification scenarios.
    """
    
    def __init__(self, num_classes: int = 6, device: Optional[torch.device] = None):
        """
        Initialize crisis loss functions.
        
        Args:
            num_classes: Number of classes
            device: Device for computations
        """
        self.num_classes = num_classes
        self.device = device or torch.device('cpu')
        logger.info(f"CrisisLoss initialized for {num_classes} classes")
    
    def weighted_cross_entropy(
        self, 
        class_weights: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """
        Create weighted cross-entropy loss.
        
        Args:
            class_weights: Weights for each class. If None, uniform weights are used.
            
        Returns:
            Weighted CrossEntropyLoss
        """
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            logger.info(f"Using class weights: {class_weights.tolist()}")
        
        return nn.CrossEntropyLoss(weight=class_weights)
    
    def focal_loss(
        self, 
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ) -> 'FocalLoss':
        """
        Create focal loss for handling hard examples.
        
        Args:
            alpha: Weighting factor for classes
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
            
        Returns:
            FocalLoss instance
        """
        return FocalLoss(
            alpha=alpha,
            gamma=gamma,
            num_classes=self.num_classes,
            reduction=reduction,
            device=self.device
        )
    
    def label_smoothing_loss(
        self, 
        smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ) -> 'LabelSmoothingLoss':
        """
        Create label smoothing loss.
        
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform)
            class_weights: Optional class weights
            
        Returns:
            LabelSmoothingLoss instance
        """
        return LabelSmoothingLoss(
            num_classes=self.num_classes,
            smoothing=smoothing,
            class_weights=class_weights,
            device=self.device
        )


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        num_classes: int = 6,
        reduction: str = 'mean',
        device: Optional[torch.device] = None
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        self.device = device or torch.device('cpu')
        
        if alpha is not None:
            self.alpha = alpha.to(self.device)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss implementation."""
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.device = device or torch.device('cpu')
        self.class_weights = class_weights
        
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smooth labels
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute loss
        loss = -smooth_labels * log_probs
        
        if self.class_weights is not None:
            # Apply class weights
            weight_mask = self.class_weights[targets]
            loss = loss.sum(dim=-1) * weight_mask
            return loss.mean()
        else:
            return loss.sum(dim=-1).mean()


class TrainingVisualizer:
    """
    Training progress visualization and logging utilities.
    
    Provides real-time visualization of training metrics, loss curves,
    and comprehensive logging for crisis classification training.
    """
    
    def __init__(self, save_dir: Optional[str] = None, log_level: str = 'INFO'):
        """
        Initialize training visualizer.
        
        Args:
            save_dir: Directory to save plots and logs
            log_level: Logging level
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.epochs = []
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.visualizer")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Style configuration
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info("TrainingVisualizer initialized")
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        epoch_time: float
    ):
        """
        Log metrics for one epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            learning_rate: Current learning rate
            epoch_time: Time taken for epoch
        """
        self.epochs.append(epoch)
        
        # Store metrics
        for key, value in train_metrics.items():
            self.train_history[key].append(value)
        
        for key, value in val_metrics.items():
            self.val_history[key].append(value)
        
        # Add learning rate and time
        self.train_history['learning_rate'].append(learning_rate)
        self.train_history['epoch_time'].append(epoch_time)
        
        # Log to console
        self._log_epoch_summary(epoch, train_metrics, val_metrics, learning_rate, epoch_time)
        
        # Save plots periodically
        if epoch % 5 == 0 or epoch == 1:
            self.plot_training_curves()
    
    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        epoch_time: float
    ):
        """Log epoch summary to console."""
        self.logger.info(f"\nEpoch {epoch} Summary:")
        self.logger.info(f"  Time: {epoch_time:.2f}s, LR: {learning_rate:.2e}")
        
        # Key training metrics
        train_loss = train_metrics.get('loss', 0)
        train_acc = train_metrics.get('accuracy', 0)
        train_f1 = train_metrics.get('macro_f1', 0)
        
        val_loss = val_metrics.get('loss', 0)
        val_acc = val_metrics.get('accuracy', 0)
        val_f1 = val_metrics.get('macro_f1', 0)
        
        self.logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        self.logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Crisis-specific metrics if available
        humanitarian_f1 = val_metrics.get('humanitarian_f1')
        if humanitarian_f1:
            self.logger.info(f"  Humanitarian F1: {humanitarian_f1:.4f}")
    
    def plot_training_curves(self, show: bool = False):
        """
        Plot comprehensive training curves.
        
        Args:
            show: Whether to display plots
        """
        if not self.epochs:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Crisis Classification Training Progress', fontsize=16)
        
        # Loss curves
        if 'loss' in self.train_history:
            axes[0, 0].plot(self.epochs, self.train_history['loss'], label='Train', linewidth=2)
            axes[0, 0].plot(self.epochs, self.val_history.get('loss', []), label='Validation', linewidth=2)
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        if 'accuracy' in self.train_history:
            axes[0, 1].plot(self.epochs, self.train_history['accuracy'], label='Train', linewidth=2)
            axes[0, 1].plot(self.epochs, self.val_history.get('accuracy', []), label='Validation', linewidth=2)
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score curves
        if 'macro_f1' in self.train_history:
            axes[0, 2].plot(self.epochs, self.train_history['macro_f1'], label='Train Macro F1', linewidth=2)
            axes[0, 2].plot(self.epochs, self.val_history.get('macro_f1', []), label='Val Macro F1', linewidth=2)
            if 'weighted_f1' in self.val_history:
                axes[0, 2].plot(self.epochs, self.val_history['weighted_f1'], label='Val Weighted F1', linewidth=2, linestyle='--')
            axes[0, 2].set_title('F1 Scores')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('F1 Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rate' in self.train_history:
            axes[1, 0].plot(self.epochs, self.train_history['learning_rate'], linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Crisis-specific metrics
        humanitarian_metrics = ['humanitarian_f1', 'critical_crisis_f1']
        crisis_data = []
        crisis_labels = []
        
        for metric in humanitarian_metrics:
            if metric in self.val_history:
                crisis_data.append(self.val_history[metric])
                crisis_labels.append(metric.replace('_', ' ').title())
        
        if crisis_data:
            for i, (data, label) in enumerate(zip(crisis_data, crisis_labels)):
                axes[1, 1].plot(self.epochs, data, label=label, linewidth=2)
            axes[1, 1].set_title('Crisis-Specific Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Epoch time
        if 'epoch_time' in self.train_history:
            axes[1, 2].plot(self.epochs, self.train_history['epoch_time'], linewidth=2, color='red')
            axes[1, 2].set_title('Training Time per Epoch')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Time (seconds)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if self.save_dir:
            plot_path = self.save_dir / 'training_curves.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Training curves saved to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        epoch: Optional[int] = None,
        show: bool = False
    ):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Class names for labels
            epoch: Current epoch number
            show: Whether to display plot
        """
        class_names = class_names or CRISIS_LABELS
        
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=[name.replace('_', '\n') for name in class_names],
            yticklabels=[name.replace('_', '\n') for name in class_names],
            cbar_kws={'label': 'Normalized Frequency'}
        )
        
        title = 'Confusion Matrix'
        if epoch is not None:
            title += f' - Epoch {epoch}'
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        if self.save_dir:
            filename = f'confusion_matrix_epoch_{epoch}.png' if epoch else 'confusion_matrix.png'
            plot_path = self.save_dir / filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Confusion matrix saved to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_training_history(self):
        """Save training history to JSON file."""
        if not self.save_dir:
            return
        
        history = {
            'epochs': self.epochs,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for split_history in [history['train_history'], history['val_history']]:
            for key, values in split_history.items():
                if isinstance(values, (list, tuple)):
                    split_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
        
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")


def compute_classification_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    label_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels
        label_names: Names of classes
        
    Returns:
        Dictionary with comprehensive metrics
    """
    metrics_calculator = CrisisMetrics(label_names)
    metrics_calculator.update(predictions, targets)
    
    # Get standard metrics
    metrics = metrics_calculator.compute(return_dict=True)
    
    # Add crisis-specific metrics
    crisis_metrics = metrics_calculator.get_crisis_specific_metrics()
    metrics.update(crisis_metrics)
    
    return metrics


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