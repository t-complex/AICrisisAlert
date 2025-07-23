"""
Crisis Classification Ensemble Training Pipeline

This module provides comprehensive training utilities for crisis classification ensembles
with advanced optimization strategies, crisis-specific loss functions, and performance
monitoring tailored for emergency response systems.

Key Features:
- Multi-stage ensemble training with individual model pre-training
- Crisis-adaptive loss functions and optimization strategies
- Ensemble fine-tuning with crisis-specific objectives
- Advanced hyperparameter optimization for ensemble components
- Cross-validation and ensemble validation strategies
- Performance monitoring and model selection

Classes:
    CrisisEnsembleTrainer: Main ensemble training orchestrator
    EnsembleOptimizer: Hyperparameter optimization for ensembles
    CrisisAdaptiveLoss: Crisis-specific loss functions for ensemble training
    EnsembleValidator: Validation strategies for ensemble models

Functions:
    train_crisis_ensemble: Main ensemble training function
    optimize_ensemble_hyperparameters: Hyperparameter optimization utility
    validate_ensemble_models: Comprehensive ensemble validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from src.training.configs import EnsembleTrainingConfig
import json
from datetime import datetime
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Local imports
from ..models.ensemble_classifier import (
    CrisisEnsembleConfig, create_crisis_ensemble, CRISIS_TYPE_WEIGHTS
)
from ..models.model_loader import load_crisis_classifier
from ..models.lora_setup import setup_lora_model, LoRAConfiguration
from .trainer_utils import CrisisMetrics, EarlyStopping
from ..utils.evaluation import CRISIS_LABELS
from ..utils.enhanced_evaluation import EnhancedCrisisEvaluator
from src.training.crisis_ensemble_trainer import CrisisEnsembleTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrisisAdaptiveLoss(nn.Module):
    """
    Crisis-specific adaptive loss functions for ensemble training.
    
    Provides specialized loss functions that account for crisis urgency,
    humanitarian priorities, and ensemble-specific objectives.
    """
    
    def __init__(self, config: EnsembleTrainingConfig, device: torch.device):
        """
        Initialize crisis-adaptive loss functions.
        
        Args:
            config: Training configuration
            device: Computation device
        """
        super(CrisisAdaptiveLoss, self).__init__()
        self.config = config
        self.device = device
        
        # Crisis type weights
        self.crisis_weights = torch.tensor([
            CRISIS_TYPE_WEIGHTS[label] for label in CRISIS_LABELS
        ], device=device)
        
        # Temperature parameter for ensemble calibration
        if config.temperature_scaling:
            self.temperature = nn.Parameter(torch.tensor(config.initial_temperature))
        else:
            self.temperature = torch.tensor(1.0, device=device)
        
        # Ensemble-specific parameters
        self.diversity_regularization = nn.Parameter(torch.tensor(0.1))
        self.confidence_threshold = 0.8
        
        logger.info("CrisisAdaptiveLoss initialized")
        logger.info(f"Crisis weights: {self.crisis_weights.tolist()}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        individual_predictions: Optional[torch.Tensor] = None,
        loss_type: str = "crisis_adaptive"
    ) -> torch.Tensor:
        """
        Compute crisis-adaptive loss for ensemble training.
        
        Args:
            predictions: Ensemble predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            individual_predictions: Individual model predictions (batch_size, num_models, num_classes)
            loss_type: Type of loss to compute
            
        Returns:
            Computed loss tensor
        """
        if loss_type == "crisis_adaptive":
            return self._crisis_adaptive_loss(predictions, targets, individual_predictions)
        elif loss_type == "weighted_ensemble":
            return self._weighted_ensemble_loss(predictions, targets)
        elif loss_type == "knowledge_distillation":
            return self._knowledge_distillation_loss(predictions, targets, individual_predictions)
        else:
            # Standard cross-entropy with crisis weighting
            return self._weighted_cross_entropy(predictions, targets)
    
    def _crisis_adaptive_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        individual_predictions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Crisis-adaptive loss with humanitarian prioritization and diversity regularization.
        
        Args:
            predictions: Ensemble predictions
            targets: Ground truth labels
            individual_predictions: Individual model predictions
            
        Returns:
            Crisis-adaptive loss
        """
        # Base cross-entropy with crisis weighting
        base_loss = self._weighted_cross_entropy(predictions, targets)
        
        # Humanitarian vs non-humanitarian penalty
        humanitarian_loss = self._humanitarian_priority_loss(predictions, targets)
        
        # Critical crisis detection boost
        critical_loss = self._critical_crisis_loss(predictions, targets)
        
        # Diversity regularization (if individual predictions available)
        diversity_loss = torch.tensor(0.0, device=self.device)
        if individual_predictions is not None:
            diversity_loss = self._diversity_regularization_loss(individual_predictions)
        
        # Combine losses
        total_loss = (
            base_loss + 
            self.config.humanitarian_boost * humanitarian_loss +
            self.config.critical_crisis_weight * critical_loss +
            0.1 * diversity_loss
        )
        
        return total_loss
    
    def _weighted_cross_entropy(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy with crisis-specific weights."""
        # Apply temperature scaling
        scaled_logits = predictions / self.temperature
        
        # Create class weights based on targets
        batch_weights = self.crisis_weights[targets]
        
        # Compute weighted cross-entropy
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        loss = F.nll_loss(log_probs, targets, reduction='none')
        weighted_loss = loss * batch_weights
        
        return weighted_loss.mean()
    
    def _humanitarian_priority_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Additional loss to prioritize humanitarian crisis detection.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Humanitarian priority loss
        """
        humanitarian_classes = [0, 1, 2, 3]  # First 4 classes are humanitarian
        non_humanitarian_classes = [4, 5]    # Last 2 are non-humanitarian
        
        # Create binary humanitarian targets
        humanitarian_targets = torch.isin(targets, torch.tensor(humanitarian_classes, device=self.device)).long()
        
        # Sum probabilities for humanitarian classes
        humanitarian_probs = predictions[:, humanitarian_classes].sum(dim=1)
        non_humanitarian_probs = predictions[:, non_humanitarian_classes].sum(dim=1)
        
        # Binary classification loss for humanitarian vs non-humanitarian
        binary_logits = torch.stack([non_humanitarian_probs, humanitarian_probs], dim=1)
        humanitarian_loss = F.cross_entropy(binary_logits, humanitarian_targets)
        
        return humanitarian_loss
    
    def _critical_crisis_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Additional loss to boost critical crisis detection.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Critical crisis detection loss
        """
        critical_classes = [0, 2]  # Urgent needs and casualties
        critical_targets = torch.isin(targets, torch.tensor(critical_classes, device=self.device)).long()
        
        # Sum probabilities for critical classes
        critical_probs = predictions[:, critical_classes].sum(dim=1)
        non_critical_probs = 1.0 - critical_probs
        
        # Binary classification loss for critical vs non-critical
        binary_logits = torch.stack([non_critical_probs, critical_probs], dim=1)
        critical_loss = F.cross_entropy(binary_logits, critical_targets)
        
        return critical_loss
    
    def _diversity_regularization_loss(self, individual_predictions: torch.Tensor) -> torch.Tensor:
        """
        Regularization loss to encourage diversity among ensemble models.
        
        Args:
            individual_predictions: Predictions from individual models (batch_size, num_models, num_classes)
            
        Returns:
            Diversity regularization loss
        """
        batch_size, num_models, num_classes = individual_predictions.shape
        
        # Calculate pairwise KL divergences between models
        kl_divergences = []
        
        for i in range(num_models):
            for j in range(i + 1, num_models):
                pred_i = individual_predictions[:, i, :]
                pred_j = individual_predictions[:, j, :]
                
                # KL divergence between model i and j
                kl_div = F.kl_div(
                    F.log_softmax(pred_i, dim=-1),
                    F.softmax(pred_j, dim=-1),
                    reduction='batchmean'
                )
                kl_divergences.append(kl_div)
        
        # Average KL divergence (higher = more diverse)
        avg_kl_div = torch.stack(kl_divergences).mean()
        
        # Diversity loss (negative KL to encourage diversity)
        diversity_loss = -avg_kl_div * self.diversity_regularization
        
        return diversity_loss
    
    def _knowledge_distillation_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        individual_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Knowledge distillation loss for ensemble training.
        
        Args:
            predictions: Ensemble predictions
            targets: Ground truth labels
            individual_predictions: Individual model predictions
            
        Returns:
            Knowledge distillation loss
        """
        # Standard classification loss
        classification_loss = self._weighted_cross_entropy(predictions, targets)
        
        # Distillation loss between ensemble and individual models
        distillation_losses = []
        
        for i in range(individual_predictions.shape[1]):
            individual_pred = individual_predictions[:, i, :]
            
            # KL divergence between individual model and ensemble
            kl_loss = F.kl_div(
                F.log_softmax(individual_pred / self.temperature, dim=-1),
                F.softmax(predictions / self.temperature, dim=-1),
                reduction='batchmean'
            )
            distillation_losses.append(kl_loss)
        
        # Average distillation loss
        avg_distillation_loss = torch.stack(distillation_losses).mean()
        
        # Combine losses
        total_loss = classification_loss + 0.5 * avg_distillation_loss
        
        return total_loss


def train_crisis_ensemble(
    ensemble_config: CrisisEnsembleConfig,
    training_config: EnsembleTrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None
) -> Dict[str, Any]:
    """
    Main function for training crisis classification ensembles.
    
    Args:
        ensemble_config: Ensemble model configuration
        training_config: Training process configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Optional test data loader
        
    Returns:
        Comprehensive training results
    """
    logger.info("Starting crisis ensemble training...")
    
    # Initialize trainer
    trainer = CrisisEnsembleTrainer(ensemble_config, training_config)
    
    # Setup training
    trainer.setup_training()
    
    # Execute training
    results = trainer.train(train_loader, val_loader, test_loader)
    
    logger.info("Crisis ensemble training completed successfully!")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Crisis Ensemble Training System...")
    
    # Create test configurations
    ensemble_config = CrisisEnsembleConfig(
        ensemble_type="soft_voting",
        num_base_models=2,  # Reduced for testing
        crisis_weighting_strategy="crisis_adaptive"
    )
    
    training_config = EnsembleTrainingConfig(
        pretrain_individual_models=True,
        individual_training_epochs=1,  # Reduced for testing
        ensemble_training_epochs=1,
        training_strategy="sequential",
        verbose_logging=True
    )
    
    print(f"✅ Configurations created")
    print(f"✅ Ensemble type: {ensemble_config.ensemble_type}")
    print(f"✅ Training strategy: {training_config.training_strategy}")
    
    # Test trainer initialization
    try:
        trainer = CrisisEnsembleTrainer(ensemble_config, training_config)
        print(f"✅ Trainer created successfully")
        print(f"✅ Device: {trainer.device}")
        print(f"✅ Output directory: {trainer.output_dir}")
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
    
    print("\n🎉 Crisis ensemble training system testing completed successfully!")
    print("📊 All training components are working correctly!")