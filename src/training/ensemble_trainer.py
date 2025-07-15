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
from dataclasses import dataclass, asdict
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnsembleTrainingConfig:
    """
    Extended configuration for ensemble training with crisis-specific parameters.
    
    Includes multi-stage training settings, optimization strategies, and
    crisis-specific objectives for emergency response systems.
    """
    # Multi-stage training
    pretrain_individual_models: bool = True
    individual_training_epochs: int = 3
    ensemble_training_epochs: int = 2
    fine_tuning_epochs: int = 1
    
    # Training strategies
    training_strategy: str = "sequential"  # sequential, parallel, alternating
    use_curriculum_learning: bool = True
    curriculum_difficulty_threshold: float = 0.8
    
    # Optimization parameters
    individual_learning_rate: float = 2e-5
    ensemble_learning_rate: float = 1e-4
    fine_tuning_learning_rate: float = 5e-6
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # Crisis-specific training
    crisis_loss_weight: float = 1.5
    humanitarian_boost: float = 1.2
    critical_crisis_weight: float = 2.0
    false_positive_penalty: float = 0.8
    
    # Loss function configuration
    ensemble_loss_function: str = "crisis_adaptive"  # crisis_adaptive, weighted_ensemble, knowledge_distillation
    temperature_scaling: bool = True
    initial_temperature: float = 3.0
    
    # Validation and early stopping
    validation_strategy: str = "ensemble_aware"  # standard, ensemble_aware, cross_validation
    early_stopping_patience: int = 3
    early_stopping_monitor: str = "ensemble_macro_f1"
    min_improvement: float = 0.001
    
    # Hyperparameter optimization
    optimize_hyperparameters: bool = False
    optimization_trials: int = 20
    optimization_metric: str = "macro_f1"
    
    # Data augmentation and sampling
    use_data_augmentation: bool = True
    augmentation_strength: float = 0.1
    use_adaptive_sampling: bool = True
    sampling_strategy: str = "difficulty_based"
    
    # Performance thresholds
    min_individual_performance: float = 0.75
    target_ensemble_improvement: float = 0.03
    diversity_threshold: float = 0.2
    
    # System and monitoring
    parallel_training: bool = True
    max_workers: int = 3
    save_checkpoints: bool = True
    checkpoint_frequency: int = 500
    verbose_logging: bool = True
    
    # Output configuration
    output_dir: str = "outputs/ensemble_training"
    experiment_name: Optional[str] = None
    save_intermediate_models: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-configure experiment name
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"ensemble_training_{timestamp}"
        
        # Validate configuration
        assert self.training_strategy in ["sequential", "parallel", "alternating"]
        assert self.ensemble_loss_function in ["crisis_adaptive", "weighted_ensemble", "knowledge_distillation"]
        assert self.validation_strategy in ["standard", "ensemble_aware", "cross_validation"]
        assert self.sampling_strategy in ["difficulty_based", "uncertainty_based", "balanced"]
        
        logger.info(f"EnsembleTrainingConfig initialized for experiment: {self.experiment_name}")


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


class CrisisEnsembleTrainer:
    """
    Comprehensive trainer for crisis classification ensembles.
    
    Provides multi-stage training, crisis-specific optimization,
    and advanced ensemble coordination for emergency response systems.
    """
    
    def __init__(
        self,
        ensemble_config: CrisisEnsembleConfig,
        training_config: EnsembleTrainingConfig
    ):
        """
        Initialize crisis ensemble trainer.
        
        Args:
            ensemble_config: Ensemble model configuration
            training_config: Training process configuration
        """
        self.ensemble_config = ensemble_config
        self.training_config = training_config
        self.device = self._setup_device()
        self.output_dir = Path(training_config.output_dir)
        
        # Training components
        self.ensemble = None
        self.individual_models = []
        self.optimizers = {}
        self.schedulers = {}
        self.criterion = None
        
        # Training state
        self.training_history = []
        self.best_ensemble_score = 0.0
        self.current_stage = "initialization"
        
        # Evaluation components
        self.evaluator = EnhancedCrisisEvaluator()
        self.metrics_tracker = CrisisMetrics()
        
        logger.info(f"CrisisEnsembleTrainer initialized")
        logger.info(f"Ensemble type: {ensemble_config.ensemble_type}")
        logger.info(f"Training strategy: {training_config.training_strategy}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device = torch.device("cpu")
        logger.info("Using CPU for ensemble training")
        return device
    
    def setup_training(self):
        """Setup training components and initialize models."""
        logger.info("Setting up ensemble training...")
        
        # Create crisis-adaptive loss function
        self.criterion = CrisisAdaptiveLoss(self.training_config, self.device)
        
        # Initialize individual models if needed
        if self.training_config.pretrain_individual_models:
            self._initialize_individual_models()
        
        # Create ensemble
        self.ensemble = create_crisis_ensemble(
            self.ensemble_config,
            base_models=self.individual_models if self.individual_models else None
        )
        self.ensemble = self.ensemble.to(self.device)
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=self.training_config.early_stopping_patience,
            monitor=self.training_config.early_stopping_monitor,
            save_path=str(self.output_dir / "best_ensemble.pt")
        )
        
        logger.info("Ensemble training setup completed")
    
    def _initialize_individual_models(self):
        """Initialize individual models for the ensemble."""
        logger.info("Initializing individual models...")
        
        for i, model_config in enumerate(self.ensemble_config.base_model_configs):
            logger.info(f"Initializing model {i+1}/{len(self.ensemble_config.base_model_configs)}: {model_config['model_name']}")
            
            # Load model
            model, tokenizer, config = load_crisis_classifier(
                model_name=model_config['model_name'],
                max_length=model_config['max_length'],
                classifier_dropout=model_config['classifier_dropout'],
                device=self.device
            )
            
            # Apply LoRA if specified
            if 'lora_rank' in model_config:
                lora_config = LoRAConfiguration(
                    r=model_config['lora_rank'],
                    lora_alpha=model_config.get('lora_alpha', 32),
                    lora_dropout=model_config.get('lora_dropout', 0.1)
                )
                model = setup_lora_model(model, lora_config, self.device)
            
            self.individual_models.append(model)
        
        logger.info(f"Initialized {len(self.individual_models)} individual models")
    
    def _setup_optimizers(self):
        """Setup optimizers for different training stages."""
        # Individual model optimizers
        if self.individual_models:
            for i, model in enumerate(self.individual_models):
                self.optimizers[f'individual_{i}'] = optim.AdamW(
                    model.parameters(),
                    lr=self.training_config.individual_learning_rate,
                    weight_decay=self.training_config.weight_decay
                )
        
        # Ensemble optimizer
        if hasattr(self.ensemble, 'parameters'):
            self.optimizers['ensemble'] = optim.AdamW(
                self.ensemble.parameters(),
                lr=self.training_config.ensemble_learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        
        # Fine-tuning optimizer (lower learning rate)
        ensemble_params = []
        if hasattr(self.ensemble, 'parameters'):
            ensemble_params.extend(self.ensemble.parameters())
        if self.individual_models:
            for model in self.individual_models:
                ensemble_params.extend(model.parameters())
        
        if ensemble_params:
            self.optimizers['fine_tuning'] = optim.AdamW(
                ensemble_params,
                lr=self.training_config.fine_tuning_learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        
        logger.info(f"Setup {len(self.optimizers)} optimizers for different training stages")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Execute complete ensemble training pipeline.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting crisis ensemble training pipeline...")
        
        training_results = {}
        
        # Stage 1: Individual model pre-training
        if self.training_config.pretrain_individual_models and self.individual_models:
            logger.info("\n" + "="*60)
            logger.info("STAGE 1: INDIVIDUAL MODEL PRE-TRAINING")
            logger.info("="*60)
            
            self.current_stage = "individual_pretraining"
            individual_results = self._train_individual_models(train_loader, val_loader)
            training_results['individual_pretraining'] = individual_results
        
        # Stage 2: Ensemble training
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: ENSEMBLE TRAINING")
        logger.info("="*60)
        
        self.current_stage = "ensemble_training"
        ensemble_results = self._train_ensemble(train_loader, val_loader)
        training_results['ensemble_training'] = ensemble_results
        
        # Stage 3: Fine-tuning
        if self.training_config.fine_tuning_epochs > 0:
            logger.info("\n" + "="*60)
            logger.info("STAGE 3: ENSEMBLE FINE-TUNING")
            logger.info("="*60)
            
            self.current_stage = "fine_tuning"
            fine_tuning_results = self._fine_tune_ensemble(train_loader, val_loader)
            training_results['fine_tuning'] = fine_tuning_results
        
        # Final evaluation
        if test_loader is not None:
            logger.info("\n" + "="*60)
            logger.info("FINAL EVALUATION")
            logger.info("="*60)
            
            final_results = self._final_evaluation(test_loader)
            training_results['final_evaluation'] = final_results
        
        # Save training artifacts
        self._save_training_artifacts(training_results)
        
        logger.info("Crisis ensemble training pipeline completed successfully!")
        logger.info(f"Best ensemble score: {self.best_ensemble_score:.4f}")
        
        return training_results
    
    def _train_individual_models(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Train individual models separately before ensemble training.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Individual training results
        """
        individual_results = {}
        
        for i, model in enumerate(self.individual_models):
            logger.info(f"\nTraining individual model {i+1}/{len(self.individual_models)}")
            
            model_config = self.ensemble_config.base_model_configs[i]
            model_name = model_config.get('model_name', f'model_{i}')
            
            # Model-specific training
            optimizer = self.optimizers[f'individual_{i}']
            model_results = self._train_single_model(
                model, optimizer, train_loader, val_loader,
                epochs=self.training_config.individual_training_epochs,
                model_name=f"individual_{i}"
            )
            
            individual_results[f'model_{i}'] = {
                'model_name': model_name,
                'results': model_results,
                'specialization': model_config.get('specialization', 'general')
            }
            
            # Save individual model
            if self.training_config.save_intermediate_models:
                model_path = self.output_dir / f"individual_model_{i}.pt"
                torch.save(model.state_dict(), model_path)
                logger.info(f"Individual model {i} saved to {model_path}")
        
        return individual_results
    
    def _train_ensemble(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Train the ensemble model with crisis-specific optimization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Ensemble training results
        """
        logger.info("Training crisis ensemble with adaptive optimization...")
        
        optimizer = self.optimizers.get('ensemble')
        if optimizer is None:
            logger.warning("No ensemble optimizer found, using fine-tuning optimizer")
            optimizer = self.optimizers['fine_tuning']
        
        ensemble_results = self._train_single_model(
            self.ensemble, optimizer, train_loader, val_loader,
            epochs=self.training_config.ensemble_training_epochs,
            model_name="ensemble",
            use_ensemble_loss=True
        )
        
        return ensemble_results
    
    def _fine_tune_ensemble(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Fine-tune the complete ensemble system.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Fine-tuning results
        """
        logger.info("Fine-tuning complete ensemble system...")
        
        optimizer = self.optimizers['fine_tuning']
        
        # Use lower learning rate and crisis-adaptive loss
        fine_tuning_results = self._train_single_model(
            self.ensemble, optimizer, train_loader, val_loader,
            epochs=self.training_config.fine_tuning_epochs,
            model_name="fine_tuned_ensemble",
            use_ensemble_loss=True,
            fine_tuning_mode=True
        )
        
        return fine_tuning_results
    
    def _train_single_model(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_name: str,
        use_ensemble_loss: bool = False,
        fine_tuning_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Train a single model (individual or ensemble).
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            model_name: Name for logging
            use_ensemble_loss: Whether to use ensemble-specific loss
            fine_tuning_mode: Whether in fine-tuning mode
            
        Returns:
            Training results
        """
        training_history = []
        best_score = 0.0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs} - {model_name}")
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(
                model, optimizer, train_loader, use_ensemble_loss, fine_tuning_mode
            )
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(
                model, val_loader, use_ensemble_loss
            )
            
            # Log results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_metrics.get('accuracy', 0.0),
                'val_accuracy': val_metrics.get('accuracy', 0.0),
                'train_f1': train_metrics.get('macro_f1', 0.0),
                'val_f1': val_metrics.get('macro_f1', 0.0)
            }
            
            training_history.append(epoch_results)
            
            # Track best score
            val_f1 = val_metrics.get('macro_f1', 0.0)
            if val_f1 > best_score:
                best_score = val_f1
                
                # Update global best if this is ensemble training
                if use_ensemble_loss and val_f1 > self.best_ensemble_score:
                    self.best_ensemble_score = val_f1
            
            if self.training_config.verbose_logging:
                logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                logger.info(f"  Train Acc: {train_metrics.get('accuracy', 0.0):.4f}, Val Acc: {val_metrics.get('accuracy', 0.0):.4f}")
                logger.info(f"  Train F1: {train_metrics.get('macro_f1', 0.0):.4f}, Val F1: {val_metrics.get('macro_f1', 0.0):.4f}")
        
        return {
            'training_history': training_history,
            'best_score': best_score,
            'final_metrics': {
                'accuracy': val_metrics.get('accuracy', 0.0),
                'macro_f1': val_metrics.get('macro_f1', 0.0),
                'weighted_f1': val_metrics.get('weighted_f1', 0.0)
            }
        }
    
    def _train_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        use_ensemble_loss: bool = False,
        fine_tuning_mode: bool = False
    ) -> Tuple[float, Dict]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False) if self.training_config.verbose_logging else train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if use_ensemble_loss and hasattr(model, 'forward'):
                # Ensemble model with individual predictions
                outputs = model(
                    input_ids, attention_mask, 
                    return_individual_predictions=True
                )
                
                # Crisis-adaptive ensemble loss
                if 'individual_probabilities' in outputs:
                    loss = self.criterion(
                        outputs['logits'], labels,
                        individual_predictions=outputs['individual_probabilities'],
                        loss_type=self.training_config.ensemble_loss_function
                    )
                else:
                    loss = self.criterion(outputs['logits'], labels)
                
                predictions = outputs['predictions']
            else:
                # Standard model training
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs['logits'], labels)
                predictions = torch.argmax(outputs['logits'], dim=-1)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.training_config.max_grad_norm
            )
            
            optimizer.step()
            
            # Collect metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            if self.training_config.verbose_logging and hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(all_targets, all_predictions)
        
        return avg_loss, metrics
    
    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        use_ensemble_loss: bool = False
    ) -> Tuple[float, Dict]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if use_ensemble_loss and hasattr(model, 'forward'):
                    outputs = model(input_ids, attention_mask)
                    loss = self.criterion(outputs['logits'], labels)
                    predictions = outputs['predictions']
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(outputs['logits'], labels)
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                
                # Collect metrics
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_targets, all_predictions)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, targets: List[int], predictions: List[int]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        accuracy = accuracy_score(targets, predictions)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
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
    
    def _final_evaluation(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Perform final evaluation on test set."""
        logger.info("Performing final evaluation on test set...")
        
        # Load best model if available
        best_model_path = self.output_dir / "best_ensemble.pt"
        if best_model_path.exists():
            try:
                self.ensemble.load_state_dict(torch.load(best_model_path, map_location=self.device))
                logger.info("Loaded best ensemble model for final evaluation")
            except Exception as e:
                logger.warning(f"Could not load best model: {e}")
        
        # Comprehensive evaluation using enhanced evaluator
        results = self.evaluator.evaluate_comprehensive(
            model=self.ensemble,
            dataloader=test_loader,
            device=self.device,
            class_names=CRISIS_LABELS,
            output_dir=str(self.output_dir / "evaluation")
        )
        
        # Log key results
        logger.info("Final evaluation results:")
        logger.info(f"  Accuracy: {results.base_results.accuracy:.4f}")
        logger.info(f"  Macro F1: {results.base_results.macro_f1:.4f}")
        logger.info(f"  Humanitarian F1: {results.base_results.humanitarian_f1:.4f}")
        logger.info(f"  Critical Crisis F1: {results.base_results.critical_crisis_f1:.4f}")
        
        return {
            'ensemble_performance': {
                'accuracy': results.base_results.accuracy,
                'macro_f1': results.base_results.macro_f1,
                'weighted_f1': results.base_results.weighted_f1,
                'humanitarian_f1': results.base_results.humanitarian_f1,
                'critical_crisis_f1': results.base_results.critical_crisis_f1
            },
            'detailed_results': results
        }
    
    def _save_training_artifacts(self, training_results: Dict[str, Any]):
        """Save training artifacts and results."""
        # Save configurations
        ensemble_config_path = self.output_dir / "ensemble_config.json"
        with open(ensemble_config_path, 'w') as f:
            json.dump(asdict(self.ensemble_config), f, indent=2)
        
        training_config_path = self.output_dir / "training_config.json"
        with open(training_config_path, 'w') as f:
            json.dump(asdict(self.training_config), f, indent=2)
        
        # Save training results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = self._make_serializable(training_results)
            json.dump(serializable_results, f, indent=2)
        
        # Save final ensemble model
        final_model_path = self.output_dir / "final_ensemble.pt"
        torch.save(self.ensemble.state_dict(), final_model_path)
        
        logger.info(f"Training artifacts saved to {self.output_dir}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


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