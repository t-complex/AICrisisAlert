#!/usr/bin/env python3
"""
Enhanced Crisis Classification Training Pipeline with BERTweet and Focal Loss

This module provides an enhanced training pipeline specifically optimized for 
social media crisis classification using:

- BERTweet-base model for social media text understanding
- Focal loss for handling remaining class imbalance
- Advanced training techniques with gradient accumulation
- Learning rate warmup and sophisticated scheduling
- Leak-free balanced datasets

Expected Performance:
- Overall Accuracy: 87-92%
- Macro F1-Score: 0.82-0.87
- All classes F1-Score: 0.78+

Usage:
    python enhanced_train.py --config configs/enhanced_training_config.json
    python enhanced_train.py --data_dir data/processed --output_dir outputs/models/bertweet
"""

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import numpy as np
from pathlib import Path
import argparse
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
import gc
from tqdm.auto import tqdm
import torch.backends.mps

# Optional imports for monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Local imports
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from models.model_loader import load_crisis_classifier
from models.lora_setup import setup_lora_model, LoRAConfiguration, LoRATrainingOptimizer
from training.trainer_utils import (
    CrisisMetrics, EarlyStopping, CrisisLoss, TrainingVisualizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedTrainingConfig:
    """
    Enhanced configuration class for BERTweet-based crisis classification.
    
    Contains optimized hyperparameters for social media crisis text
    classification with advanced training techniques.
    """
    # Data configuration
    data_dir: str = "data/processed"
    output_dir: str = "outputs/models/bertweet_enhanced"
    model_name: str = "vinai/bertweet-base"  # Social media optimized
    max_length: int = 128  # BERTweet's maximum supported length
    
    # Enhanced training hyperparameters
    epochs: int = 5  # Increased from 3
    learning_rate: float = 1e-5  # Reduced for better convergence
    batch_size: int = 16
    gradient_accumulation_steps: int = 4  # Increased for larger effective batch
    warmup_steps: int = 500  # Increased warmup
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Enhanced LoRA configuration
    lora_rank: int = 16  # Increased for better adaptation
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Advanced optimization settings
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_balanced_sampling: bool = True
    apply_augmentation: bool = True
    
    # Enhanced early stopping
    early_stopping_patience: int = 3
    early_stopping_monitor: str = "val_macro_f1"
    
    # Monitoring
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 500
    
    # Loss function configuration
    loss_function: str = "focal_loss"  # Changed from weighted_cross_entropy
    focal_loss_alpha: Optional[List[float]] = None  # Will be calculated from data
    focal_loss_gamma: float = 2.0  # Focus on hard examples
    class_weighting_method: str = "balanced"
    
    # Dataset configuration
    use_leak_free_datasets: bool = True
    
    # Experiment tracking
    use_wandb: bool = False
    use_tensorboard: bool = True
    experiment_name: Optional[str] = None
    
    # System settings
    num_workers: Optional[int] = None
    pin_memory: bool = True
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-configure experiment name
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"bertweet_enhanced_{timestamp}"
        
        # Validate configuration
        assert self.epochs > 0, "epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.loss_function in ["focal_loss", "weighted_cross_entropy", "label_smoothing"]
        
        # BERTweet-specific validation
        if "bertweet" in self.model_name.lower():
            if self.max_length > 128:
                logger.warning(f"BERTweet maximum length is 128, but {self.max_length} was specified. Setting to 128.")
                self.max_length = 128
        
        logger.info(f"EnhancedTrainingConfig initialized for experiment: {self.experiment_name}")
        logger.info(f"Using BERTweet model: {self.model_name}")
        logger.info(f"Loss function: {self.loss_function}")
        logger.info(f"Max length: {self.max_length}")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Enhanced training configuration saved to {path}")
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'EnhancedTrainingConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class EnhancedCrisisTrainer:
    """
    Enhanced training orchestrator for BERTweet-based crisis classification.
    
    Features advanced training techniques including focal loss, sophisticated
    scheduling, and optimizations for social media text understanding.
    """
    
    def __init__(self, config: EnhancedTrainingConfig):
        """
        Initialize enhanced crisis trainer.
        
        Args:
            config: Enhanced training configuration
        """
        self.config = config
        self.device = self._setup_device()
        self.output_dir = Path(config.output_dir)
        
        # Initialize monitoring
        self.visualizer = None
        self.wandb_run = None
        self.tensorboard_writer = None
        self._setup_monitoring()
        
        # Components to be initialized in setup()
        self.model = None
        self.lora_model = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.training_history = []
        
        logger.info(f"EnhancedCrisisTrainer initialized for experiment: {config.experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup device with proper CUDA detection for training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA available: Using GPU {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("MPS available: Using Apple Silicon GPU")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU available: Using CPU (training will be slow)")
        
        return device
    
    def _setup_monitoring(self):
        """Setup monitoring and logging."""
        # Setup visualizer
        self.visualizer = TrainingVisualizer(
            save_dir=str(self.output_dir / "logs"),
            log_level='INFO'
        )
        
        # Setup Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project="crisis-classification-enhanced",
                name=self.config.experiment_name,
                config=asdict(self.config),
                reinit=True
            )
            logger.info("Weights & Biases initialized")
        elif self.config.use_wandb:
            logger.warning("Weights & Biases requested but not available")
        
        # Setup TensorBoard
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tensorboard_writer = SummaryWriter(
                log_dir=str(self.output_dir / "tensorboard")
            )
            logger.info("TensorBoard initialized")
        elif self.config.use_tensorboard:
            logger.warning("TensorBoard requested but not available")
    
    def setup(self):
        """Setup all training components."""
        logger.info("Setting up enhanced training components...")
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Load data
        self._setup_data()
        
        # Initialize model
        self._setup_model()
        
        # Setup optimization
        self._setup_optimization()
        
        # Save configuration
        self.config.save(self.output_dir / "enhanced_training_config.json")
        
        logger.info("Enhanced training setup completed successfully")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.backends.mps.is_available():
            pass  # No extra seed needed for MPS
        elif torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        logger.info(f"Random seeds set to {self.config.seed}")
    
    def _setup_data(self):
        """Setup data loaders with leak-free datasets."""
        logger.info("Loading leak-free balanced datasets...")
        
        # Determine dataset file names based on configuration
        if self.config.use_leak_free_datasets:
            train_file = "train_balanced_leak_free.csv"
            val_file = "validation_balanced_leak_free.csv"
            test_file = "test_balanced_leak_free.csv"
            logger.info("Using NEW leak-free balanced datasets")
        else:
            train_file = "train_balanced.csv"
            val_file = "validation_balanced.csv" 
            test_file = "test_balanced.csv"
            logger.info("Using OLD datasets (with potential leakage)")
        
        # Create custom data loading with specific files
        from training.dataset_utils import create_data_loaders
        
        train_path = Path(self.config.data_dir) / train_file
        val_path = Path(self.config.data_dir) / val_file
        test_path = Path(self.config.data_dir) / test_file
        
        # Verify files exist
        for file_path in [train_path, val_path, test_path]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required dataset file not found: {file_path}")
        
        self.train_loader, self.val_loader, self.test_loader, self.class_weights = create_data_loaders(
            train_csv_path=str(train_path),
            val_csv_path=str(val_path),
            test_csv_path=str(test_path),
            tokenizer_name=self.config.model_name,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            apply_augmentation=self.config.apply_augmentation,
            use_balanced_sampling=self.config.use_balanced_sampling
        )
        
        # Move class weights to device
        self.class_weights = self.class_weights.to(self.device)
        
        logger.info(f"Enhanced data loading completed:")
        logger.info(f"  Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"  Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"  Test samples: {len(self.test_loader.dataset)}")
        logger.info(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"  Class weights: {self.class_weights.tolist()}")
    
    def _setup_model(self):
        """Setup BERTweet model with enhanced LoRA adapters."""
        logger.info("Initializing BERTweet model...")
        
        # Load BERTweet base model
        self.model, self.tokenizer, model_config = load_crisis_classifier(
            model_name=self.config.model_name,
            max_length=self.config.max_length,
            device=self.device
        )
        
        # BERTweet-specific tokenizer setup
        if self.config.model_name == "vinai/bertweet-base":
            # Ensure BERTweet tokenizer has proper padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("BERTweet tokenizer configured with proper padding token")
        
        logger.info(f"BERTweet model loaded: {self.config.model_name}")
        logger.info(f"Model vocabulary size: {self.tokenizer.vocab_size}")
        
        # Configure enhanced LoRA
        lora_config = LoRAConfiguration(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            gradient_checkpointing=self.config.use_gradient_checkpointing
        )
        
        # Apply LoRA
        self.lora_model = setup_lora_model(
            base_model=self.model,
            lora_config=lora_config,
            device=self.device
        )
        
        # Log model efficiency
        efficiency_metrics = self.lora_model.get_efficiency_metrics()
        logger.info(f"Enhanced model efficiency:")
        logger.info(f"  Parameter reduction: {efficiency_metrics['memory_reduction']:.2%}")
        logger.info(f"  Trainable parameters: {efficiency_metrics['trainable_parameters']:,}")
        logger.info(f"  LoRA parameters: {efficiency_metrics['lora_parameters']:,}")
    
    def _setup_optimization(self):
        """Setup enhanced optimization components."""
        logger.info("Setting up enhanced optimization...")
        total_steps = len(self.train_loader) * self.config.epochs // self.config.gradient_accumulation_steps
        self.optimizer = optim.AdamW(
            self.lora_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        # Mixed precision: only enable for CUDA or MPS
        if self.config.use_mixed_precision and (self.device.type == 'cuda' or self.device.type == 'mps'):
            self.scaler = GradScaler() if self.device.type == 'cuda' else None  # GradScaler not supported on MPS
            logger.info(f"Mixed precision training enabled on {self.device.type.upper()}")
        else:
            self.scaler = None
        crisis_loss = CrisisLoss(num_classes=6, device=self.device)
        if self.config.loss_function == "focal_loss":
            alpha = self.class_weights if self.config.focal_loss_alpha is None else torch.tensor(self.config.focal_loss_alpha)
            self.criterion = crisis_loss.focal_loss(
                alpha=alpha,
                gamma=self.config.focal_loss_gamma
            )
            logger.info(f"Using Focal Loss with gamma={self.config.focal_loss_gamma}")
        elif self.config.loss_function == "label_smoothing":
            self.criterion = crisis_loss.label_smoothing_loss(
                smoothing=0.1,
                class_weights=self.class_weights
            )
            logger.info("Using Label Smoothing Loss")
        else:
            self.criterion = crisis_loss.weighted_cross_entropy(self.class_weights)
            logger.info("Using Weighted Cross-Entropy Loss")
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            monitor=self.config.early_stopping_monitor,
            save_path=str(self.output_dir / "best_model.pt")
        )
        self.memory_optimizer = LoRATrainingOptimizer(
            LoRAConfiguration(
                gradient_checkpointing=self.config.use_gradient_checkpointing,
                accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision="fp16" if self.scaler else "no"
            )
        )
        logger.info(f"Enhanced optimization setup completed:")
        logger.info(f"  Total training steps: {total_steps}")
        logger.info(f"  Warmup steps: {self.config.warmup_steps}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
    
    def train(self):
        """Enhanced training loop with advanced techniques."""
        logger.info("Starting enhanced training...")
        
        # Initialize metrics
        train_metrics = CrisisMetrics()
        val_metrics = CrisisMetrics()
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            logger.info("=" * 60)
            
            # Training phase
            train_loss, train_results = self._train_epoch(train_metrics)
            
            # Validation phase
            val_loss, val_results = self._validate_epoch(val_metrics)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
            
            # Log epoch results
            self._log_epoch_results(
                epoch, train_loss, val_loss, train_results, val_results,
                current_lr, epoch_time
            )
            
            # Check early stopping
            should_stop = self.early_stopping(
                val_results.get('macro_f1', 0.0),
                self.lora_model.model,
                epoch
            )
            
            if should_stop:
                logger.info("Early stopping triggered")
                break
            
            # Memory cleanup
            if epoch % 2 == 0:
                self._cleanup_memory()
        
        # Final evaluation
        self._final_evaluation()
        
        # Save final model
        self._save_final_model()
        
        logger.info("Enhanced training completed successfully!")
        logger.info(f"Best validation macro F1: {self.best_val_f1:.4f}")
    
    def get_best_validation_metrics(self) -> Dict[str, float]:
        """Get the best validation metrics from the training run."""
        val_metrics_file = Path(self.config.output_dir) / 'logs' / 'val_metrics.json'
        if val_metrics_file.exists():
            with open(val_metrics_file, 'r') as f:
                return json.load(f)
        return None
    
    def _train_epoch(self, metrics: CrisisMetrics) -> Tuple[float, Dict]:
        """Enhanced training epoch with advanced techniques."""
        self.lora_model.train()
        metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Training Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.scaler is not None):
                outputs = self.lora_model.forward(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                loss = self.criterion(outputs['logits'], batch['labels'])
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update metrics
            with torch.no_grad():
                predictions = torch.argmax(outputs['logits'], dim=-1)
                metrics.update(predictions, batch['labels'])
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Gradient accumulation and update
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.lora_model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Enhanced memory monitoring
                if self.global_step % 100 == 0:
                    self.memory_optimizer.monitor_memory_usage(self.global_step)
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Enhanced logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_training_step(avg_loss, current_lr)
        
        # Compute final metrics
        train_results = metrics.compute()
        avg_loss = total_loss / num_batches
        
        return avg_loss, train_results
    
    def _validate_epoch(self, metrics: CrisisMetrics) -> Tuple[float, Dict]:
        """Enhanced validation epoch."""
        self.lora_model.eval()
        metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.lora_model.forward(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                loss = self.criterion(outputs['logits'], batch['labels'])
                total_loss += loss.item()
                
                # Update metrics
                predictions = torch.argmax(outputs['logits'], dim=-1)
                metrics.update(predictions, batch['labels'])
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
        
        # Compute final metrics
        val_results = metrics.compute()
        avg_loss = total_loss / num_batches

        # Save best validation metrics for Optuna
        output_dir = self.config.output_dir if hasattr(self.config, 'output_dir') else 'outputs/models/bertweet_enhanced'
        logs_dir = Path(output_dir) / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        val_metrics_path = logs_dir / 'val_metrics.json'
        with open(val_metrics_path, 'w') as f:
            json.dump(val_results, f, indent=2)

        return avg_loss, val_results
    
    def _log_epoch_results(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_results: Dict,
        val_results: Dict,
        learning_rate: float,
        epoch_time: float
    ):
        """Enhanced epoch results logging."""
        # Console logging
        logger.info(f"Epoch {epoch + 1} Enhanced Results:")
        logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"  Train Acc: {train_results['accuracy']:.4f} | Val Acc: {val_results['accuracy']:.4f}")
        logger.info(f"  Train F1: {train_results['macro_f1']:.4f} | Val F1: {val_results['macro_f1']:.4f}")
        logger.info(f"  Learning Rate: {learning_rate:.2e} | Time: {epoch_time:.2f}s")
        
        # Crisis-specific metrics
        crisis_metrics = val_results
        if 'humanitarian_f1' in crisis_metrics:
            logger.info(f"  Humanitarian F1: {crisis_metrics.get('humanitarian_f1', 0):.4f}")
            logger.info(f"  Critical Crisis F1: {crisis_metrics.get('critical_crisis_f1', 0):.4f}")
        
        # Update training history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_results['accuracy'],
            'val_accuracy': val_results['accuracy'],
            'train_macro_f1': train_results['macro_f1'],
            'val_macro_f1': val_results['macro_f1'],
            'learning_rate': learning_rate,
            'epoch_time': epoch_time
        }
        self.training_history.append(epoch_data)
        
        # Enhanced monitoring
        if self.visualizer:
            train_metrics_logged = {f'train_{k}': v for k, v in train_results.items() if isinstance(v, (int, float))}
            val_metrics_logged = {f'val_{k}': v for k, v in val_results.items() if isinstance(v, (int, float))}
            
            self.visualizer.log_epoch(
                epoch + 1,
                {**train_metrics_logged, 'loss': train_loss},
                {**val_metrics_logged, 'loss': val_loss},
                learning_rate,
                epoch_time
            )
        
        # Track best validation F1
        if val_results['macro_f1'] > self.best_val_f1:
            self.best_val_f1 = val_results['macro_f1']
            logger.info(f"  🎯 New best validation F1: {self.best_val_f1:.4f}")
    
    def _log_training_step(self, loss: float, learning_rate: float):
        """Enhanced training step logging."""
        if self.wandb_run:
            self.wandb_run.log({
                'step_loss': loss,
                'step_learning_rate': learning_rate,
                'global_step': self.global_step
            }, step=self.global_step)
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Step/Loss', loss, self.global_step)
            self.tensorboard_writer.add_scalar('Step/Learning_Rate', learning_rate, self.global_step)
    
    def _final_evaluation(self):
        """Enhanced final evaluation on test set."""
        logger.info("Performing enhanced final evaluation on test set...")
        
        # Load best model if available
        best_model_path = self.output_dir / "best_model.pt"
        if best_model_path.exists():
            self.lora_model.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            logger.info("Loaded best model for final evaluation")
        
        # Test evaluation
        test_metrics = CrisisMetrics()
        test_loss, test_results = self._validate_epoch(test_metrics)
        
        # Enhanced results logging
        logger.info("\n" + "="*60)
        logger.info("🎯 FINAL ENHANCED TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Test Macro F1: {test_results['macro_f1']:.4f}")
        logger.info(f"Test Weighted F1: {test_results['weighted_f1']:.4f}")
        
        # Crisis-specific metrics
        crisis_metrics = test_metrics.get_crisis_specific_metrics()
        logger.info("\nCrisis-Specific Performance:")
        for metric, value in crisis_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Performance expectations check
        accuracy = test_results['accuracy']
        macro_f1 = test_results['macro_f1']
        min_class_f1 = min(test_results['per_class_f1'])
        
        logger.info("\n📊 Performance vs Expectations:")
        logger.info(f"  Accuracy: {accuracy:.1%} (Expected: 87-92%)")
        logger.info(f"  Macro F1: {macro_f1:.3f} (Expected: 0.82-0.87)")
        logger.info(f"  Min Class F1: {min_class_f1:.3f} (Expected: 0.78+)")
        
        # Save enhanced test results
        enhanced_results = {
            **test_results,
            **crisis_metrics,
            'performance_expectations': {
                'accuracy_target': '87-92%',
                'macro_f1_target': '0.82-0.87',
                'min_class_f1_target': '0.78+',
                'accuracy_achieved': f"{accuracy:.1%}",
                'macro_f1_achieved': f"{macro_f1:.3f}",
                'min_class_f1_achieved': f"{min_class_f1:.3f}"
            }
        }
        
        test_results_path = self.output_dir / "enhanced_test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        logger.info(f"Enhanced test results saved to {test_results_path}")
        logger.info("="*60)
    
    def _save_final_model(self):
        """Save enhanced final model and artifacts."""
        logger.info("Saving enhanced model and artifacts...")
        
        # Save LoRA adapters
        lora_path = self.output_dir / "lora_adapters"
        self.lora_model.save_lora_adapters(lora_path)
        
        # Save merged model
        merged_model = self.lora_model.merge_and_unload()
        torch.save(merged_model.state_dict(), self.output_dir / "enhanced_merged_model.pt")
        
        # Save tokenizer
        tokenizer_path = self.output_dir / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save training history
        history_path = self.output_dir / "enhanced_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save visualizer artifacts
        if self.visualizer:
            self.visualizer.save_training_history()
        
        logger.info(f"Enhanced model and artifacts saved to {self.output_dir}")
    
    def _cleanup_memory(self):
        """Enhanced memory cleanup."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def __del__(self):
        """Cleanup when trainer is destroyed."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()


def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create enhanced command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced Crisis Classification Training Pipeline with BERTweet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/processed",
        help="Directory containing leak-free datasets"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/models/bertweet_enhanced",
        help="Directory to save enhanced model and artifacts"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="vinai/bertweet-base",
        help="BERTweet model name"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tweets"
    )
    
    # Enhanced training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for enhanced training"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps for larger effective batch"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    
    # Enhanced LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Enhanced LoRA rank"
    )
    
    # Loss function arguments
    parser.add_argument(
        "--loss_function",
        type=str,
        default="focal_loss",
        choices=["focal_loss", "weighted_cross_entropy", "label_smoothing"],
        help="Loss function to use"
    )
    
    parser.add_argument(
        "--focal_loss_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss"
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to enhanced configuration JSON file"
    )
    
    # Dataset selection
    parser.add_argument(
        "--use_leak_free_datasets",
        action="store_true",
        default=True,
        help="Use leak-free balanced datasets"
    )
    
    # Monitoring
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for monitoring"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Experiment name for logging"
    )
    
    return parser


def main():
    """Enhanced main training function."""
    logger.info("🚀 Starting Enhanced Crisis Classification Training")
    logger.info("Features: BERTweet + Focal Loss + Leak-Free Data + Advanced Training")
    
    # Parse arguments
    parser = create_enhanced_parser()
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (optional)"
    )
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = EnhancedTrainingConfig.from_file(args.config)
        logger.info(f"Loaded enhanced configuration from {args.config}")
    else:
        # Create config from command line arguments
        config_dict = {k: v for k, v in vars(args).items() if v is not None and k not in ['config', 'resume_from_checkpoint']}
        config = EnhancedTrainingConfig(**config_dict)
    
    # Initialize enhanced trainer
    trainer = EnhancedCrisisTrainer(config)
    
    try:
        # Setup training
        trainer.setup()
        
        # Resume logic
        if args.resume_from_checkpoint:
            checkpoint_path = Path(args.resume_from_checkpoint)
            if checkpoint_path.exists():
                logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
                # Robustly handle different checkpoint formats
                model_state = None
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    model_state = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    model_state = checkpoint['model']
                elif isinstance(checkpoint, dict) and all(k.startswith('lora_') or k.startswith('bert') or k.startswith('classifier') for k in checkpoint.keys()):
                    # Looks like a raw state_dict
                    model_state = checkpoint
                if model_state is not None:
                    if trainer.lora_model is not None and hasattr(trainer.lora_model, 'model') and trainer.lora_model.model is not None:
                        trainer.lora_model.model.load_state_dict(model_state)
                        logger.info("Model weights loaded from checkpoint.")
                    else:
                        logger.error("lora_model or lora_model.model is not initialized. Cannot load weights.")
                        raise RuntimeError("lora_model or lora_model.model is not initialized.")
                else:
                    logger.error("No model weights found in checkpoint. Keys: %s", list(checkpoint.keys()))
                    raise KeyError("No model weights found in checkpoint.")
                # Optimizer and scheduler (optional)
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if hasattr(trainer, 'scheduler') and trainer.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                trainer.current_epoch = checkpoint.get('epoch', 0)
                trainer.global_step = checkpoint.get('global_step', 0)
                logger.info(f"Checkpoint loaded: epoch={trainer.current_epoch}, global_step={trainer.global_step}")
            else:
                logger.warning(f"Checkpoint file not found: {checkpoint_path}")
        
        # Start enhanced training
        trainer.train()
        
        logger.info("✅ Enhanced training completed successfully!")
        logger.info(f"Best validation macro F1: {trainer.best_val_f1:.4f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer._save_final_model()
        
    except Exception as e:
        logger.error(f"Enhanced training failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        trainer._cleanup_memory()


if __name__ == "__main__":
    main()