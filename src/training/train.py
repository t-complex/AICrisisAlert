#!/usr/bin/env python3
"""
Crisis Classification Training Pipeline

This module provides the main training orchestrator for fine-tuning DistilBERT
with LoRA adapters for crisis classification. Features include gradient accumulation,
mixed precision training, real-time monitoring, and comprehensive checkpointing.

Usage:
    python train.py --data_dir data/processed --output_dir outputs/models
    python train.py --config configs/training_config.json
    python train.py --help

Classes:
    TrainingConfig: Configuration for training hyperparameters
    CrisisTrainer: Main training orchestrator

Functions:
    main: Entry point for training script
    setup_logging: Configure logging and monitoring
    create_optimizer: Create optimizer with warmup and scheduling
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
from typing import Dict, Optional, Tuple, Union
import gc
from tqdm.auto import tqdm

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
sys.path.append(str(Path(__file__).parent.parent))

from models.model_loader import load_crisis_classifier
from models.lora_setup import setup_lora_model, LoRAConfiguration, LoRATrainingOptimizer
from training.dataset_utils import prepare_crisis_data
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
class TrainingConfig:
    """
    Configuration class for training hyperparameters.
    
    Contains all training settings including model configuration,
    optimization parameters, and system-specific settings.
    """
    # Data configuration
    data_dir: str = "data/processed"
    output_dir: str = "outputs/models"
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    
    # Training hyperparameters
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Optimization settings
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_balanced_sampling: bool = True
    apply_augmentation: bool = True
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_monitor: str = "val_macro_f1"
    
    # Monitoring
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
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
            self.experiment_name = f"crisis_classification_{timestamp}"
        
        # Validate configuration
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        logger.info(f"TrainingConfig initialized for experiment: {self.experiment_name}")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Training configuration saved to {path}")
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class CrisisTrainer:
    """
    Main training orchestrator for crisis classification.
    
    Integrates all components including data loading, model initialization,
    LoRA adaptation, training loops, and monitoring for efficient crisis
    classification fine-tuning.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize crisis trainer.
        
        Args:
            config: Training configuration
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
        
        logger.info(f"CrisisTrainer initialized for experiment: {config.experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup and configure device for training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Clear cache and optimize GPU settings
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            
        else:
            device = torch.device("cpu")
            logger.info("Using CPU (GPU not available)")
        
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
                project="crisis-classification",
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
        logger.info("Setting up training components...")
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Load data
        self._setup_data()
        
        # Initialize model
        self._setup_model()
        
        # Setup optimization
        self._setup_optimization()
        
        # Save configuration
        self.config.save(self.output_dir / "training_config.json")
        
        logger.info("Training setup completed successfully")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        logger.info(f"Random seeds set to {self.config.seed}")
    
    def _setup_data(self):
        """Setup data loaders."""
        logger.info("Loading datasets...")
        
        self.train_loader, self.val_loader, self.test_loader, self.class_weights = prepare_crisis_data(
            data_dir=self.config.data_dir,
            tokenizer_name=self.config.model_name,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            apply_augmentation=self.config.apply_augmentation,
            use_balanced_sampling=self.config.use_balanced_sampling
        )
        
        # Move class weights to device
        self.class_weights = self.class_weights.to(self.device)
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"  Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"  Test samples: {len(self.test_loader.dataset)}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Class weights: {self.class_weights.tolist()}")
    
    def _setup_model(self):
        """Setup model with LoRA adapters."""
        logger.info("Initializing model...")
        
        # Load base model
        self.model, self.tokenizer, model_config = load_crisis_classifier(
            model_name=self.config.model_name,
            max_length=self.config.max_length,
            device=self.device
        )
        
        # Configure LoRA
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
        logger.info(f"Model efficiency:")
        logger.info(f"  Parameter reduction: {efficiency_metrics['memory_reduction']:.2%}")
        logger.info(f"  Trainable parameters: {efficiency_metrics['trainable_parameters']:,}")
        logger.info(f"  LoRA parameters: {efficiency_metrics['lora_parameters']:,}")
    
    def _setup_optimization(self):
        """Setup optimization components."""
        logger.info("Setting up optimization...")
        
        # Calculate total training steps
        total_steps = len(self.train_loader) * self.config.epochs // self.config.gradient_accumulation_steps
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.lora_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        # Setup learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup mixed precision scaler
        if self.config.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Setup loss function
        crisis_loss = CrisisLoss(num_classes=6, device=self.device)
        self.criterion = crisis_loss.weighted_cross_entropy(self.class_weights)
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            monitor=self.config.early_stopping_monitor,
            save_path=str(self.output_dir / "best_model.pt")
        )
        
        # Setup memory optimizer
        self.memory_optimizer = LoRATrainingOptimizer(
            LoRAConfiguration(
                gradient_checkpointing=self.config.use_gradient_checkpointing,
                accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision="fp16" if self.scaler else "no"
            )
        )
        
        logger.info(f"Optimization setup completed:")
        logger.info(f"  Total training steps: {total_steps}")
        logger.info(f"  Warmup steps: {self.config.warmup_steps}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Initialize metrics
        train_metrics = CrisisMetrics()
        val_metrics = CrisisMetrics()
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            logger.info("=" * 50)
            
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
        
        logger.info("Training completed successfully!")
    
    def _train_epoch(self, metrics: CrisisMetrics) -> Tuple[float, Dict]:
        """Train for one epoch."""
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
            
            # Gradient accumulation
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
                
                # Memory monitoring
                if self.global_step % 50 == 0:
                    self.memory_optimizer.monitor_memory_usage(self.global_step)
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_training_step(avg_loss)
        
        # Compute final metrics
        train_results = metrics.compute()
        avg_loss = total_loss / num_batches
        
        return avg_loss, train_results
    
    def _validate_epoch(self, metrics: CrisisMetrics) -> Tuple[float, Dict]:
        """Validate for one epoch."""
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
        """Log results for the epoch."""
        # Console logging
        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Train Accuracy: {train_results['accuracy']:.4f}")
        logger.info(f"  Val Accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Train Macro F1: {train_results['macro_f1']:.4f}")
        logger.info(f"  Val Macro F1: {val_results['macro_f1']:.4f}")
        logger.info(f"  Learning Rate: {learning_rate:.2e}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s")
        
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
        
        # Add train/val prefix to results
        train_metrics_logged = {f'train_{k}': v for k, v in train_results.items() if isinstance(v, (int, float))}
        val_metrics_logged = {f'val_{k}': v for k, v in val_results.items() if isinstance(v, (int, float))}
        
        # Visualizer logging
        if self.visualizer:
            self.visualizer.log_epoch(
                epoch + 1,
                {**train_metrics_logged, 'loss': train_loss},
                {**val_metrics_logged, 'loss': val_loss},
                learning_rate,
                epoch_time
            )
        
        # Weights & Biases logging
        if self.wandb_run:
            log_dict = {
                **train_metrics_logged,
                **val_metrics_logged,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': learning_rate,
                'epoch_time': epoch_time,
                'epoch': epoch + 1
            }
            self.wandb_run.log(log_dict, step=self.global_step)
        
        # TensorBoard logging
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            self.tensorboard_writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
            self.tensorboard_writer.add_scalar('Accuracy/Train', train_results['accuracy'], epoch + 1)
            self.tensorboard_writer.add_scalar('Accuracy/Validation', val_results['accuracy'], epoch + 1)
            self.tensorboard_writer.add_scalar('F1/Train', train_results['macro_f1'], epoch + 1)
            self.tensorboard_writer.add_scalar('F1/Validation', val_results['macro_f1'], epoch + 1)
            self.tensorboard_writer.add_scalar('Learning_Rate', learning_rate, epoch + 1)
        
        # Track best validation F1
        if val_results['macro_f1'] > self.best_val_f1:
            self.best_val_f1 = val_results['macro_f1']
    
    def _log_training_step(self, loss: float):
        """Log training step metrics."""
        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
        
        if self.wandb_run:
            self.wandb_run.log({
                'step_loss': loss,
                'step_learning_rate': current_lr,
                'global_step': self.global_step
            }, step=self.global_step)
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Step/Loss', loss, self.global_step)
            self.tensorboard_writer.add_scalar('Step/Learning_Rate', current_lr, self.global_step)
    
    def _final_evaluation(self):
        """Perform final evaluation on test set."""
        logger.info("Performing final evaluation on test set...")
        
        # Load best model if available
        best_model_path = self.output_dir / "best_model.pt"
        if best_model_path.exists():
            self.lora_model.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            logger.info("Loaded best model for final evaluation")
        
        # Test evaluation
        test_metrics = CrisisMetrics()
        test_loss, test_results = self._validate_epoch(test_metrics)
        
        logger.info("Final Test Results:")
        logger.info(f"  Test Loss: {test_loss:.4f}")
        logger.info(f"  Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"  Test Macro F1: {test_results['macro_f1']:.4f}")
        logger.info(f"  Test Weighted F1: {test_results['weighted_f1']:.4f}")
        
        # Crisis-specific metrics
        crisis_metrics = test_metrics.get_crisis_specific_metrics()
        logger.info("Crisis-Specific Metrics:")
        for metric, value in crisis_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save test results
        test_results_path = self.output_dir / "test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump({**test_results, **crisis_metrics}, f, indent=2, default=str)
        
        # Plot confusion matrix
        if self.visualizer:
            self.visualizer.plot_confusion_matrix(
                test_results['confusion_matrix'],
                epoch=self.current_epoch,
                show=False
            )
    
    def _save_final_model(self):
        """Save final model and artifacts."""
        logger.info("Saving final model and artifacts...")
        
        # Save LoRA adapters
        lora_path = self.output_dir / "lora_adapters"
        self.lora_model.save_lora_adapters(lora_path)
        
        # Save merged model
        merged_model = self.lora_model.merge_and_unload()
        torch.save(merged_model.state_dict(), self.output_dir / "merged_model.pt")
        
        # Save tokenizer
        tokenizer_path = self.output_dir / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save visualizer artifacts
        if self.visualizer:
            self.visualizer.save_training_history()
        
        logger.info(f"Model and artifacts saved to {self.output_dir}")
    
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def __del__(self):
        """Cleanup when trainer is destroyed."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Crisis Classification Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/processed",
        help="Directory containing train.csv, validation.csv, test.csv"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/models",
        help="Directory to save model and artifacts"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
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
        default=2,
        help="Gradient accumulation steps"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
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
    
    # System
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser


def main():
    """Main training function."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration
    config_path = 'configs/best_hyperparams.json' if Path('configs/best_hyperparams.json').exists() else 'configs/enhanced_training_config.json'
    if args.config:
        config = TrainingConfig.from_file(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = TrainingConfig.from_file(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    
    # Initialize trainer
    trainer = CrisisTrainer(config)
    
    try:
        # Setup training
        trainer.setup()
        
        # Start training
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer._save_final_model()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        trainer._cleanup_memory()


if __name__ == "__main__":
    main()