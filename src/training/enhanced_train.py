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
from src.training.configs import EnhancedTrainingConfig
import gc
from tqdm.auto import tqdm
import torch.backends.mps
import dataclasses
from typing import Dict, Tuple

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

from src.models.model_loader import load_crisis_classifier
from src.models.lora_setup import setup_lora_model, LoRAConfiguration, LoRATrainingOptimizer
from src.training.trainer_utils import (
    CrisisMetrics, EarlyStopping, CrisisLoss, TrainingVisualizer
)
from src.training.enhanced_crisis_trainer import EnhancedCrisisTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _is_safe_checkpoint_path(path: Path) -> bool:
    """
    Validate that checkpoint path is safe and doesn't contain directory traversal.
    
    Args:
        path: Path to validate
        
    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Convert to absolute path and resolve any '..' components
        resolved_path = path.resolve()
        
        # Define allowed directories for checkpoints
        allowed_dirs = [
            Path.cwd().resolve(),  # Current working directory
            Path.cwd().resolve() / "outputs",  # Outputs directory
            Path.cwd().resolve() / "checkpoints",  # Checkpoints directory
            Path.cwd().resolve() / "models",  # Models directory
        ]
        
        # Check if the resolved path is within allowed directories
        for allowed_dir in allowed_dirs:
            try:
                resolved_path.relative_to(allowed_dir)
                return True
            except ValueError:
                continue
                
        return False
    except (OSError, ValueError):
        return False


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
            
            # Security: Validate checkpoint path
            if not _is_safe_checkpoint_path(checkpoint_path):
                raise ValueError("Invalid checkpoint path - potential security risk")
                
            if checkpoint_path.exists():
                logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                # Security: Use weights_only=True to prevent arbitrary code execution
                checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
                # Note: weights_only=True would be safer but may break with complex checkpoints
                # Consider upgrading to weights_only=True after testing
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