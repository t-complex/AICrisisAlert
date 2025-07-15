#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuner for BERTweet crisis classification.

This module provides automated hyperparameter optimization using Optuna
to maximize crisis classification performance.
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress NLTK warnings and recursion errors
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")
warnings.filterwarnings("ignore", message=".*recursion.*", category=RuntimeWarning)

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from typing import Dict, Any
import os
import time

# Import EnhancedTrainingConfig for config manipulation
from src.training.enhanced_train import EnhancedTrainingConfig, EnhancedCrisisTrainer

# Search space for Optuna
HYPERPARAM_SPACE = {
    'learning_rate': (1e-6, 5e-5),
    'batch_size': [8, 16, 32],
    'lora_rank': [4, 8, 16, 32],
    'lora_alpha': [16, 32, 64],
    'warmup_steps': [100, 500, 1000],
    'weight_decay': [0.01, 0.1],
    'lora_dropout': [0.1, 0.3],
}

# Metrics to optimize
PRIMARY_METRIC = 'macro_f1'
SECONDARY_METRIC = 'humanitarian_f1'


def suggest_hyperparams(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', *HYPERPARAM_SPACE['learning_rate'], log=True),
        'batch_size': trial.suggest_categorical('batch_size', HYPERPARAM_SPACE['batch_size']),
        'lora_rank': trial.suggest_categorical('lora_rank', HYPERPARAM_SPACE['lora_rank']),
        'lora_alpha': trial.suggest_categorical('lora_alpha', HYPERPARAM_SPACE['lora_alpha']),
        'warmup_steps': trial.suggest_categorical('warmup_steps', HYPERPARAM_SPACE['warmup_steps']),
        'weight_decay': trial.suggest_categorical('weight_decay', HYPERPARAM_SPACE['weight_decay']),
        'lora_dropout': trial.suggest_categorical('lora_dropout', HYPERPARAM_SPACE['lora_dropout']),
    }


def run_training_with_config(config_dict: Dict[str, Any], trial_dir: str) -> Dict[str, float]:
    """
    Run the enhanced training pipeline with the given config dict.
    Returns the validation metrics (macro F1, humanitarian F1, etc.).
    """
    try:
        # Create trial directory
        os.makedirs(trial_dir, exist_ok=True)
        
        # Update config with trial-specific settings
        config_dict['output_dir'] = trial_dir
        config_dict['experiment_name'] = f"trial_{int(time.time())}"
        config_dict['num_workers'] = 0  # Disable multiprocessing to avoid NLTK issues
        config_dict['apply_augmentation'] = False  # Disable augmentation to avoid NLTK
        config_dict['use_balanced_sampling'] = False  # Disable balanced sampling to avoid NLTK
        
        # Create config object
        config = EnhancedTrainingConfig(**config_dict)
        
        # Initialize trainer
        trainer = EnhancedCrisisTrainer(config)
        trainer.setup()
        
        # Run training
        print(f"Starting trial with config: {config_dict}")
        trainer.train()
        
        # Get best validation metrics
        best_metrics = trainer.get_best_validation_metrics()
        
        if best_metrics is None:
            print("No validation metrics found, using default values")
            return {'macro_f1': 0.0, 'humanitarian_f1': 0.0}
        
        print(f"Found validation metrics: {best_metrics}")
        return best_metrics
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return {'macro_f1': 0.0, 'humanitarian_f1': 0.0}


def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    try:
        # Load base config
        base_config = EnhancedTrainingConfig.from_file('configs/enhanced_training_config.json')
        config_dict = base_config.__dict__.copy()
        
        # Suggest hyperparameters
        config_dict.update(suggest_hyperparams(trial))
        
        # Use a unique output dir for each trial
        trial_dir = f"outputs/models/hyperopt_trial_{trial.number}"
        
        # Run training
        metrics = run_training_with_config(config_dict, trial_dir)
        
        macro_f1 = metrics.get('macro_f1', 0.0)
        humanitarian_f1 = metrics.get('humanitarian_f1', 0.0)
        
        # Penalty for poor minority class performance
        penalty = 0.0
        if humanitarian_f1 < 0.7:
            penalty = 0.1 * (0.7 - humanitarian_f1)
        
        # Main objective: maximize macro F1, with secondary humanitarian F1 and penalty
        score = macro_f1 - penalty + 0.05 * humanitarian_f1
        
        # Store metrics for reporting
        trial.set_user_attr('macro_f1', macro_f1)
        trial.set_user_attr('humanitarian_f1', humanitarian_f1)
        
        print(f"Trial {trial.number}: Score={score:.4f}, Macro F1={macro_f1:.4f}, Humanitarian F1={humanitarian_f1:.4f}")
        
        return score
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        return 0.0 