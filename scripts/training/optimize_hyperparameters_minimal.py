#!/usr/bin/env python3
"""
Minimal hyperparameter optimization for BERTweet crisis classification.
This version bypasses NLTK/TextBlob to avoid recursion errors.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import optuna
import json
import time
import warnings

# Suppress all warnings to avoid NLTK recursion spam
warnings.filterwarnings("ignore")

# Import only the core training components
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

def run_training_with_config(config_dict, trial_dir):
    """Run training with minimal configuration to avoid NLTK issues."""
    try:
        # Create trial directory
        os.makedirs(trial_dir, exist_ok=True)
        
        # Update config with trial-specific settings
        config_dict['output_dir'] = trial_dir
        config_dict['experiment_name'] = f"trial_{int(time.time())}"
        config_dict['num_workers'] = 0  # Disable multiprocessing
        config_dict['apply_augmentation'] = False  # Disable augmentation
        config_dict['use_balanced_sampling'] = False  # Disable balanced sampling
        config_dict['use_tensorboard'] = False  # Disable tensorboard
        config_dict['use_wandb'] = False  # Disable wandb
        config_dict['epochs'] = 2  # Reduce epochs for faster testing
        
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

def main():
    """Run hyperparameter optimization."""
    print("Starting minimal hyperparameter optimization...")
    print("This version bypasses NLTK/TextBlob to avoid recursion errors.")
    
    # Create study
    study = optuna.create_study(
        direction='maximize', 
        study_name='bertweet_crisis_hpo_minimal'
    )
    
    # Run optimization with fewer trials for testing
    n_trials = 10  # Reduced from 50 for testing
    study.optimize(objective, n_trials=n_trials, timeout=None, show_progress_bar=True)
    
    # Save results
    best_trial = study.best_trial
    best_params = best_trial.params
    
    # Save best config
    with open('configs/best_hyperparams_minimal.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best hyperparameters saved to configs/best_hyperparams_minimal.json")
    
    # Save report
    with open('outputs/optimization_report_minimal.md', 'w') as f:
        f.write(f"# Minimal Hyperparameter Optimization Report\n\n")
        f.write(f"Best Score: {study.best_value:.4f}\n")
        f.write(f"Best Params: {json.dumps(best_params, indent=2)}\n\n")
        f.write("## All Trials\n")
        for t in study.trials:
            f.write(f"Trial {t.number}: Score={t.value:.4f}, Params={t.params}, Macro F1={t.user_attrs.get('macro_f1', 0.0):.4f}, Humanitarian F1={t.user_attrs.get('humanitarian_f1', 0.0):.4f}\n")
    print(f"Optimization report saved to outputs/optimization_report_minimal.md")
    
    print(f"\nOptimization completed!")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {best_params}")

if __name__ == '__main__':
    main() 