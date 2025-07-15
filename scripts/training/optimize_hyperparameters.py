#!/usr/bin/env python3
"""
Optuna-based hyperparameter optimization for BERTweet crisis classification.

This script runs automated hyperparameter optimization using Optuna to find
the best configuration for achieving 87%+ accuracy on crisis classification.
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
from src.training.hyperparameter_tuner import objective, suggest_hyperparams

# Number of trials
N_TRIALS = 50
BEST_CONFIG_PATH = 'configs/best_hyperparams.json'
REPORT_PATH = 'outputs/optimization_report.md'


def save_best_config(study):
    best_trial = study.best_trial
    best_params = best_trial.params
    # Save as config compatible with EnhancedTrainingConfig
    with open(BEST_CONFIG_PATH, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best hyperparameters saved to {BEST_CONFIG_PATH}")


def save_report(study):
    with open(REPORT_PATH, 'w') as f:
        f.write(f"# Hyperparameter Optimization Report\n\n")
        f.write(f"Best Score: {study.best_value:.4f}\n")
        f.write(f"Best Params: {json.dumps(study.best_trial.params, indent=2)}\n\n")
        f.write("## All Trials\n")
        for t in study.trials:
            f.write(f"Trial {t.number}: Score={t.value:.4f}, Params={t.params}, Macro F1={t.user_attrs.get('macro_f1', 0.0):.4f}, Humanitarian F1={t.user_attrs.get('humanitarian_f1', 0.0):.4f}\n")
    print(f"Optimization report saved to {REPORT_PATH}")


def main():
    study = optuna.create_study(direction='maximize', study_name='bertweet_crisis_hpo')
    study.optimize(objective, n_trials=N_TRIALS, timeout=None, show_progress_bar=True)
    save_best_config(study)
    save_report(study)
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()

if __name__ == '__main__':
    main() 