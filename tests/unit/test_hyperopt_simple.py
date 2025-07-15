#!/usr/bin/env python3
"""
Simple test script to verify hyperparameter optimization works.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_single_trial():
    """Test a single hyperparameter optimization trial."""
    print("Testing single hyperparameter optimization trial...")
    
    try:
        from src.training.hyperparameter_tuner import objective
        import optuna
        
        # Create a study
        study = optuna.create_study(direction='maximize')
        
        # Run a single trial
        trial = study.ask()
        score = objective(trial)
        
        print(f"Trial completed with score: {score}")
        print(f"Trial parameters: {trial.params}")
        print(f"Macro F1: {trial.user_attrs.get('macro_f1', 0.0)}")
        print(f"Humanitarian F1: {trial.user_attrs.get('humanitarian_f1', 0.0)}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_trial()
    if success:
        print("\n✅ Single trial test passed!")
        print("You can now run the full optimization: python3 scripts/optimize_hyperparameters.py")
    else:
        print("\n❌ Single trial test failed!")
        sys.exit(1) 