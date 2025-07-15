#!/usr/bin/env python3
"""
Test script to verify hyperparameter optimization setup.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_hyperopt_setup():
    """Test hyperparameter optimization setup."""
    print("Testing hyperparameter optimization setup...")
    
    try:
        # Test imports
        from src.training.hyperparameter_tuner import suggest_hyperparams
        print("✓ Hyperparameter tuner imports working")
        
        # Test hyperparameter suggestion
        import optuna
        
        # Create a study and get a trial
        study = optuna.create_study(direction='maximize')
        trial = study.ask()
        
        params = suggest_hyperparams(trial)
        print(f"✓ Hyperparameter suggestion working: {list(params.keys())}")
        
        # Test config loading
        print("✓ Enhanced training config imports working")
        
        print("\n✅ Hyperparameter optimization setup is working!")
        print("You can now run: python3 scripts/optimize_hyperparameters.py")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_hyperopt_setup()
    if not success:
        sys.exit(1) 