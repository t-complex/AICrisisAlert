#!/usr/bin/env python3
"""
Comprehensive Test for Enhanced Training Setup

This script thoroughly tests all components of the enhanced training pipeline
to identify and fix any issues before running the full training.
"""

import sys
import os
from pathlib import Path
import torch
import json
import logging
import traceback
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports."""
    logger.info("�� Testing imports...")
    
    try:
        # Test core imports
        import torch
        import transformers
        import peft
        import numpy as np
        import pandas as pd
        import sklearn
        import tqdm
        
        logger.info("✅ Core imports successful")
        
        # Test local imports
        from models.model_loader import load_crisis_classifier, ModelConfig
        from models.lora_setup import setup_lora_model, LoRAConfiguration
        from training.trainer_utils import CrisisLoss, CrisisMetrics, EarlyStopping
        from training.dataset_utils import create_data_loaders
        
        logger.info("✅ Local imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_bertweet_model():
    """Test BERTweet model loading and tokenization."""
    logger.info("🧪 Testing BERTweet model...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        logger.info(f"✅ BERTweet tokenizer loaded (vocab size: {tokenizer.vocab_size})")
        
        # Test tokenization
        test_text = "Emergency help needed! #crisis @emergency_team"
        tokens = tokenizer(
            test_text, 
            max_length=128, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        logger.info(f"✅ Tokenization test passed (input shape: {tokens['input_ids'].shape})")
        
        # Test model config
        model_config = AutoModel.from_pretrained("vinai/bertweet-base", config_only=True)
        logger.info("✅ BERTweet model configuration validated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BERTweet test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_enhanced_config():
    """Test enhanced configuration loading and validation."""
    logger.info("🧪 Testing enhanced configuration...")
    
    try:
        from training.enhanced_train import EnhancedTrainingConfig
        
        # Test loading from file
        config = EnhancedTrainingConfig.from_file("configs/enhanced_training_config.json")
        logger.info(f"✅ Configuration loaded: {config.experiment_name}")
        
        # Validate key settings
        assert config.model_name == "vinai/bertweet-base", "Model should be BERTweet"
        assert config.loss_function == "focal_loss", "Should use focal loss"
        assert config.epochs == 5, "Should have 5 epochs"
        assert config.learning_rate == 1e-5, "Learning rate should be 1e-5"
        assert config.max_length <= 128, "BERTweet max length should be <= 128"
        
        logger.info("✅ Enhanced configuration validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_focal_loss():
    """Test focal loss implementation."""
    logger.info(" Testing focal loss...")
    
    try:
        from training.trainer_utils import CrisisLoss
        
        # Test focal loss creation
        crisis_loss = CrisisLoss(num_classes=6, device=torch.device('cpu'))
        focal_loss = crisis_loss.focal_loss(gamma=2.0)
        
        # Test forward pass
        batch_size = 8
        num_classes = 6
        dummy_logits = torch.randn(batch_size, num_classes)
        dummy_targets = torch.randint(0, num_classes, (batch_size,))
        
        loss = focal_loss(dummy_logits, dummy_targets)
        
        logger.info(f"✅ Focal loss test passed (loss: {loss.item():.4f})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Focal loss test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_lora_setup():
    """Test LoRA setup with BERTweet."""
    logger.info(" Testing LoRA setup...")
    
    try:
        from models.model_loader import load_crisis_classifier
        from models.lora_setup import setup_lora_model, LoRAConfiguration
        
        # Load base model
        model, tokenizer, config = load_crisis_classifier(
            model_name="vinai/bertweet-base",
            max_length=128
        )
        logger.info("✅ Base model loaded")
        
        # Create LoRA configuration
        lora_config = LoRAConfiguration(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            gradient_checkpointing=True
        )
        logger.info("✅ LoRA configuration created")
        
        # Apply LoRA
        lora_model = setup_lora_model(model, lora_config)
        logger.info("✅ LoRA applied successfully")
        
        # Test efficiency metrics
        metrics = lora_model.get_efficiency_metrics()
        logger.info(f"✅ LoRA efficiency: {metrics['memory_reduction']:.2%} parameter reduction")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LoRA test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_data_loading():
    """Test data loading with leak-free datasets."""
    logger.info("🧪 Testing data loading...")
    
    try:
        from training.dataset_utils import create_data_loaders
        
        # Test data loading
        train_loader, val_loader, test_loader, class_weights = create_data_loaders(
            train_csv_path="data/processed/train_balanced_leak_free.csv",
            val_csv_path="data/processed/validation_balanced_leak_free.csv",
            test_csv_path="data/processed/test_balanced_leak_free.csv",
            tokenizer_name="vinai/bertweet-base",
            max_length=128,
            batch_size=4,  # Small batch for testing
            num_workers=0,  # No workers for testing
            apply_augmentation=False,  # No augmentation for testing
            use_balanced_sampling=False  # No sampling for testing
        )
        
        logger.info(f"✅ Data loaders created:")
        logger.info(f"  Train samples: {len(train_loader.dataset)}")
        logger.info(f"  Validation samples: {len(val_loader.dataset)}")
        logger.info(f"  Test samples: {len(test_loader.dataset)}")
        logger.info(f"  Class weights: {class_weights.tolist()}")
        
        # Test a single batch
        for batch in train_loader:
            logger.info(f"✅ Batch test passed (batch size: {batch['input_ids'].shape[0]})")
            break
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_full_pipeline():
    """Test the complete enhanced training pipeline setup."""
    logger.info("🧪 Testing full pipeline setup...")
    
    try:
        from training.enhanced_train import EnhancedTrainingConfig, EnhancedCrisisTrainer
        
        # Create configuration
        config = EnhancedTrainingConfig(
            data_dir="data/processed",
            output_dir="outputs/models/test_bertweet_enhanced",
            model_name="vinai/bertweet-base",
            max_length=128,
            epochs=1,  # Just 1 epoch for testing
            batch_size=4,  # Small batch for testing
            use_wandb=False,
            use_tensorboard=False,
            experiment_name="test_enhanced_pipeline"
        )
        
        # Create trainer
        trainer = EnhancedCrisisTrainer(config)
        logger.info("✅ Trainer created")
        
        # Setup components
        trainer.setup()
        logger.info("✅ Trainer setup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all comprehensive tests."""
    logger.info("🚀 Comprehensive Enhanced Training Setup Validation")
    logger.info("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("BERTweet Model", test_bertweet_model),
        ("Enhanced Configuration", test_enhanced_config),
        ("Focal Loss", test_focal_loss),
        ("LoRA Setup", test_lora_setup),
        ("Data Loading", test_data_loading),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 50)
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
                results.append((test_name, "PASSED", None))
            else:
                logger.info(f"❌ {test_name}: FAILED")
                results.append((test_name, "FAILED", "Test returned False"))
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, "ERROR", str(e)))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"🎯 COMPREHENSIVE VALIDATION SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All tests PASSED - Enhanced training setup is ready!")
        logger.info("🚀 You can now run: python src/training/enhanced_train.py --config configs/enhanced_training_config.json --experiment_name "bertweet_focal_enhanced"")
    else:
        logger.info(f"❌ {total - passed} tests FAILED - Setup needs attention")
        logger.info("\nDetailed Results:")
        for test_name, status, error in results:
            if status != "PASSED":
                logger.info(f"  ❌ {test_name}: {status}")
                if error:
                    logger.info(f"     Error: {error}")
    
    logger.info("=" * 80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 