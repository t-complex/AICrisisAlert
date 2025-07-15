#!/usr/bin/env python3
"""
Test Enhanced Training Setup

Validates that all components are properly configured for the enhanced
BERTweet training with focal loss and leak-free datasets.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import json
import logging
from transformers import AutoTokenizer, AutoConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_config():
    """Test enhanced configuration loading."""
    logger.info("🧪 Testing enhanced configuration...")
    
    config_path = "configs/enhanced_training_config.json"
    
    if not Path(config_path).exists():
        logger.error(f"❌ Configuration file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate key configuration
    assert config['model_name'] == 'vinai/bertweet-base', "Model should be BERTweet"
    assert config['loss_function'] == 'focal_loss', "Should use focal loss"
    assert config['epochs'] == 5, "Should have 5 epochs"
    assert config['learning_rate'] == 1e-5, "Learning rate should be 1e-5"
    assert config['gradient_accumulation_steps'] == 4, "Should have gradient accumulation"
    assert config['use_leak_free_datasets'] == True, "Should use leak-free datasets"
    
    logger.info("✅ Enhanced configuration validated")
    return True

def test_bertweet_model():
    """Test BERTweet model loading."""
    logger.info("🧪 Testing BERTweet model loading...")
    
    try:
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        logger.info(f"✅ BERTweet tokenizer loaded (vocab size: {tokenizer.vocab_size})")
        
        # Test model loading (just check structure, don't load weights)
        model_config = AutoConfig.from_pretrained("vinai/bertweet-base")
        logger.info("✅ BERTweet model configuration validated")
        
        # Test tokenization
        test_text = "Emergency help needed! #crisis @emergency_team https://example.com"
        tokens = tokenizer(test_text, max_length=256, padding=True, truncation=True, return_tensors="pt")
        logger.info(f"✅ Tokenization test passed (input shape: {tokens['input_ids'].shape})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BERTweet model test failed: {e}")
        return False

def test_leak_free_datasets():
    """Test leak-free dataset availability."""
    logger.info("🧪 Testing leak-free datasets...")
    
    data_dir = Path("data/processed")
    required_files = [
        "train_balanced_leak_free.csv",
        "validation_balanced_leak_free.csv", 
        "test_balanced_leak_free.csv",
        "dataset_recreation_report.json"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            logger.info(f"✅ Found: {file_name}")
    
    if missing_files:
        logger.error(f"❌ Missing leak-free dataset files: {missing_files}")
        return False
    
    # Test dataset integrity
    try:
        import pandas as pd
        train_df = pd.read_csv(data_dir / "train_balanced_leak_free.csv")
        val_df = pd.read_csv(data_dir / "validation_balanced_leak_free.csv")
        test_df = pd.read_csv(data_dir / "test_balanced_leak_free.csv")
        
        logger.info(f"✅ Train samples: {len(train_df):,}")
        logger.info(f"✅ Validation samples: {len(val_df):,}")
        logger.info(f"✅ Test samples: {len(test_df):,}")
        
        # Verify class balance
        train_class_counts = train_df['label'].value_counts()
        logger.info(f"✅ Train class balance: {train_class_counts.min()}-{train_class_counts.max()} samples per class")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset integrity test failed: {e}")
        return False

def test_focal_loss():
    """Test focal loss implementation."""
    logger.info("🧪 Testing focal loss implementation...")
    
    try:
        sys.path.append("src/training")
        from trainer_utils import CrisisLoss
        
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
        return False

def test_enhanced_training_imports():
    """Test enhanced training script imports."""
    logger.info("🧪 Testing enhanced training imports...")
    
    try:
        sys.path.append("src/training")
        from enhanced_train import EnhancedTrainingConfig, EnhancedCrisisTrainer
        
        # Test configuration creation
        config = EnhancedTrainingConfig()
        logger.info(f"✅ Enhanced config created: {config.experiment_name}")
        
        logger.info("✅ Enhanced training imports successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced training imports failed: {e}")
        return False

def test_system_requirements():
    """Test system requirements for enhanced training."""
    logger.info("🧪 Testing system requirements...")
    
    # Check PyTorch
    logger.info(f"✅ PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.info("ℹ️  No GPU available - will use CPU")
    
    # Check key packages
    required_packages = ['transformers', 'sklearn', 'pandas', 'numpy', 'tqdm']
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} available")
        except ImportError:
            logger.error(f"❌ {package} not available")
            return False
    
    return True

def main():
    """Run all tests for enhanced training setup."""
    logger.info("🚀 Enhanced Training Setup Validation")
    logger.info("=" * 60)
    
    tests = [
        ("System Requirements", test_system_requirements),
        ("Enhanced Configuration", test_enhanced_config),
        ("BERTweet Model", test_bertweet_model),
        ("Leak-Free Datasets", test_leak_free_datasets),
        ("Focal Loss", test_focal_loss),
        ("Enhanced Training Imports", test_enhanced_training_imports),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.info(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"🎯 VALIDATION SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All tests PASSED - Enhanced training setup is ready!")
        logger.info("🚀 You can now run: ./scripts/run_enhanced_training.sh")
    else:
        logger.info(f"❌ {total - passed} tests FAILED - Setup needs attention")
    
    logger.info("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)