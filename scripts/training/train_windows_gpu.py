#!/usr/bin/env python3
"""
Windows-specific training script for AICrisisAlert with GPU optimization.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.gpu_config import GPUConfig, setup_windows_training
from src.training.enhanced_train import EnhancedCrisisTrainer, EnhancedTrainingConfig  # Fixed import (was EnhancedTrainer)
from src.utils.config import get_settings

def main():
    """Main training function for Windows GPU setup."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/windows_training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Windows GPU training setup...")
    
    try:
        # Setup Windows-specific optimizations
        setup_windows_training()
        
        # Initialize GPU configuration
        gpu_config = GPUConfig()
        training_config = gpu_config.get_training_config()
        
        logger.info("GPU Configuration:")
        logger.info(f"Device: {training_config['device']}")
        logger.info(f"GPU Info: {training_config['gpu_info']}")
        logger.info(f"Training Config: {training_config['training_config']}")
        
        # Get settings
        settings = get_settings()

        # Build EnhancedTrainingConfig from settings and training_config
        config = EnhancedTrainingConfig(
            data_dir=str(settings.MODEL_PATH),
            output_dir=str(settings.MODEL_PATH),
            batch_size=training_config['training_config']['batch_size'],
            num_workers=training_config['training_config']['num_workers'],
            pin_memory=training_config['training_config']['pin_memory'],
            use_mixed_precision=training_config['training_config']['mixed_precision'],
            use_gradient_checkpointing=training_config['training_config']['gradient_checkpointing']
        )

        # Initialize trainer with config
        trainer = EnhancedCrisisTrainer(config)
        
        # Start training
        logger.info("Starting training with Windows GPU optimizations...")
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()