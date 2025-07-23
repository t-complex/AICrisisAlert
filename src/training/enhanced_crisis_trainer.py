import logging
import dataclasses
import torch
from pathlib import Path
from src.training.configs import EnhancedTrainingConfig
from training.trainer_utils import TrainingVisualizer
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class EnhancedCrisisTrainer:
    """
    Enhanced training orchestrator for BERTweet-based crisis classification.
    Features advanced training techniques including focal loss, sophisticated
    scheduling, and optimizations for social media text understanding.
    """
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.output_dir = Path(config.output_dir)
        self.visualizer = None
        self.wandb_run = None
        self.tensorboard_writer = None
        self._setup_monitoring()
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
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.training_history = []
        logger.info(f"EnhancedCrisisTrainer initialized for experiment: {config.experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA available: Using GPU {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("MPS available: Using Apple Silicon GPU")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU available: Using CPU (training will be slow)")
        return device

    def _setup_monitoring(self):
        self.visualizer = TrainingVisualizer(
            save_dir=str(self.output_dir / "logs"),
            log_level='INFO'
        )
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
        if self.config.use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project="crisis-classification-enhanced",
                name=self.config.experiment_name,
                config=dataclasses.asdict(self.config),
                reinit=True
            )
            logger.info("Weights & Biases initialized")
        elif self.config.use_wandb:
            logger.warning("Weights & Biases requested but not available")
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tensorboard_writer = SummaryWriter(
                log_dir=str(self.output_dir / "tensorboard")
            )
            logger.info("TensorBoard initialized")
        elif self.config.use_tensorboard:
            logger.warning("TensorBoard requested but not available")

    def setup(self):
        logger.info("Setting up enhanced training components...")
        self._set_seeds()
        self._setup_data()
        self._setup_model()
        self._setup_optimization()
        # Save configuration

    def _set_seeds(self):
        import random
        import numpy as np
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        logger.info(f"Random seeds set to {self.config.seed}")

    def _setup_data(self):
        from training.dataset_utils import create_data_loaders
        self.train_loader, self.val_loader, self.test_loader, self.class_weights = create_data_loaders(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            use_balanced_sampling=self.config.use_balanced_sampling,
            apply_augmentation=self.config.apply_augmentation,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        logger.info(f"Data loaders created. Train: {len(self.train_loader)}, Val: {len(self.val_loader)}, Test: {len(self.test_loader)}")

    def _setup_model(self):
        # Placeholder for model setup logic
        logger.info("Model setup not implemented in this stub.")

    def _setup_optimization(self):
        # Placeholder for optimization setup logic
        logger.info("Optimization setup not implemented in this stub.")

    def train(self):
        logger.info("Training not implemented in this stub.")

    def get_best_validation_metrics(self) -> Dict[str, float]:
        return {}

    def _train_epoch(self, metrics) -> Tuple[float, Dict]:
        return 0.0, {}

    def _validate_epoch(self, metrics) -> Tuple[float, Dict]:
        return 0.0, {}

    def _log_epoch_results(self, epoch: int, train_loss: float, val_loss: float, train_results: Dict, val_results: Dict, learning_rate: float, epoch_time: float):
        pass

    def _log_training_step(self, loss: float, learning_rate: float):
        pass

    def _final_evaluation(self):
        pass

    def _save_final_model(self):
        pass

    def _cleanup_memory(self):
        """Clean up GPU memory and other resources."""
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("CUDA cache cleared")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear any large objects that might be holding memory
            if hasattr(self, 'model') and self.model is not None:
                # Move model to CPU to free GPU memory
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                    
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                # Clear optimizer state if needed
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None:
                            param.grad = None
                            
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.warning(f"Memory cleanup encountered an error: {e}")

    def __del__(self):
        """Destructor to ensure cleanup when object is deleted."""
        try:
            self._cleanup_memory()
        except Exception as e:
            # Use print instead of logger as logger might not be available during destruction
            print(f"Warning: Error during EnhancedCrisisTrainer destruction: {e}") 