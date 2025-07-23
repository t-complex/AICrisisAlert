from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnhancedTrainingConfig:
    """
    Enhanced configuration class for BERTweet-based crisis classification.
    
    Contains optimized hyperparameters for social media crisis text
    classification with advanced training techniques.
    """
    # Data configuration
    data_dir: str = "data/processed"
    output_dir: str = "outputs/models/bertweet_enhanced"
    model_name: str = "vinai/bertweet-base"  # Social media optimized
    max_length: int = 128  # BERTweet's maximum supported length
    
    # Enhanced training hyperparameters
    epochs: int = 5  # Increased from 3
    learning_rate: float = 1e-5  # Reduced for better convergence
    batch_size: int = 16
    gradient_accumulation_steps: int = 4  # Increased for larger effective batch
    warmup_steps: int = 500  # Increased warmup
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Enhanced LoRA configuration
    lora_rank: int = 16  # Increased for better adaptation
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Advanced optimization settings
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_balanced_sampling: bool = True
    apply_augmentation: bool = True
    
    # Enhanced early stopping
    early_stopping_patience: int = 3
    early_stopping_monitor: str = "val_macro_f1"
    
    # Monitoring
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 500
    
    # Loss function configuration
    loss_function: str = "focal_loss"  # Changed from weighted_cross_entropy
    focal_loss_alpha: Optional[List[float]] = None  # Will be calculated from data
    focal_loss_gamma: float = 2.0  # Focus on hard examples
    class_weighting_method: str = "balanced"
    
    # Dataset configuration
    use_leak_free_datasets: bool = True
    
    # Experiment tracking
    use_wandb: bool = False
    use_tensorboard: bool = True
    experiment_name: Optional[str] = None
    
    # System settings
    num_workers: Optional[int] = None
    pin_memory: bool = True
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-configure experiment name
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"bertweet_enhanced_{timestamp}"
        
        # Validate configuration
        assert self.epochs > 0, "epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.loss_function in ["focal_loss", "weighted_cross_entropy", "label_smoothing"]
        
        # BERTweet-specific validation
        if "bertweet" in self.model_name.lower():
            if self.max_length > 128:
                logger.warning(f"BERTweet maximum length is 128, but {self.max_length} was specified. Setting to 128.")
                self.max_length = 128
        
        logger.info(f"EnhancedTrainingConfig initialized for experiment: {self.experiment_name}")
        logger.info(f"Using BERTweet model: {self.model_name}")
        logger.info(f"Loss function: {self.loss_function}")
        logger.info(f"Max length: {self.max_length}")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        # Security: Validate path to prevent directory traversal
        if not EnhancedTrainingConfig._is_safe_path(path):
            raise ValueError("Invalid file path - potential security risk")
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Enhanced training configuration saved to {path}")
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'EnhancedTrainingConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        # Security: Validate path to prevent directory traversal
        if not EnhancedTrainingConfig._is_safe_path(path):
            raise ValueError("Invalid file path - potential security risk")
            
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @staticmethod
    def _is_safe_path(path: Path) -> bool:
        """Validate that path is safe and doesn't contain directory traversal."""
        try:
            # Convert to absolute path and resolve any '..' components
            resolved_path = path.resolve()
            
            # Define allowed directories (adjust as needed for your project)
            allowed_dirs = [
                Path.cwd().resolve(),  # Current working directory
                Path.cwd().resolve() / "configs",  # Configs directory
                Path.cwd().resolve() / "outputs",  # Outputs directory
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

@dataclass
class EnsembleTrainingConfig:
    """
    Extended configuration for ensemble training with crisis-specific parameters.
    
    Includes multi-stage training settings, optimization strategies, and
    crisis-specific objectives for emergency response systems.
    """
    # Multi-stage training
    pretrain_individual_models: bool = True
    individual_training_epochs: int = 3
    ensemble_training_epochs: int = 2
    fine_tuning_epochs: int = 1
    
    # Training strategies
    training_strategy: str = "sequential"  # sequential, parallel, alternating
    use_curriculum_learning: bool = True
    curriculum_difficulty_threshold: float = 0.8
    
    # Optimization parameters
    individual_learning_rate: float = 2e-5
    ensemble_learning_rate: float = 1e-4
    fine_tuning_learning_rate: float = 5e-6
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # Crisis-specific training
    crisis_loss_weight: float = 1.5
    humanitarian_boost: float = 1.2
    critical_crisis_weight: float = 2.0
    false_positive_penalty: float = 0.8
    
    # Loss function configuration
    ensemble_loss_function: str = "crisis_adaptive"  # crisis_adaptive, weighted_ensemble, knowledge_distillation
    temperature_scaling: bool = True
    initial_temperature: float = 3.0
    
    # Validation and early stopping
    validation_strategy: str = "ensemble_aware"  # standard, ensemble_aware, cross_validation
    early_stopping_patience: int = 3
    early_stopping_monitor: str = "ensemble_macro_f1"
    min_improvement: float = 0.001
    
    # Hyperparameter optimization
    optimize_hyperparameters: bool = False
    optimization_trials: int = 20
    optimization_metric: str = "macro_f1"
    
    # Data augmentation and sampling
    use_data_augmentation: bool = True
    augmentation_strength: float = 0.1
    use_adaptive_sampling: bool = True
    sampling_strategy: str = "difficulty_based"
    
    # Performance thresholds
    min_individual_performance: float = 0.75
    target_ensemble_improvement: float = 0.03
    diversity_threshold: float = 0.2
    
    # System and monitoring
    parallel_training: bool = True
    max_workers: int = 3
    save_checkpoints: bool = True
    checkpoint_frequency: int = 500
    verbose_logging: bool = True
    
    # Output configuration
    output_dir: str = "outputs/ensemble_training"
    experiment_name: Optional[str] = None
    save_intermediate_models: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-configure experiment name
        if self.experiment_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"ensemble_training_{timestamp}"
        
        # Validate configuration
        assert self.training_strategy in ["sequential", "parallel", "alternating"]
        assert self.ensemble_loss_function in ["crisis_adaptive", "weighted_ensemble", "knowledge_distillation"]
        assert self.validation_strategy in ["standard", "ensemble_aware", "cross_validation"]
        assert self.sampling_strategy in ["difficulty_based", "uncertainty_based", "balanced"]
        
        logger.info(f"EnsembleTrainingConfig initialized for experiment: {self.experiment_name}") 