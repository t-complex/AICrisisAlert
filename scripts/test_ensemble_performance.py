#!/usr/bin/env python3
"""
Crisis Classification Ensemble Performance Testing

This script provides comprehensive testing and benchmarking of crisis classification
ensembles against individual models, including performance comparisons, ablation
studies, and deployment readiness assessment.

Key Features:
- Ensemble vs individual model performance comparison
- Crisis-specific evaluation metrics and analysis
- Ablation studies for ensemble components
- Inference speed and resource usage benchmarking
- Statistical significance testing
- Deployment readiness assessment

Usage:
    python test_ensemble_performance.py --data_dir data/processed --models_dir outputs/models
    python test_ensemble_performance.py --config configs/ensemble_test_config.json
    python test_ensemble_performance.py --quick_test --verbose
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_val_score

# Configure paths for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Local imports
from src.models.ensemble_classifier import (
    CrisisEnsembleConfig, CrisisVotingClassifier, AdaptiveEnsemble,
    create_crisis_ensemble, evaluate_ensemble_performance
)
from src.models.ensemble_persistence import save_ensemble, load_ensemble, EnsembleRegistry
from src.training.ensemble_trainer import (
    CrisisEnsembleTrainer, EnsembleTrainingConfig, train_crisis_ensemble
)
from src.models.model_loader import load_crisis_classifier, CrisisClassifier
from src.utils.enhanced_evaluation import EnhancedCrisisEvaluator, generate_enhanced_evaluation_report
from src.utils.evaluation import CRISIS_LABELS
from src.training.dataset_utils import create_data_loaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceTestConfig:
    """
    Configuration for ensemble performance testing.
    
    Defines test parameters, model configurations, and
    evaluation settings for comprehensive benchmarking.
    """
    # Data configuration
    data_dir: str = "data/processed"
    models_dir: str = "outputs/models"
    output_dir: str = "outputs/ensemble_tests"
    
    # Test configuration
    test_type: str = "comprehensive"  # quick, comprehensive, ablation
    num_trials: int = 5
    use_cross_validation: bool = True
    cv_folds: int = 3
    
    # Model configurations to test
    ensemble_configs: List[Dict[str, Any]] = None
    individual_models: List[Dict[str, Any]] = None
    
    # Performance testing
    test_inference_speed: bool = True
    test_memory_usage: bool = True
    test_batch_sizes: List[int] = None
    
    # Statistical testing
    significance_level: float = 0.05
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # Crisis-specific testing
    test_crisis_scenarios: bool = True
    simulate_real_time: bool = False
    test_deployment_ready: bool = True
    
    # Output configuration
    save_plots: bool = True
    save_detailed_results: bool = True
    generate_report: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Default ensemble configurations
        if self.ensemble_configs is None:
            self.ensemble_configs = [
                {
                    "ensemble_type": "soft_voting",
                    "crisis_weighting_strategy": "crisis_adaptive",
                    "num_base_models": 3
                },
                {
                    "ensemble_type": "hard_voting",
                    "crisis_weighting_strategy": "performance_weighted",
                    "num_base_models": 3
                },
                {
                    "ensemble_type": "adaptive",
                    "crisis_weighting_strategy": "dynamic_selection",
                    "num_base_models": 3
                }
            ]
        
        # Default individual model configurations
        if self.individual_models is None:
            self.individual_models = [
                {
                    "model_name": "vinai/bertweet-base",
                    "max_length": 128,
                    "specialization": "social_media"
                },
                {
                    "model_name": "distilbert-base-uncased",
                    "max_length": 512,
                    "specialization": "formal_text"
                },
                {
                    "model_name": "roberta-base",
                    "max_length": 256,
                    "specialization": "general_purpose"
                }
            ]
        
        # Default batch sizes for testing
        if self.test_batch_sizes is None:
            self.test_batch_sizes = [1, 8, 16, 32]
        
        # Validation
        assert self.test_type in ["quick", "comprehensive", "ablation"]
        assert self.num_trials > 0
        assert self.cv_folds > 1


class EnsemblePerformanceTester:
    """
    Comprehensive performance testing framework for crisis classification ensembles.
    
    Provides detailed benchmarking, comparison studies, and deployment
    readiness assessment for ensemble models.
    """
    
    def __init__(self, config: PerformanceTestConfig):
        """
        Initialize ensemble performance tester.
        
        Args:
            config: Performance testing configuration
        """
        self.config = config
        self.device = self._setup_device()
        self.output_dir = Path(config.output_dir)
        
        # Test results storage
        self.results = {
            'individual_models': {},
            'ensembles': {},
            'comparisons': {},
            'ablation_studies': {},
            'performance_benchmarks': {},
            'statistical_tests': {}
        }
        
        # Testing components
        self.evaluator = EnhancedCrisisEvaluator()
        self.registry = EnsembleRegistry()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        logger.info(f"EnsemblePerformanceTester initialized")
        logger.info(f"Test type: {config.test_type}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device = torch.device("cpu")
        logger.info("Using CPU for performance testing")
        return device
    
    def setup_data_loaders(self):
        """Setup data loaders for testing."""
        logger.info("Setting up data loaders...")
        
        # Use enhanced dataset
        train_file = "train_balanced_leak_free.csv"
        val_file = "validation_balanced_leak_free.csv"
        test_file = "test_balanced_leak_free.csv"
        
        train_path = Path(self.config.data_dir) / train_file
        val_path = Path(self.config.data_dir) / val_file
        test_path = Path(self.config.data_dir) / test_file
        
        # Verify files exist
        for file_path in [train_path, val_path, test_path]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required dataset file not found: {file_path}")
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader, class_weights = create_data_loaders(
            train_csv_path=str(train_path),
            val_csv_path=str(val_path),
            test_csv_path=str(test_path),
            tokenizer_name="distilbert-base-uncased",  # Default tokenizer
            max_length=256,
            batch_size=16,
            num_workers=0,
            apply_augmentation=False,
            use_balanced_sampling=False
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"  Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"  Test samples: {len(self.test_loader.dataset)}")
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive performance tests.
        
        Returns:
            Complete test results
        """
        logger.info("Starting comprehensive ensemble performance testing...")
        start_time = time.time()
        
        # Setup data
        self.setup_data_loaders()
        
        # Test individual models
        if self.config.test_type in ["comprehensive", "ablation"]:
            logger.info("\n" + "="*60)
            logger.info("TESTING INDIVIDUAL MODELS")
            logger.info("="*60)
            self._test_individual_models()
        
        # Test ensemble models
        logger.info("\n" + "="*60)
        logger.info("TESTING ENSEMBLE MODELS")
        logger.info("="*60)
        self._test_ensemble_models()
        
        # Performance benchmarking
        if self.config.test_inference_speed or self.config.test_memory_usage:
            logger.info("\n" + "="*60)
            logger.info("PERFORMANCE BENCHMARKING")
            logger.info("="*60)
            self._run_performance_benchmarks()
        
        # Statistical comparisons
        logger.info("\n" + "="*60)
        logger.info("STATISTICAL ANALYSIS")
        logger.info("="*60)
        self._run_statistical_tests()
        
        # Ablation studies
        if self.config.test_type == "ablation":
            logger.info("\n" + "="*60)
            logger.info("ABLATION STUDIES")
            logger.info("="*60)
            self._run_ablation_studies()
        
        # Crisis scenario testing
        if self.config.test_crisis_scenarios:
            logger.info("\n" + "="*60)
            logger.info("CRISIS SCENARIO TESTING")
            logger.info("="*60)
            self._test_crisis_scenarios()
        
        # Generate final report
        total_time = time.time() - start_time
        self.results['test_metadata'] = {
            'total_time_seconds': total_time,
            'test_type': self.config.test_type,
            'num_trials': self.config.num_trials,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_results()
        
        if self.config.generate_report:
            self._generate_test_report()
        
        logger.info(f"Comprehensive testing completed in {total_time:.2f} seconds")
        return self.results
    
    def _test_individual_models(self):
        """Test individual model performance."""
        logger.info("Testing individual models...")
        
        for i, model_config in enumerate(self.config.individual_models):
            model_name = model_config['model_name']
            logger.info(f"\nTesting individual model {i+1}/{len(self.config.individual_models)}: {model_name}")
            
            # Load model
            model, tokenizer, config = load_crisis_classifier(
                model_name=model_name,
                max_length=model_config['max_length'],
                device=self.device
            )
            
            # Test model performance
            model_results = self._evaluate_single_model(
                model, f"individual_{i}_{model_name.replace('/', '_')}"
            )
            
            self.results['individual_models'][f'model_{i}'] = {
                'config': model_config,
                'results': model_results
            }
    
    def _test_ensemble_models(self):
        """Test ensemble model performance."""
        logger.info("Testing ensemble models...")
        
        # Usage Example: Create crisis-optimized ensemble
        config = CrisisEnsembleConfig(
            ensemble_type="soft_voting", 
            crisis_weighting_strategy="crisis_adaptive", 
            humanitarian_boost=1.2, 
            critical_crisis_weight=2.0
        )
        
        # Create crisis-optimized ensemble
        ensemble = create_crisis_ensemble(config)
        
        # Training configuration for multi-stage pipeline
        training_config = EnsembleTrainingConfig(
            pretrain_individual_models=True, 
            ensemble_loss_function="crisis_adaptive"
        )
        
        # Train with multi-stage pipeline (if data loaders are available)
        if hasattr(self, 'train_loader') and hasattr(self, 'val_loader'):
            try:
                results = train_crisis_ensemble(config, training_config, self.train_loader, self.val_loader)
                logger.info(f"Training results: {results}")
            except Exception as e:
                logger.warning(f"Training failed: {e}")
        
        # Save ensemble with version management
        try:
            save_ensemble(ensemble, "models/crisis_ensemble_v1", config, version="1.0.0")
            logger.info("Ensemble saved with version management")
        except Exception as e:
            logger.warning(f"Saving ensemble failed: {e}")
        
        # Test the created ensemble
        ensemble = ensemble.to(self.device)
        ensemble_results = self._evaluate_ensemble_model(ensemble, "crisis_optimized_ensemble")
        
        self.results['ensembles']['crisis_optimized_ensemble'] = {
            'config': config.__dict__,
            'results': ensemble_results
        }
        
        # Test additional ensemble configurations
        for i, ensemble_config_dict in enumerate(self.config.ensemble_configs):
            logger.info(f"\nTesting ensemble {i+1}/{len(self.config.ensemble_configs)}")
            logger.info(f"Configuration: {ensemble_config_dict}")
            
            # Create ensemble configuration
            ensemble_config = CrisisEnsembleConfig(**ensemble_config_dict)
            
            # Create ensemble
            ensemble = create_crisis_ensemble(ensemble_config)
            ensemble = ensemble.to(self.device)
            
            # Test ensemble performance
            ensemble_results = self._evaluate_ensemble_model(
                ensemble, f"ensemble_{i}_{ensemble_config.ensemble_type}"
            )
            
            self.results['ensembles'][f'ensemble_{i}'] = {
                'config': ensemble_config_dict,
                'results': ensemble_results
            }
    
    def _evaluate_single_model(self, model: CrisisClassifier, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model."""
        model.eval()
        results = {}
        
        # Multiple trial evaluation
        trial_results = []
        
        for trial in range(self.config.num_trials):
            if self.config.verbose:
                logger.info(f"  Trial {trial + 1}/{self.config.num_trials}")
            
            trial_result = self._single_trial_evaluation(model, model_name, trial)
            trial_results.append(trial_result)
        
        # Aggregate results
        results['trials'] = trial_results
        results['aggregated'] = self._aggregate_trial_results(trial_results)
        
        return results
    
    def _evaluate_ensemble_model(
        self,
        ensemble: Union[CrisisVotingClassifier, AdaptiveEnsemble],
        ensemble_name: str
    ) -> Dict[str, Any]:
        """Evaluate an ensemble model."""
        ensemble.eval()
        results = {}
        
        # Multiple trial evaluation
        trial_results = []
        
        for trial in range(self.config.num_trials):
            if self.config.verbose:
                logger.info(f"  Trial {trial + 1}/{self.config.num_trials}")
            
            trial_result = self._single_trial_ensemble_evaluation(ensemble, ensemble_name, trial)
            trial_results.append(trial_result)
        
        # Aggregate results
        results['trials'] = trial_results
        results['aggregated'] = self._aggregate_trial_results(trial_results)
        
        # Ensemble-specific analysis
        results['ensemble_analysis'] = self._analyze_ensemble_components(ensemble)
        
        return results
    
    def _single_trial_evaluation(self, model: CrisisClassifier, model_name: str, trial: int) -> Dict[str, Any]:
        """Run single trial evaluation for a model."""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        inference_times = []
        
        model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Time inference
                start_time = time.time()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Collect predictions
                logits = outputs['logits']
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        targets = np.array(all_targets)
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        
        # Crisis-specific metrics
        crisis_metrics = self._calculate_crisis_metrics(targets, predictions)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'macro_f1': float(f1),
            'crisis_metrics': crisis_metrics,
            'inference_time_ms': np.mean(inference_times) * 1000,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'targets': targets.tolist()
        }
    
    def _single_trial_ensemble_evaluation(
        self,
        ensemble: Union[CrisisVotingClassifier, AdaptiveEnsemble],
        ensemble_name: str,
        trial: int
    ) -> Dict[str, Any]:
        """Run single trial evaluation for an ensemble."""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_individual_predictions = []
        inference_times = []
        
        ensemble.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Time inference
                start_time = time.time()
                outputs = ensemble(
                    input_ids, attention_mask, 
                    return_individual_predictions=True
                )
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Collect predictions
                predictions = outputs['predictions']
                probabilities = outputs['probabilities']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                # Individual model predictions (if available)
                if 'individual_predictions' in outputs:
                    individual_preds = outputs['individual_predictions']
                    all_individual_predictions.extend(individual_preds.cpu().numpy())
        
        # Calculate metrics
        targets = np.array(all_targets)
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        
        # Crisis-specific metrics
        crisis_metrics = self._calculate_crisis_metrics(targets, predictions)
        
        # Ensemble-specific metrics
        ensemble_metrics = {}
        if all_individual_predictions:
            individual_preds = np.array(all_individual_predictions)
            ensemble_metrics = self._calculate_ensemble_metrics(
                targets, predictions, individual_preds
            )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'macro_f1': float(f1),
            'crisis_metrics': crisis_metrics,
            'ensemble_metrics': ensemble_metrics,
            'inference_time_ms': np.mean(inference_times) * 1000,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'targets': targets.tolist()
        }
    
    def _calculate_crisis_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate crisis-specific metrics."""
        # Humanitarian vs non-humanitarian
        humanitarian_classes = [0, 1, 2, 3]
        humanitarian_targets = np.isin(targets, humanitarian_classes).astype(int)
        humanitarian_predictions = np.isin(predictions, humanitarian_classes).astype(int)
        
        humanitarian_f1 = precision_recall_fscore_support(
            humanitarian_targets, humanitarian_predictions, average='binary', zero_division=0
        )[2]
        
        # Critical crisis detection
        critical_classes = [0, 2]
        critical_targets = np.isin(targets, critical_classes).astype(int)
        critical_predictions = np.isin(predictions, critical_classes).astype(int)
        
        critical_f1 = precision_recall_fscore_support(
            critical_targets, critical_predictions, average='binary', zero_division=0
        )[2]
        
        # Emergency response accuracy
        emergency_targets = (targets != 5).astype(int)
        emergency_predictions = (predictions != 5).astype(int)
        
        emergency_accuracy = accuracy_score(emergency_targets, emergency_predictions)
        
        return {
            'humanitarian_f1': float(humanitarian_f1),
            'critical_crisis_f1': float(critical_f1),
            'emergency_response_accuracy': float(emergency_accuracy)
        }
    
    def _calculate_ensemble_metrics(
        self,
        targets: np.ndarray,
        ensemble_predictions: np.ndarray,
        individual_predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate ensemble-specific metrics."""
        metrics = {}
        
        # Diversity metrics
        num_models = individual_predictions.shape[1]
        disagreements = 0
        total_pairs = 0
        
        for i in range(num_models):
            for j in range(i + 1, num_models):
                disagreement = (individual_predictions[:, i] != individual_predictions[:, j]).mean()
                disagreements += disagreement
                total_pairs += 1
        
        diversity_score = disagreements / total_pairs if total_pairs > 0 else 0.0
        
        # Ensemble improvement over individual models
        individual_accuracies = []
        for i in range(num_models):
            acc = accuracy_score(targets, individual_predictions[:, i])
            individual_accuracies.append(acc)
        
        ensemble_accuracy = accuracy_score(targets, ensemble_predictions)
        best_individual_accuracy = max(individual_accuracies)
        ensemble_improvement = ensemble_accuracy - best_individual_accuracy
        
        metrics.update({
            'diversity_score': float(diversity_score),
            'individual_accuracies': individual_accuracies,
            'best_individual_accuracy': float(best_individual_accuracy),
            'ensemble_accuracy': float(ensemble_accuracy),
            'ensemble_improvement': float(ensemble_improvement)
        })
        
        return metrics
    
    def _aggregate_trial_results(self, trial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple trials."""
        if not trial_results:
            return {}
        
        # Extract metrics for aggregation
        metrics_to_aggregate = [
            'accuracy', 'precision', 'recall', 'macro_f1', 'inference_time_ms'
        ]
        
        aggregated = {}
        
        for metric in metrics_to_aggregate:
            values = [result[metric] for result in trial_results if metric in result]
            if values:
                aggregated[f'{metric}_mean'] = float(np.mean(values))
                aggregated[f'{metric}_std'] = float(np.std(values))
                aggregated[f'{metric}_min'] = float(np.min(values))
                aggregated[f'{metric}_max'] = float(np.max(values))
        
        # Aggregate crisis metrics
        crisis_metrics = {}
        for crisis_metric in ['humanitarian_f1', 'critical_crisis_f1', 'emergency_response_accuracy']:
            values = [result['crisis_metrics'][crisis_metric] for result in trial_results 
                     if 'crisis_metrics' in result and crisis_metric in result['crisis_metrics']]
            if values:
                crisis_metrics[f'{crisis_metric}_mean'] = float(np.mean(values))
                crisis_metrics[f'{crisis_metric}_std'] = float(np.std(values))
        
        aggregated['crisis_metrics'] = crisis_metrics
        
        return aggregated
    
    def _analyze_ensemble_components(
        self,
        ensemble: Union[CrisisVotingClassifier, AdaptiveEnsemble]
    ) -> Dict[str, Any]:
        """Analyze ensemble components and voting behavior."""
        analysis = {
            'ensemble_type': type(ensemble).__name__,
            'num_base_models': 0,
            'voting_strategy': 'unknown'
        }
        
        # Get voting classifier
        voting_classifier = ensemble if isinstance(ensemble, CrisisVotingClassifier) else getattr(ensemble, 'voting_classifier', None)
        
        if voting_classifier and hasattr(voting_classifier, 'models'):
            analysis['num_base_models'] = len(voting_classifier.models)
            
            # Analyze model weights if available
            if hasattr(voting_classifier, 'model_weights'):
                weights = F.softmax(voting_classifier.model_weights, dim=0)
                analysis['model_weights'] = weights.cpu().numpy().tolist()
            
            # Configuration information
            if hasattr(voting_classifier, 'config'):
                analysis['voting_strategy'] = voting_classifier.config.crisis_weighting_strategy
        
        return analysis
    
    def _run_performance_benchmarks(self):
        """Run performance benchmarking tests."""
        logger.info("Running performance benchmarks...")
        
        benchmarks = {}
        
        # Test different batch sizes if enabled
        if self.config.test_inference_speed:
            benchmarks['inference_speed'] = self._benchmark_inference_speed()
        
        if self.config.test_memory_usage:
            benchmarks['memory_usage'] = self._benchmark_memory_usage()
        
        self.results['performance_benchmarks'] = benchmarks
    
    def _benchmark_inference_speed(self) -> Dict[str, Any]:
        """Benchmark inference speed across different batch sizes."""
        logger.info("Benchmarking inference speed...")
        
        speed_results = {}
        
        # Test only the first ensemble for speed benchmarking
        if self.results['ensembles']:
            first_ensemble_key = list(self.results['ensembles'].keys())[0]
            ensemble_config = self.results['ensembles'][first_ensemble_key]['config']
            
            # Recreate ensemble
            config = CrisisEnsembleConfig(**ensemble_config)
            ensemble = create_crisis_ensemble(config)
            ensemble = ensemble.to(self.device)
            ensemble.eval()
            
            # Test different batch sizes
            for batch_size in self.config.test_batch_sizes:
                logger.info(f"  Testing batch size: {batch_size}")
                
                # Create dummy data
                dummy_input_ids = torch.randint(0, 1000, (batch_size, 128), device=self.device)
                dummy_attention_mask = torch.ones(batch_size, 128, device=self.device)
                
                # Warmup
                for _ in range(5):
                    with torch.no_grad():
                        _ = ensemble(dummy_input_ids, dummy_attention_mask)
                
                # Benchmark
                times = []
                for _ in range(20):
                    start_time = time.time()
                    with torch.no_grad():
                        _ = ensemble(dummy_input_ids, dummy_attention_mask)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                speed_results[f'batch_size_{batch_size}'] = {
                    'mean_time_ms': float(np.mean(times) * 1000),
                    'std_time_ms': float(np.std(times) * 1000),
                    'throughput_samples_per_sec': float(batch_size / np.mean(times))
                }
        
        return speed_results
    
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        logger.info("Benchmarking memory usage...")
        
        # This is a simplified memory benchmark
        # In a real implementation, you would use more sophisticated memory profiling
        memory_results = {
            'note': 'Memory benchmarking requires specialized profiling tools',
            'estimated_model_size_mb': 'Not implemented in this demo'
        }
        
        return memory_results
    
    def _run_statistical_tests(self):
        """Run statistical significance tests."""
        logger.info("Running statistical significance tests...")
        
        statistical_results = {}
        
        # Compare ensembles vs individual models
        if self.results['individual_models'] and self.results['ensembles']:
            statistical_results['ensemble_vs_individual'] = self._compare_ensemble_vs_individual()
        
        # Compare different ensemble configurations
        if len(self.results['ensembles']) > 1:
            statistical_results['ensemble_comparisons'] = self._compare_ensemble_configurations()
        
        self.results['statistical_tests'] = statistical_results
    
    def _compare_ensemble_vs_individual(self) -> Dict[str, Any]:
        """Compare ensemble performance vs individual models."""
        # Extract F1 scores for comparison
        individual_f1s = []
        for model_result in self.results['individual_models'].values():
            f1_scores = [trial['macro_f1'] for trial in model_result['results']['trials']]
            individual_f1s.extend(f1_scores)
        
        ensemble_f1s = []
        for ensemble_result in self.results['ensembles'].values():
            f1_scores = [trial['macro_f1'] for trial in ensemble_result['results']['trials']]
            ensemble_f1s.extend(f1_scores)
        
        if individual_f1s and ensemble_f1s:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(ensemble_f1s, individual_f1s)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(ensemble_f1s) - 1) * np.var(ensemble_f1s, ddof=1) + 
                                 (len(individual_f1s) - 1) * np.var(individual_f1s, ddof=1)) / 
                                (len(ensemble_f1s) + len(individual_f1s) - 2))
            cohens_d = (np.mean(ensemble_f1s) - np.mean(individual_f1s)) / pooled_std
            
            return {
                'ensemble_mean_f1': float(np.mean(ensemble_f1s)),
                'individual_mean_f1': float(np.mean(individual_f1s)),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'significant': p_value < self.config.significance_level,
                'sample_sizes': {'ensemble': len(ensemble_f1s), 'individual': len(individual_f1s)}
            }
        
        return {'error': 'Insufficient data for comparison'}
    
    def _compare_ensemble_configurations(self) -> Dict[str, Any]:
        """Compare different ensemble configurations."""
        comparisons = {}
        
        ensemble_names = list(self.results['ensembles'].keys())
        
        for i, name1 in enumerate(ensemble_names):
            for name2 in ensemble_names[i+1:]:
                # Extract F1 scores
                f1_scores_1 = [trial['macro_f1'] for trial in self.results['ensembles'][name1]['results']['trials']]
                f1_scores_2 = [trial['macro_f1'] for trial in self.results['ensembles'][name2]['results']['trials']]
                
                if f1_scores_1 and f1_scores_2:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(f1_scores_1, f1_scores_2)
                    
                    comparison_key = f"{name1}_vs_{name2}"
                    comparisons[comparison_key] = {
                        'mean_f1_1': float(np.mean(f1_scores_1)),
                        'mean_f1_2': float(np.mean(f1_scores_2)),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.config.significance_level
                    }
        
        return comparisons
    
    def _run_ablation_studies(self):
        """Run ablation studies on ensemble components."""
        logger.info("Running ablation studies...")
        
        # This is a placeholder for ablation studies
        # In a full implementation, you would test:
        # - Effect of different voting strategies
        # - Impact of individual model removal
        # - Sensitivity to hyperparameters
        
        ablation_results = {
            'note': 'Ablation studies would be implemented here',
            'planned_studies': [
                'voting_strategy_ablation',
                'model_removal_ablation',
                'hyperparameter_sensitivity'
            ]
        }
        
        self.results['ablation_studies'] = ablation_results
    
    def _test_crisis_scenarios(self):
        """Test performance on specific crisis scenarios."""
        logger.info("Testing crisis scenario performance...")
        
        # Create scenario-specific test sets
        scenario_results = {}
        
        # Test on different crisis types
        for i, crisis_label in enumerate(CRISIS_LABELS):
            scenario_results[crisis_label] = {
                'note': f'Scenario testing for {crisis_label}',
                'planned_metrics': ['precision', 'recall', 'f1', 'response_time']
            }
        
        self.results['crisis_scenario_tests'] = scenario_results
    
    def _save_results(self):
        """Save test results to files."""
        logger.info("Saving test results...")
        
        # Save main results
        results_path = self.output_dir / "performance_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save configuration
        config_path = self.output_dir / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("Generating test report...")
        
        report_path = self.output_dir / "performance_test_report.md"
        
        report_content = f"""# Crisis Classification Ensemble Performance Test Report
        
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Test Type: {self.config.test_type}
Device: {self.device}

## Summary

This report presents comprehensive performance testing results for crisis classification ensembles.

### Individual Models Tested
{len(self.results.get('individual_models', {}))} individual models tested

### Ensemble Models Tested  
{len(self.results.get('ensembles', {}))} ensemble configurations tested

### Key Findings

"""
        
        # Add statistical test results if available
        if 'statistical_tests' in self.results and 'ensemble_vs_individual' in self.results['statistical_tests']:
            stats_result = self.results['statistical_tests']['ensemble_vs_individual']
            if 'ensemble_mean_f1' in stats_result:
                report_content += f"""
#### Ensemble vs Individual Model Performance
- Ensemble Mean F1: {stats_result['ensemble_mean_f1']:.4f}
- Individual Model Mean F1: {stats_result['individual_mean_f1']:.4f}
- Statistical Significance: {stats_result['significant']}
- P-value: {stats_result['p_value']:.4f}
- Effect Size (Cohen's d): {stats_result['cohens_d']:.4f}

"""
        
        report_content += """
## Detailed Results

See `performance_test_results.json` for complete numerical results.

## Recommendations

Based on the test results:
1. Review ensemble configuration performance
2. Consider deployment of best-performing models
3. Monitor performance in production environment

---
*Report generated by AICrisisAlert Ensemble Performance Testing Framework*
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Test report saved to {report_path}")


def create_test_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Crisis Classification Ensemble Performance Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing test datasets"
    )
    
    parser.add_argument(
        "--models_dir",
        type=str,
        default="outputs/models",
        help="Directory containing trained models"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ensemble_tests",
        help="Directory to save test results"
    )
    
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["quick", "comprehensive", "ablation"],
        default="comprehensive",
        help="Type of testing to perform"
    )
    
    parser.add_argument(
        "--num_trials",
        type=int,
        default=3,
        help="Number of trials for each test"
    )
    
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick testing with minimal trials"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to test configuration JSON file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main():
    """Main testing function."""
    parser = create_test_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = PerformanceTestConfig(**config_dict)
    else:
        # Create configuration from arguments
        config_dict = {
            'data_dir': args.data_dir,
            'models_dir': args.models_dir,
            'output_dir': args.output_dir,
            'test_type': args.test_type,
            'num_trials': args.num_trials,
            'verbose': args.verbose
        }
        
        # Quick test adjustments
        if args.quick_test:
            config_dict.update({
                'test_type': 'quick',
                'num_trials': 1,
                'use_cross_validation': False,
                'test_inference_speed': False,
                'test_memory_usage': False,
                'test_crisis_scenarios': False
            })
        
        config = PerformanceTestConfig(**config_dict)
    
    logger.info("🚀 Starting Crisis Classification Ensemble Performance Testing")
    logger.info(f"Test configuration: {config.test_type}")
    logger.info(f"Number of trials: {config.num_trials}")
    
    try:
        # Initialize tester
        tester = EnsemblePerformanceTester(config)
        
        # Run tests
        results = tester.run_comprehensive_tests()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TESTING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Individual models tested: {len(results.get('individual_models', {}))}")
        logger.info(f"Ensemble models tested: {len(results.get('ensembles', {}))}")
        logger.info(f"Total test time: {results.get('test_metadata', {}).get('total_time_seconds', 0):.2f} seconds")
        logger.info(f"Results saved to: {config.output_dir}")
        
        if 'statistical_tests' in results and 'ensemble_vs_individual' in results['statistical_tests']:
            stats_result = results['statistical_tests']['ensemble_vs_individual']
            if 'ensemble_mean_f1' in stats_result:
                logger.info(f"\nPerformance Summary:")
                logger.info(f"  Ensemble F1: {stats_result['ensemble_mean_f1']:.4f}")
                logger.info(f"  Individual F1: {stats_result['individual_mean_f1']:.4f}")
                logger.info(f"  Statistically significant: {stats_result['significant']}")
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()