"""
Model service layer for AICrisisAlert API.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

from .models import (
    CrisisClassificationResponse,
    CrisisCategory
)
from .config import get_settings

logger = logging.getLogger(__name__)

class ModelService:
    """Service for handling crisis classification model operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self.is_initialized = False
        self.model_info = {
            "model_name": "bertweet-crisis-classifier",
            "model_version": self.settings.model_version,
            "model_type": self.settings.model_type,
            "classes": [cat.value for cat in CrisisCategory],
            "accuracy": 0.84,  # Will be updated from actual model
            "f1_score": 0.78,  # Will be updated from actual model
            "loaded_at": None,
            "last_updated": None
        }
    
    async def initialize(self):
        """Initialize the model and related components."""
        try:
            logger.info("Initializing model service")
            
            # Import here to avoid circular imports
            from src.models.model_loader import ModelLoader
            from src.utils.crisis_features import CrisisFeatureExtractor
            
            # Load the model
            model_loader = ModelLoader()
            self.model, self.tokenizer = await model_loader.load_model_async(
                model_path=self.settings.model_path
            )
            
            # Initialize feature extractor
            self.feature_extractor = CrisisFeatureExtractor()
            
            # Update model info
            self.model_info["loaded_at"] = datetime.now().isoformat()
            self.model_info["last_updated"] = datetime.now().isoformat()
            
            self.is_initialized = True
            logger.info("Model service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            logger.info("Cleaning up model service")
            
            # Clear GPU memory if model was on GPU
            if self.model is not None:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
                
            if self.tokenizer is not None:
                del self.tokenizer
                
            if self.feature_extractor is not None:
                del self.feature_extractor
            
            # Clear references
            self.model = None
            self.tokenizer = None
            self.feature_extractor = None
            self.is_initialized = False
            
            # Force memory cleanup
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("CUDA cache cleared during model service cleanup")
            except ImportError:
                pass
                
            logger.info("Model service cleanup completed")
        except Exception as e:
            logger.error(f"Error during model service cleanup: {e}")
    
    async def classify_text(self, text: str) -> CrisisClassificationResponse:
        """Classify a single text input."""
        if not self.is_initialized:
            raise RuntimeError("Model service not initialized")
        
        start_time = time.time()
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Extract features
            features = self.feature_extractor.extract_features(processed_text)
            
            # Perform classification
            prediction, probabilities = await self._predict(processed_text, features)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create response
            response = CrisisClassificationResponse(
                predicted_class=CrisisCategory(prediction),
                confidence=probabilities[prediction],
                class_probabilities=probabilities,
                processing_time_ms=processing_time_ms,
                model_version=self.settings.model_version,
                features_used=list(features.keys())
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
    
    async def classify_batch(self, texts: List[str]) -> List[CrisisClassificationResponse]:
        """Classify multiple text inputs."""
        if not self.is_initialized:
            raise RuntimeError("Model service not initialized")
        
        if len(texts) > self.settings.batch_size_limit:
            raise ValueError(f"Batch size exceeds limit of {self.settings.batch_size_limit}")
        
        start_time = time.time()
        results = []
        
        try:
            # Process texts in parallel
            tasks = [self.classify_text(text) for text in texts]
            results = await asyncio.gather(*tasks)
            
            total_time = (time.time() - start_time) * 1000
            logger.info(f"Batch classification completed: {len(texts)} texts in {total_time:.2f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            raise
    
    async def classify_emergency(self, text: str) -> CrisisClassificationResponse:
        """Emergency classification with priority processing."""
        if not self.is_initialized:
            raise RuntimeError("Model service not initialized")
        
        start_time = time.time()
        
        try:
            # Emergency classification with higher priority
            result = await self.classify_text(text)
            
            # Check if confidence meets emergency threshold
            if result.confidence >= self.settings.emergency_threshold:
                logger.warning(f"High-confidence emergency classification: {result.predicted_class} ({result.confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Emergency classification failed: {e}")
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.model_info.copy()
    
    async def log_classification(self, text: str):
        """Log classification for analytics (background task)."""
        try:
            # This would typically log to a database or analytics service
            logger.info("Classification logged for analytics", text_length=len(text))
        except Exception as e:
            logger.error(f"Failed to log classification: {e}")
    
    async def log_batch_classification(self, batch_size: int):
        """Log batch classification for analytics (background task)."""
        try:
            # This would typically log to a database or analytics service
            logger.info("Batch classification logged for analytics", batch_size=batch_size)
        except Exception as e:
            logger.error(f"Failed to log batch classification: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for classification."""
        # Basic preprocessing
        text = text.strip()
        if len(text) > self.settings.max_text_length:
            text = text[:self.settings.max_text_length]
        
        return text
    
    async def _predict(self, text: str, features: Dict[str, Any]) -> tuple:
        """Perform prediction using the loaded model."""
        try:
            # Import here to avoid circular imports
            from src.inference.predictor import CrisisPredictor
            
            predictor = CrisisPredictor(self.model, self.tokenizer)
            
            # Combine text and features for prediction
            prediction, probabilities = await predictor.predict_with_features(text, features)
            
            return prediction, probabilities
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

# Global model service instance
_model_service = None

def get_model_service() -> ModelService:
    """Get the global model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service 