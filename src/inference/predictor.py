"""
Crisis prediction module for AICrisisAlert.

This module provides the CrisisPredictor class for performing 
crisis classification predictions using trained models.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Tuple, List
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class CrisisPredictor:
    """
    Crisis classification predictor.
    
    Handles prediction logic for crisis classification models,
    combining text inputs with extracted features.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize the predictor.
        
        Args:
            model: The loaded crisis classification model
            tokenizer: The tokenizer for text preprocessing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device if model else torch.device('cpu')
        
        # Crisis categories mapping
        self.label_to_category = {
            0: "urgent_help",
            1: "infrastructure_damage", 
            2: "casualty_info",
            3: "resource_availability",
            4: "general_info"
        }
        
        logger.info(f"CrisisPredictor initialized on device: {self.device}")
        
    async def predict_with_features(self, text: str, features: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """
        Perform prediction combining text and extracted features.
        
        Args:
            text: Input text to classify
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (predicted_category, probability_dict)
        """
        try:
            # Tokenize input text
            inputs = self._tokenize_text(text)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=-1)
                
                # Get predicted class
                predicted_idx = torch.argmax(probabilities, dim=-1).item()
                predicted_category = self.label_to_category.get(predicted_idx, "general_info")
                
                # Convert probabilities to dictionary
                prob_dict = {}
                for idx, category in self.label_to_category.items():
                    if idx < probabilities.shape[-1]:
                        prob_dict[category] = probabilities[0][idx].item()
                    else:
                        prob_dict[category] = 0.0
                
                # Enhance prediction with features if available
                enhanced_probs = self._enhance_with_features(prob_dict, features)
                
                # Get final prediction
                final_prediction = max(enhanced_probs, key=enhanced_probs.get)
                
                logger.debug(f"Prediction: {final_prediction}, confidence: {enhanced_probs[final_prediction]:.3f}")
                
                return final_prediction, enhanced_probs
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return safe default
            default_probs = {category: 0.2 for category in self.label_to_category.values()}
            return "general_info", default_probs
    
    async def predict_batch(self, texts: List[str], features_list: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, float]]]:
        """
        Perform batch prediction.
        
        Args:
            texts: List of input texts
            features_list: List of feature dictionaries
            
        Returns:
            List of (predicted_category, probability_dict) tuples
        """
        try:
            results = []
            
            # Process in batches for efficiency
            batch_size = 8
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_features = features_list[i:i + batch_size] if features_list else [{} for _ in batch_texts]
                
                # Process batch
                batch_results = await asyncio.gather(*[
                    self.predict_with_features(text, features)
                    for text, features in zip(batch_texts, batch_features)
                ])
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Return safe defaults
            default_probs = {category: 0.2 for category in self.label_to_category.values()}
            return [("general_info", default_probs) for _ in texts]
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text for model inference.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary of tokenized inputs
        """
        try:
            # Tokenize with truncation and padding
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to correct device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            return encoding
            
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            # Return minimal tokenization
            return {
                'input_ids': torch.tensor([[0]], device=self.device),
                'attention_mask': torch.tensor([[1]], device=self.device)
            }
    
    def _enhance_with_features(self, base_probs: Dict[str, float], features: Dict[str, Any]) -> Dict[str, float]:
        """
        Enhance base probabilities using extracted features.
        
        Args:
            base_probs: Base probability scores from model
            features: Extracted features dictionary
            
        Returns:
            Enhanced probability scores
        """
        enhanced_probs = base_probs.copy()
        
        try:
            # Feature-based adjustments
            if features.get('emergency_keywords_count', 0) > 0:
                enhanced_probs['urgent_help'] *= 1.2
                
            if features.get('infrastructure_keywords_count', 0) > 0:
                enhanced_probs['infrastructure_damage'] *= 1.15
                
            if features.get('casualty_indicators_count', 0) > 0:
                enhanced_probs['casualty_info'] *= 1.25
                
            if features.get('location_mentions_count', 0) > 0:
                # Location mentions often indicate urgent situations
                enhanced_probs['urgent_help'] *= 1.1
                enhanced_probs['infrastructure_damage'] *= 1.05
            
            # Normalize probabilities
            total = sum(enhanced_probs.values())
            if total > 0:
                enhanced_probs = {k: v / total for k, v in enhanced_probs.items()}
            
        except Exception as e:
            logger.warning(f"Feature enhancement failed: {e}")
            # Return original probabilities if enhancement fails
            return base_probs
        
        return enhanced_probs
    
    def get_prediction_info(self) -> Dict[str, Any]:
        """
        Get information about the predictor configuration.
        
        Returns:
            Dictionary with predictor information
        """
        return {
            'model_type': type(self.model).__name__ if self.model else 'Unknown',
            'device': str(self.device),
            'categories': list(self.label_to_category.values()),
            'tokenizer_type': type(self.tokenizer).__name__ if self.tokenizer else 'Unknown'
        }