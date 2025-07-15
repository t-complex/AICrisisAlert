#!/usr/bin/env python3
"""
Crisis-Specific Preprocessing

This module provides enhanced preprocessing specifically designed for crisis classification,
including named entity recognition, sentiment analysis, text normalization, and feature scaling.

Features:
- Named entity recognition for locations and organizations
- Sentiment analysis for urgency detection
- Text normalization preserving crisis terminology
- Feature scaling and normalization
- Crisis-specific text cleaning
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import spacy
from textblob import TextBlob
import nltk

def safe_nltk_download():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('omw-1.4')

safe_nltk_download()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrisisPreprocessor:
    """
    Enhanced crisis-specific text preprocessor.
    
    Provides specialized preprocessing for crisis classification including
    NER, sentiment analysis, and crisis terminology preservation.
    """
    
    def __init__(self, preserve_crisis_terms: bool = True):
        """
        Initialize crisis preprocessor.
        
        Args:
            preserve_crisis_terms: Whether to preserve crisis-specific terminology
        """
        self.preserve_crisis_terms = preserve_crisis_terms
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.warning("SpaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Crisis-specific patterns to preserve
        self.crisis_patterns = [
            r'#\w+',  # Hashtags
            r'@\w+',  # Mentions
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
            r'\b(?:urgent|emergency|critical|asap|help|rescue|trapped|injured|dead|missing)\b',  # Crisis terms
            r'\b(?:hurricane|earthquake|fire|flood|storm|tornado)\b',  # Disaster types
            r'\b(?:hospital|police|ambulance|firefighter|paramedic)\b',  # Emergency services
        ]
        
        # Compile patterns
        self.crisis_regex = re.compile('|'.join(self.crisis_patterns), re.IGNORECASE)
        
        logger.info("CrisisPreprocessor initialized")
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Preprocess text with crisis-specific enhancements.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Dictionary containing processed text and extracted information
        """
        if not text or not isinstance(text, str):
            return {
                'processed_text': '',
                'entities': {},
                'sentiment': {'polarity': 0.0, 'subjectivity': 0.0},
                'urgency_score': 0.0,
                'crisis_indicators': []
            }
        
        # Extract entities before cleaning
        entities = self._extract_entities(text)
        
        # Extract sentiment
        sentiment = self._analyze_sentiment(text)
        
        # Calculate urgency score
        urgency_score = self._calculate_urgency_score(text)
        
        # Extract crisis indicators
        crisis_indicators = self._extract_crisis_indicators(text)
        
        # Clean text while preserving crisis terms
        processed_text = self._clean_text(text)
        
        return {
            'processed_text': processed_text,
            'entities': entities,
            'sentiment': sentiment,
            'urgency_score': urgency_score,
            'crisis_indicators': crisis_indicators
        }
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using SpaCy."""
        try:
            doc = self.nlp(text)
            
            entities = {
                'locations': [],
                'organizations': [],
                'persons': [],
                'dates': [],
                'numbers': []
            }
            
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC', 'FAC']:
                    entities['locations'].append(ent.text)
                elif ent.label_ == 'ORG':
                    entities['organizations'].append(ent.text)
                elif ent.label_ == 'PERSON':
                    entities['persons'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                elif ent.label_ == 'CARDINAL':
                    entities['numbers'].append(ent.text)
            
            return entities
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {'locations': [], 'organizations': [], 'persons': [], 'dates': [], 'numbers': []}
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            return {
                'polarity': float(blob.sentiment.polarity),
                'subjectivity': float(blob.sentiment.subjectivity)
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on crisis indicators."""
        urgency_terms = [
            'urgent', 'emergency', 'critical', 'immediate', 'asap', 'now',
            'hurry', 'rush', 'quick', 'fast', 'instant', 'right away'
        ]
        
        urgency_count = sum(text.lower().count(term) for term in urgency_terms)
        exclamation_count = text.count('!')
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Weighted urgency score
        urgency_score = urgency_count * 2.0 + exclamation_count * 0.5 + uppercase_ratio * 3.0
        
        return float(urgency_score)
    
    def _extract_crisis_indicators(self, text: str) -> List[str]:
        """Extract crisis-specific indicators from text."""
        indicators = []
        
        # Find crisis patterns
        matches = self.crisis_regex.findall(text)
        indicators.extend(matches)
        
        # Add specific crisis terms
        crisis_terms = [
            'hurricane', 'earthquake', 'fire', 'flood', 'storm', 'tornado',
            'hospital', 'police', 'ambulance', 'firefighter', 'paramedic',
            'trapped', 'injured', 'dead', 'missing', 'help', 'rescue'
        ]
        
        for term in crisis_terms:
            if term.lower() in text.lower():
                indicators.append(term)
        
        return list(set(indicators))  # Remove duplicates
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving crisis-specific terms."""
        # Store crisis terms before cleaning
        crisis_terms = {}
        for match in self.crisis_regex.finditer(text):
            crisis_terms[match.group()] = f"CRISIS_TERM_{len(crisis_terms)}"
        
        # Replace crisis terms with placeholders
        for term, placeholder in crisis_terms.items():
            text = text.replace(term, placeholder)
        
        # Basic text cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation except spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Restore crisis terms
        for term, placeholder in crisis_terms.items():
            text = text.replace(placeholder, term.lower())
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessing results
        """
        results = []
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Preprocessing text {i+1}/{len(texts)}")
            
            result = self.preprocess_text(text)
            results.append(result)
        
        return results


class CrisisFeatureScaler:
    """
    Feature scaler specifically designed for crisis features.
    
    Provides scaling and normalization for crisis-specific features
    with appropriate handling of different feature types.
    """
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize crisis feature scaler.
        
        Args:
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
        """
        self.scaling_method = scaling_method
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")
        
        self.is_fitted = False
        logger.info(f"CrisisFeatureScaler initialized with {scaling_method} scaling")
    
    def fit(self, X: pd.DataFrame) -> 'CrisisFeatureScaler':
        """
        Fit the scaler to the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Self for chaining
        """
        self.scaler.fit(X)
        self.is_fitted = True
        logger.info(f"Scaler fitted on {X.shape[0]} samples with {X.shape[1]} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transformation")
        
        scaled_X = self.scaler.transform(X)
        return pd.DataFrame(scaled_X, columns=X.columns, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled features.
        
        Args:
            X: Scaled feature matrix
            
        Returns:
            Original scale feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transformation")
        
        original_X = self.scaler.inverse_transform(X)
        return pd.DataFrame(original_X, columns=X.columns, index=X.index)


class CrisisFeatureSelector:
    """
    Feature selector for crisis-specific features.
    
    Provides feature selection methods appropriate for crisis classification
    including statistical tests and domain knowledge.
    """
    
    def __init__(self, selection_method: str = 'kbest', k: int = 50):
        """
        Initialize crisis feature selector.
        
        Args:
            selection_method: Selection method ('kbest', 'percentile', 'threshold')
            k: Number of features to select
        """
        self.selection_method = selection_method
        self.k = k
        self.selector = None
        self.selected_features = None
        self.feature_scores = None
        
        logger.info(f"CrisisFeatureSelector initialized with {selection_method} method, k={k}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CrisisFeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Self for chaining
        """
        if self.selection_method == 'kbest':
            self.selector = SelectKBest(score_func=f_classif, k=min(self.k, X.shape[1]))
        else:
            raise ValueError(f"Unsupported selection method: {self.selection_method}")
        
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        self.feature_scores = dict(zip(X.columns, self.selector.scores_))
        
        logger.info(f"Feature selector fitted. Selected {len(self.selected_features)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted selector.
        
        Args:
            X: Feature matrix
            
        Returns:
            Selected feature matrix
        """
        if self.selector is None:
            raise ValueError("Feature selector must be fitted before transformation")
        
        selected_X = self.selector.transform(X)
        return pd.DataFrame(selected_X, columns=self.selected_features, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform features.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Selected feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature names and importance scores
        """
        if self.feature_scores is None:
            raise ValueError("Feature selector must be fitted before getting importance")
        
        return dict(sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True))


class CrisisPreprocessingPipeline:
    """
    Complete preprocessing pipeline for crisis classification.
    
    Combines text preprocessing, feature extraction, scaling, and selection
    into a unified pipeline.
    """
    
    def __init__(self, 
                 preprocessor: CrisisPreprocessor,
                 scaler: CrisisFeatureScaler,
                 selector: CrisisFeatureSelector):
        """
        Initialize preprocessing pipeline.
        
        Args:
            preprocessor: Crisis text preprocessor
            scaler: Feature scaler
            selector: Feature selector
        """
        self.preprocessor = preprocessor
        self.scaler = scaler
        self.selector = selector
        
        logger.info("CrisisPreprocessingPipeline initialized")
    
    def fit_transform(self, texts: List[str], y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform the complete pipeline.
        
        Args:
            texts: List of input texts
            y: Target variable
            
        Returns:
            Processed and selected feature matrix
        """
        # Preprocess texts
        logger.info("Preprocessing texts...")
        preprocessed = self.preprocessor.preprocess_batch(texts)
        
        # Extract features (assuming we have a feature extractor)
        # This would be integrated with CrisisFeatureExtractor
        logger.info("Feature extraction would be performed here...")
        
        # For now, create dummy features
        # In practice, this would use CrisisFeatureExtractor
        dummy_features = pd.DataFrame(np.random.rand(len(texts), 100))
        
        # Scale features
        logger.info("Scaling features...")
        scaled_features = self.scaler.fit_transform(dummy_features)
        
        # Select features
        logger.info("Selecting features...")
        selected_features = self.selector.fit_transform(scaled_features, y)
        
        logger.info(f"Pipeline completed. Final shape: {selected_features.shape}")
        return selected_features
    
    def transform(self, texts: List[str]) -> pd.DataFrame:
        """
        Transform new texts using fitted pipeline.
        
        Args:
            texts: List of input texts
            
        Returns:
            Processed and selected feature matrix
        """
        # Preprocess texts
        preprocessed = self.preprocessor.preprocess_batch(texts)
        
        # Extract features (dummy for now)
        dummy_features = pd.DataFrame(np.random.rand(len(texts), 100))
        
        # Scale features
        scaled_features = self.scaler.transform(dummy_features)
        
        # Select features
        selected_features = self.selector.transform(scaled_features)
        
        return selected_features


# Example usage
if __name__ == "__main__":
    # Initialize components
    preprocessor = CrisisPreprocessor()
    scaler = CrisisFeatureScaler()
    selector = CrisisFeatureSelector()
    
    # Create pipeline
    pipeline = CrisisPreprocessingPipeline(preprocessor, scaler, selector)
    
    # Example texts
    sample_texts = [
        "URGENT: Multiple people trapped in building collapse on Main Street! Need immediate rescue assistance. #emergency #rescue",
        "Hurricane Sandy causing widespread power outages in NYC. Hospitals running on generators.",
        "Just another day at the office. Nothing special happening."
    ]
    
    # Process texts
    for text in sample_texts:
        result = preprocessor.preprocess_text(text)
        print(f"\nOriginal: {text}")
        print(f"Processed: {result['processed_text']}")
        print(f"Urgency Score: {result['urgency_score']:.2f}")
        print(f"Entities: {result['entities']}")
        print(f"Crisis Indicators: {result['crisis_indicators']}") 