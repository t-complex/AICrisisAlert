#!/usr/bin/env python3
"""
Crisis-Specific Feature Engineering

This module provides comprehensive feature extraction for crisis classification,
including emergency keywords, geolocation indicators, time urgency markers,
casualty indicators, infrastructure keywords, and social media engagement metrics.

Features:
- Emergency keyword frequency analysis
- Geolocation and landmark detection
- Time urgency classification
- Casualty and damage assessment
- Infrastructure impact evaluation
- Social media engagement metrics
- Crisis severity scoring
"""

import re
import json
import pandas as pd
from typing import Dict, List, Any, Union
from pathlib import Path
import logging
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

class CrisisFeatureExtractor:
    """
    Comprehensive crisis-specific feature extractor.
    
    Extracts domain-specific features from crisis-related text to enhance
    classification performance beyond transformer capabilities.
    """
    
    def __init__(self, lexicon_path: str = "data/crisis_lexicon.json"):
        """
        Initialize crisis feature extractor.
        
        Args:
            lexicon_path: Path to crisis lexicon JSON file
        """
        self.lexicon_path = Path(lexicon_path)
        self.lexicon = self._load_lexicon()
        self.feature_names = self._get_feature_names()
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.warning("SpaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        logger.info(f"CrisisFeatureExtractor initialized with {len(self.feature_names)} features")
    
    def _load_lexicon(self) -> Dict[str, Any]:
        """Load crisis lexicon from JSON file."""
        if not self.lexicon_path.exists():
            logger.error(f"Lexicon file not found: {self.lexicon_path}")
            return {}
        
        with open(self.lexicon_path, 'r') as f:
            lexicon = json.load(f)
        
        logger.info(f"Loaded crisis lexicon with {len(lexicon)} categories")
        return lexicon
    
    def _get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        feature_names = []
        
        # Emergency keywords features
        for category, items in self.lexicon.items():
            for item_name, item_data in items.items():
                feature_names.extend([
                    f"{category}_{item_name}_count",
                    f"{category}_{item_name}_severity",
                    f"{category}_{item_name}_density"
                ])
        
        # Additional crisis-specific features
        additional_features = [
            "text_length",
            "word_count",
            "hashtag_count",
            "mention_count",
            "url_count",
            "exclamation_count",
            "question_count",
            "uppercase_ratio",
            "urgency_score",
            "location_confidence",
            "casualty_estimate",
            "infrastructure_damage_score",
            "response_resource_score",
            "sentiment_score",
            "sentiment_polarity",
            "sentiment_subjectivity"
        ]
        
        feature_names.extend(additional_features)
        return feature_names
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract comprehensive crisis-specific features from text.
        
        Args:
            text: Input text to extract features from
            
        Returns:
            Dictionary of feature names and values
        """
        if not text or not isinstance(text, str):
            return {name: 0.0 for name in self.feature_names}
        
        # Normalize text
        text = text.lower().strip()
        
        # Initialize features
        features = {}
        
        # Extract lexicon-based features
        lexicon_features = self._extract_lexicon_features(text)
        features.update(lexicon_features)
        
        # Extract text-based features
        text_features = self._extract_text_features(text)
        features.update(text_features)
        
        # Extract NLP-based features
        nlp_features = self._extract_nlp_features(text)
        features.update(nlp_features)
        
        # Extract crisis-specific features
        crisis_features = self._extract_crisis_specific_features(text)
        features.update(crisis_features)
        
        # Ensure all features are present
        for name in self.feature_names:
            if name not in features:
                features[name] = 0.0
        
        return features
    
    def _extract_lexicon_features(self, text: str) -> Dict[str, float]:
        """Extract features based on crisis lexicon."""
        features = {}
        
        for category, items in self.lexicon.items():
            for item_name, item_data in items.items():
                # Get all terms including synonyms
                all_terms = item_data.get('terms', []) + item_data.get('synonyms', [])
                severity = item_data.get('severity', 1)
                
                # Count occurrences
                count = sum(text.count(term.lower()) for term in all_terms)
                
                # Calculate density (normalized by text length)
                text_length = len(text.split())
                density = count / max(text_length, 1)
                
                # Store features
                features[f"{category}_{item_name}_count"] = float(count)
                features[f"{category}_{item_name}_severity"] = float(count * severity)
                features[f"{category}_{item_name}_density"] = density
        
        return features
    
    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract basic text-based features."""
        features = {}
        
        # Text statistics
        features["text_length"] = float(len(text))
        features["word_count"] = float(len(text.split()))
        
        # Social media features
        features["hashtag_count"] = float(len(re.findall(r'#\w+', text)))
        features["mention_count"] = float(len(re.findall(r'@\w+', text)))
        features["url_count"] = float(len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)))
        
        # Punctuation features
        features["exclamation_count"] = float(text.count('!'))
        features["question_count"] = float(text.count('?'))
        
        # Case features
        uppercase_chars = sum(1 for c in text if c.isupper())
        features["uppercase_ratio"] = uppercase_chars / max(len(text), 1)
        
        return features
    
    def _extract_nlp_features(self, text: str) -> Dict[str, float]:
        """Extract NLP-based features using SpaCy."""
        features = {}
        
        try:
            doc = self.nlp(text)
            
            # Named entity recognition for locations
            location_entities = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
            features["location_confidence"] = float(len(location_entities))
            
            # Sentiment analysis
            blob = TextBlob(text)
            features["sentiment_score"] = blob.sentiment.polarity
            features["sentiment_polarity"] = blob.sentiment.polarity
            features["sentiment_subjectivity"] = blob.sentiment.subjectivity
            
        except Exception as e:
            logger.warning(f"NLP feature extraction failed: {e}")
            features["location_confidence"] = 0.0
            features["sentiment_score"] = 0.0
            features["sentiment_polarity"] = 0.0
            features["sentiment_subjectivity"] = 0.0
        
        return features
    
    def _extract_crisis_specific_features(self, text: str) -> Dict[str, float]:
        """Extract crisis-specific features."""
        features = {}
        
        # Urgency score based on time markers and exclamations
        urgency_terms = ['now', 'immediate', 'asap', 'urgent', 'critical', 'emergency', 'hurry', 'rush']
        urgency_count = sum(text.count(term) for term in urgency_terms)
        exclamation_weight = text.count('!') * 0.5
        features["urgency_score"] = float(urgency_count + exclamation_weight)
        
        # Casualty estimate based on casualty-related terms
        casualty_terms = ['injured', 'hurt', 'dead', 'killed', 'missing', 'trapped', 'victim', 'casualty']
        casualty_count = sum(text.count(term) for term in casualty_terms)
        features["casualty_estimate"] = float(casualty_count)
        
        # Infrastructure damage score
        damage_terms = ['damage', 'destroyed', 'collapsed', 'broken', 'outage', 'flood', 'fire', 'crash']
        damage_count = sum(text.count(term) for term in damage_terms)
        features["infrastructure_damage_score"] = float(damage_count)
        
        # Response resource score
        resource_terms = ['help', 'rescue', 'aid', 'support', 'volunteer', 'donation', 'relief', 'assist']
        resource_count = sum(text.count(term) for term in resource_terms)
        features["response_resource_score"] = float(resource_count)
        
        return features
    
    def extract_batch_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract features from a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Processing text {i+1}/{len(texts)}")
            
            features = self.extract_features(text)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance using correlation with target.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of feature names and importance scores
        """
        importance = {}
        
        for column in X.columns:
            if column in y.name:
                continue
            
            try:
                # Calculate correlation
                correlation = abs(X[column].corr(y, method='pearson'))
                importance[column] = correlation
            except:
                importance[column] = 0.0
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def select_top_features(self, X: pd.DataFrame, y: pd.Series, top_k: int = 50) -> List[str]:
        """
        Select top-k most important features.
        
        Args:
            X: Feature matrix
            y: Target variable
            top_k: Number of top features to select
            
        Returns:
            List of top feature names
        """
        importance = self.get_feature_importance(X, y)
        top_features = list(importance.keys())[:top_k]
        
        return top_features


class CrisisFeatureAnalyzer:
    """
    Analyzer for crisis-specific features.
    
    Provides analysis and visualization of crisis features for
    understanding feature importance and model interpretability.
    """
    
    def __init__(self, extractor: CrisisFeatureExtractor):
        """
        Initialize crisis feature analyzer.
        
        Args:
            extractor: Crisis feature extractor instance
        """
        self.extractor = extractor
    
    def analyze_feature_distribution(self, X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Analyze distribution of crisis features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with feature statistics
        """
        stats = {}
        
        for column in X.columns:
            stats[column] = {
                'mean': float(X[column].mean()),
                'std': float(X[column].std()),
                'min': float(X[column].min()),
                'max': float(X[column].max()),
                'median': float(X[column].median())
            }
        
        return stats
    
    def generate_feature_report(self, X: pd.DataFrame, y: pd.Series, output_path: Union[str, Path] = "outputs/feature_analysis_report.md"):
        """
        Generate comprehensive feature analysis report.
        
        Args:
            X: Feature matrix
            y: Target variable
            output_path: Path to save the report
        """
        # Get feature importance
        importance = self.extractor.get_feature_importance(X, y)
        
        # Get feature statistics
        stats = self.analyze_feature_distribution(X)
        
        # Generate report
        report = f"""# Crisis Feature Analysis Report

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Feature Overview

Total features: {len(X.columns)}
Total samples: {len(X)}

## Top 20 Most Important Features

"""
        
        for i, (feature, score) in enumerate(list(importance.items())[:20]):
            report += f"{i+1}. **{feature}**: {score:.4f}\n"
        
        report += f"""

## Feature Statistics

### High-Impact Features (Correlation > 0.1)

"""
        
        high_impact = {k: v for k, v in importance.items() if v > 0.1}
        for feature, score in high_impact.items():
            stat = stats.get(feature, {})
            report += f"- **{feature}**: {score:.4f} (mean: {stat.get('mean', 0):.4f}, std: {stat.get('std', 0):.4f})\n"
        
        report += f"""

## Crisis Category Analysis

### Emergency Keywords
- Urgency features: {sum(1 for f in importance.keys() if 'urgency' in f)} features
- Assistance features: {sum(1 for f in importance.keys() if 'assistance' in f)} features
- Casualty features: {sum(1 for f in importance.keys() if 'casualty' in f)} features

### Infrastructure Impact
- Power-related: {sum(1 for f in importance.keys() if 'power' in f)} features
- Water-related: {sum(1 for f in importance.keys() if 'water' in f)} features
- Transportation: {sum(1 for f in importance.keys() if 'transportation' in f)} features

### Social Media Engagement
- Hashtag features: {sum(1 for f in importance.keys() if 'hashtag' in f)} features
- Mention features: {sum(1 for f in importance.keys() if 'mention' in f)} features

## Recommendations

1. **Focus on high-correlation features** for model training
2. **Monitor feature drift** in production
3. **Regular lexicon updates** based on new crisis patterns
4. **Feature engineering pipeline** should be versioned

---
*Report generated by AICrisisAlert Crisis Feature Engineering Framework*
"""
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(str(output_path), 'w') as f:
            f.write(report)
        
        logger.info(f"Feature analysis report saved to {output_path}")
        return report


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = CrisisFeatureExtractor()
    
    # Example text
    sample_text = "URGENT: Multiple people trapped in building collapse on Main Street! Need immediate rescue assistance. #emergency #rescue"
    
    # Extract features
    features = extractor.extract_features(sample_text)
    
    print("Extracted features:")
    for feature, value in features.items():
        if value > 0:
            print(f"  {feature}: {value}") 