#!/usr/bin/env python3
"""
Enhanced Feature Engineering for Crisis Classification
Target: Push performance from 84% to 87%+ using advanced feature engineering
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import json
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import re

# Suppress warnings
warnings.filterwarnings("ignore")

def extract_advanced_features(data):
    """Extract advanced features for crisis classification."""
    print("Extracting advanced features...")
    
    # 1. Basic text features
    data['text_length'] = data['text'].str.len()
    data['word_count'] = data['text'].str.split().str.len()
    data['avg_word_length'] = data['text'].str.split().apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
    data['hashtag_count'] = data['text'].str.count('#')
    data['mention_count'] = data['text'].str.count('@')
    data['url_count'] = data['text'].str.count('http')
    data['exclamation_count'] = data['text'].str.count('!')
    data['question_count'] = data['text'].str.count(r'\?')
    data['uppercase_ratio'] = data['text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
    data['digit_count'] = data['text'].str.count(r'\d')
    data['punctuation_count'] = data['text'].str.count(r'[^\w\s]')
    
    # 2. Crisis-specific keyword features (expanded)
    crisis_keywords = {
        'emergency': ['emergency', 'urgent', 'immediate', 'critical', 'desperate'],
        'medical': ['hospital', 'doctor', 'medical', 'ambulance', 'injured', 'wounded', 'bleeding'],
        'disaster': ['disaster', 'catastrophe', 'tragedy', 'devastation', 'destruction'],
        'rescue': ['rescue', 'help', 'save', 'evacuate', 'survivor'],
        'damage': ['damage', 'destroyed', 'collapsed', 'broken', 'flooded'],
        'victims': ['victim', 'casualty', 'dead', 'missing', 'trapped'],
        'volunteer': ['volunteer', 'donation', 'support', 'aid', 'relief'],
        'location': ['street', 'building', 'area', 'zone', 'district'],
        'time_urgency': ['now', 'immediately', 'asap', 'quickly', 'fast'],
        'weather': ['storm', 'flood', 'fire', 'earthquake', 'hurricane', 'tornado']
    }
    
    for category, keywords in crisis_keywords.items():
        for keyword in keywords:
            data[f'has_{keyword}'] = data['text'].str.contains(keyword, case=False).astype(int)
    
    # 3. Sentiment and emotion indicators
    positive_words = ['help', 'support', 'safe', 'rescued', 'saved', 'recovered', 'better']
    negative_words = ['dead', 'injured', 'destroyed', 'lost', 'missing', 'trapped', 'dangerous']
    urgent_words = ['urgent', 'immediate', 'now', 'asap', 'emergency', 'critical']
    
    for word in positive_words:
        data[f'positive_{word}'] = data['text'].str.contains(word, case=False).astype(int)
    for word in negative_words:
        data[f'negative_{word}'] = data['text'].str.contains(word, case=False).astype(int)
    for word in urgent_words:
        data[f'urgent_{word}'] = data['text'].str.contains(word, case=False).astype(int)
    
    # 4. Text complexity features
    data['sentence_count'] = data['text'].str.count(r'[.!?]+')
    data['avg_sentence_length'] = data['word_count'] / (data['sentence_count'] + 1)
    data['unique_word_ratio'] = data['text'].str.split().apply(lambda x: len(set(x)) / len(x) if x else 0)
    
    # 5. Social media specific features
    data['retweet_indicator'] = data['text'].str.startswith('RT').astype(int)
    data['reply_indicator'] = data['text'].str.startswith('@').astype(int)
    data['has_media'] = data['text'].str.contains(r'pic\.|photo|image|video', case=False).astype(int)
    
    # 6. Time-based urgency indicators
    time_patterns = [
        r'\bnow\b', r'\basap\b', r'\bimmediately\b', r'\bquickly\b', r'\bsoon\b',
        r'\btoday\b', r'\btonight\b', r'\bwithin.*hour', r'\bwithin.*minute'
    ]
    for i, pattern in enumerate(time_patterns):
        data[f'time_urgency_{i}'] = data['text'].str.contains(pattern, case=False).astype(int)
    
    # 7. Geographic indicators
    geo_patterns = [
        r'\bstreet\b', r'\bavenue\b', r'\broad\b', r'\bbuilding\b', r'\barea\b',
        r'\bzone\b', r'\bdistrict\b', r'\bneighborhood\b', r'\bcity\b', r'\btown\b'
    ]
    for i, pattern in enumerate(geo_patterns):
        data[f'geo_{i}'] = data['text'].str.contains(pattern, case=False).astype(int)
    
    print(f"Extracted {len(data.columns) - 2} features (excluding text and label)")
    return data

def create_advanced_vectorizers(data):
    """Create multiple vectorizers for different aspects of text."""
    print("Creating advanced vectorizers...")
    
    # 1. TF-IDF with different configurations
    tfidf1 = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    tfidf2 = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(2, 3))
    tfidf3 = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 1))
    
    # 2. Count vectorizer for crisis-specific terms
    crisis_terms = [
        'emergency', 'urgent', 'help', 'need', 'disaster', 'crisis', 'victim',
        'injured', 'dead', 'hospital', 'rescue', 'volunteer', 'donation',
        'damage', 'destroyed', 'flood', 'fire', 'earthquake', 'hurricane',
        'missing', 'trapped', 'evacuate', 'survivor', 'casualty', 'wounded'
    ]
    crisis_vectorizer = CountVectorizer(vocabulary=crisis_terms, binary=True)
    
    # Fit and transform
    tfidf1_features = tfidf1.fit_transform(data['text'])
    tfidf2_features = tfidf2.fit_transform(data['text'])
    tfidf3_features = tfidf3.fit_transform(data['text'])
    crisis_features = crisis_vectorizer.fit_transform(data['text'])
    
    return tfidf1_features, tfidf2_features, tfidf3_features, crisis_features

def train_advanced_ensemble(X, y):
    """Train an advanced ensemble model."""
    print("Training advanced ensemble model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Multiple diverse classifiers
    classifiers = [
        ('rf1', RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42)),
        ('rf2', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=43)),
        ('gb1', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=44)),
        ('gb2', GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=45)),
        ('lr1', LogisticRegression(C=1.0, max_iter=1000, random_state=46)),
        ('lr2', LogisticRegression(C=0.1, max_iter=1000, random_state=47)),
        ('svm1', SVC(kernel='rbf', C=1.0, probability=True, random_state=48)),
        ('svm2', SVC(kernel='linear', C=0.1, probability=True, random_state=49))
    ]
    
    # Train individual models
    individual_scores = {}
    for name, clf in classifiers:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        score = f1_score(y_test, y_pred, average='macro')
        individual_scores[name] = score
        print(f"{name}: Macro F1 = {score:.4f}")
    
    # Create weighted ensemble (weight by performance)
    weights = [individual_scores[name] for name, _ in classifiers]
    weights = np.array(weights) / sum(weights)  # Normalize
    
    ensemble = VotingClassifier(
        estimators=classifiers,
        voting='soft',
        weights=weights
    )
    
    # Train ensemble
    ensemble.fit(X_train_scaled, y_train)
    y_pred_ensemble = ensemble.predict(X_test_scaled)
    
    # Evaluate
    ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='macro')
    ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
    
    print(f"\nAdvanced Ensemble Results:")
    print(f"Macro F1: {ensemble_f1:.4f}")
    print(f"Accuracy: {ensemble_acc:.4f}")
    print(classification_report(y_test, y_pred_ensemble))
    
    return ensemble, scaler, ensemble_f1, ensemble_acc

def main():
    """Run enhanced feature engineering pipeline."""
    print("🚀 Enhanced Feature Engineering Pipeline")
    print("Target: Push performance beyond 84% accuracy")
    print("=" * 60)
    
    # Load data
    print("\n1️⃣ Loading data...")
    try:
        data = pd.read_csv('data/processed/train_balanced_leak_free.csv')
        print(f"Loaded {len(data)} samples")
    except FileNotFoundError:
        print("❌ Data file not found")
        return
    
    # Extract advanced features
    print("\n2️⃣ Extracting advanced features...")
    data = extract_advanced_features(data)
    
    # Create vectorizers
    print("\n3️⃣ Creating advanced vectorizers...")
    tfidf1_features, tfidf2_features, tfidf3_features, crisis_features = create_advanced_vectorizers(data)
    
    # Combine all features
    print("\n4️⃣ Combining all features...")
    feature_columns = [col for col in data.columns if col not in ['text', 'label']]
    basic_features = data[feature_columns].values
    
    # Stack all features
    X_combined = np.column_stack([
        tfidf1_features.toarray(),
        tfidf2_features.toarray(),
        tfidf3_features.toarray(),
        crisis_features.toarray(),
        basic_features
    ])
    
    print(f"Final feature matrix shape: {X_combined.shape}")
    
    # Train advanced ensemble
    print("\n5️⃣ Training advanced ensemble...")
    ensemble, scaler, f1_score, accuracy = train_advanced_ensemble(X_combined, data['label'])
    
    # Save results
    print("\n6️⃣ Saving results...")
    results = {
        'feature_engineering_accuracy': accuracy,
        'feature_engineering_macro_f1': f1_score,
        'feature_matrix_shape': X_combined.shape,
        'target_achieved': accuracy >= 0.87
    }
    
    with open('outputs/enhanced_feature_engineering_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Enhanced Feature Engineering Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Macro F1: {f1_score:.4f}")
    print(f"   Target 87%: {'✅ ACHIEVED' if accuracy >= 0.87 else '❌ Not yet'}")
    
    if accuracy >= 0.87:
        print("\n🎉 CONGRATULATIONS! Enhanced feature engineering achieved 87%+ accuracy!")
    else:
        print(f"\n📈 Current performance: {accuracy:.1%}")
        print("Next: Combine with your existing BERTweet model for ensemble")

if __name__ == '__main__':
    main() 