# AICrisisAlert 🚨

An AI-powered crisis management and alert system designed for real-time emergency response coordination. Built with transformers, FastAPI, and optimized for Florida hurricane response scenarios.

## 🎯 Project Overview

AICrisisAlert classifies social media posts during emergencies into actionable categories, enabling faster response coordination and resource allocation. The system processes tweets, Facebook posts, and other social media content to identify:

- **Urgent Requests for Help** - People needing immediate assistance
- **Infrastructure Damage Reports** - Critical infrastructure status updates  
- **Casualty Information** - Injured or missing persons reports
- **Resource Availability** - Volunteer coordination and donation offers
- **General Crisis Information** - Situational awareness updates

## 🏗️ Architecture

### Core Components

- **API Layer**: FastAPI with real-time crisis classification endpoints
- **ML Models**: Fine-tuned BERTweet for crisis classification with feature engineering
- **Data Pipeline**: Real-time social media ingestion and preprocessing
- **Database**: PostgreSQL for storing classifications and analytics
- **Caching**: Redis for performance optimization
- **Containerization**: Docker for easy deployment

### Technology Stack

- **ML/AI**: Transformers, PyTorch, Hugging Face, scikit-learn, Optuna
- **Backend**: FastAPI, PostgreSQL, Redis, Celery
- **Infrastructure**: Docker, Docker Compose, AWS (EC2, RDS, S3)
- **Monitoring**: Prometheus, Grafana, structured logging

### Project Structure

```
AICrisisAlert/
├── 📁 src/                    # Source code
│   ├── 📁 api/               # FastAPI application
│   ├── 📁 models/            # ML model components
│   ├── 📁 training/          # Training scripts and utilities
│   ├── 📁 utils/             # Utility functions
│   └── 📁 data_processing/   # Data processing modules
├── 📁 scripts/               # Executable scripts
│   ├── 📁 training/          # Model training scripts
│   ├── 📁 data/              # Data processing scripts
│   ├── 📁 deployment/        # Deployment scripts
│   ├── start_api.py          # API startup script
│   └── test_api.py           # API testing script
├── 📁 tests/                 # Test suite
│   ├── 📁 unit/              # Unit tests
│   ├── 📁 integration/       # Integration tests
│   └── 📁 api/               # API tests
├── 📁 configs/               # Configuration files
├── 📁 data/                  # Data files
├── 📁 outputs/               # Output files
└── 📁 docs/                  # Documentation
```

📋 **See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure documentation.**

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (optional)
- Git

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/AICrisisAlert.git
cd AICrisisAlert

# Setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start the API (Simple Mode - No Model Required)
python scripts/start_api.py

# Test the API
python scripts/test_api.py
```

### API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Testing

Test the API endpoints:

```bash
# Quick status check
python scripts/check_api_status.py

# Comprehensive testing
python scripts/test_api.py

# Test with curl
curl http://localhost:8000/health
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT: People trapped in building collapse!"}'
```

### API Modes

The API runs in two modes:

1. **Simple Mode (Default)**: Uses mock model for fast testing
   ```bash
   USE_SIMPLE_API=true python scripts/start_api.py
   ```

2. **Full Mode**: Uses trained BERTweet model (requires model files)
   ```bash
   USE_SIMPLE_API=false python scripts/start_api.py
   ```

**Note**: If you experience timeout issues, the API is likely running in full mode and loading the ML model. Use simple mode for testing and development.

### Docker Deployment

```bash
# Start all services with Docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

## Hyperparameter Optimization (Optuna)

To run automated hyperparameter optimization for the BERTweet crisis classification model:

1. Ensure requirements are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the optimization script:
   ```bash
   python scripts/optimize_hyperparameters.py
   ```

- This will run 50 Optuna trials, searching for the best config to maximize macro F1 and humanitarian F1.
- The best hyperparameters will be saved to `configs/best_hyperparams.json`.
- A summary report will be saved to `outputs/optimization_report.md`.
- Progress and parameter importance plots will be shown at the end.

You can then use the optimized config for training:
```bash
python src/training/enhanced_train.py --config configs/best_hyperparams.json
```

## Crisis-Specific Feature Engineering

The AICrisisAlert project now includes comprehensive crisis-specific feature engineering to boost classification performance beyond 87%. This pipeline extracts domain-specific features that emergency responders would recognize.

### Features Extracted

1. **Emergency Keywords**: Frequency analysis of crisis-related terms (urgent, help, trapped, emergency, etc.)
2. **Geolocation Indicators**: Street names, landmarks, coordinates detection
3. **Time Urgency Markers**: Now, immediate, ASAP, hurry indicators
4. **Casualty Indicators**: Injured, dead, missing, hurt detection
5. **Infrastructure Keywords**: Power, water, road, bridge, hospital mentions
6. **Social Media Engagement**: Retweets, mentions, hashtags analysis
7. **Crisis Severity Scoring**: Domain-specific urgency and impact assessment

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Extract features from your data**:
   ```python
   from src.utils.crisis_features import CrisisFeatureExtractor
   
   # Initialize extractor
   extractor = CrisisFeatureExtractor()
   
   # Extract features from texts
   texts = ["URGENT: People trapped in building collapse!", "Hurricane causing power outages"]
   features = extractor.extract_batch_features(texts)
   ```

3. **Use hybrid classifier with features**:
   ```python
   from src.models.hybrid_classifier import HybridClassifierConfig, HybridCrisisClassifierWrapper
   
   # Configure hybrid model
   config = HybridClassifierConfig(
       use_engineered_features=True,
       feature_dim=100,
       use_attention=True,
       crisis_weighting=True
   )
   
   # Create wrapper with feature extractor
   model = HybridCrisisClassifierWrapper(config, extractor)
   
   # Predict with features
   predictions = model.predict(texts)
   ```

### Complete Pipeline Integration

#### 1. Hyperparameter Optimization with Features
```bash
# Run Optuna optimization with engineered features
python scripts/training/optimize_hyperparameters.py
```

#### 2. Feature Engineering Analysis
```python
from src.utils.crisis_features import CrisisFeatureAnalyzer

# Analyze feature importance
analyzer = CrisisFeatureAnalyzer(extractor)
report = analyzer.generate_feature_report(X, y)
```

#### 3. Ensemble Training with Enhanced Features
```bash
# Test ensemble performance with feature-enhanced models
python scripts/training/test_ensemble_performance.py --comprehensive --verbose
```

### Key Components

#### Crisis Lexicon (`data/crisis_lexicon.json`)
- 500+ emergency-related terms with severity scoring
- Category mapping to target classes
- Synonym expansion for robust matching
- Crisis-specific terminology preservation

#### Feature Extractors (`src/utils/crisis_features.py`)
- **CrisisFeatureExtractor**: Main feature extraction engine
- **CrisisFeatureAnalyzer**: Feature importance and analysis
- Batch processing for large datasets
- Crisis-specific feature weighting

#### Enhanced Preprocessing (`src/utils/crisis_preprocessing.py`)
- **CrisisPreprocessor**: NER for locations/organizations
- **CrisisFeatureScaler**: Feature normalization
- **CrisisFeatureSelector**: Feature selection
- Sentiment analysis for urgency detection

#### Hybrid Classifier (`src/models/hybrid_classifier.py`)
- **HybridCrisisClassifier**: BERTweet + engineered features
- **FeatureAttentionLayer**: Attention mechanism for feature importance
- **CrisisFeatureWeighting**: Domain-specific weighting
- **HybridCrisisClassifierWrapper**: Easy-to-use interface

### Performance Impact

**Before Feature Engineering:**
- Accuracy: 84.2%
- Macro F1: 0.823
- Humanitarian F1: 0.789

**After Feature Engineering (Projected):**
- Accuracy: 87.5% (+3.3%)
- Macro F1: 0.861 (+0.038)
- Humanitarian F1: 0.834 (+0.045)

### Feature Categories

1. **Emergency Keywords** (15 features)
   - Urgency, assistance, casualty indicators
   - Severity-weighted frequency analysis

2. **Geolocation Indicators** (18 features)
   - Street names, landmarks, coordinates
   - Location confidence scoring

3. **Infrastructure Impact** (24 features)
   - Power, water, transportation, building damage
   - Infrastructure-specific severity assessment

4. **Social Media Engagement** (9 features)
   - Hashtags, mentions, viral indicators
   - Amplification and spread metrics

5. **Crisis-Specific Features** (6 features)
   - Urgency score, casualty estimate
   - Infrastructure damage, response resources

### Integration with Training Pipeline

The feature engineering pipeline integrates seamlessly with existing training:

1. **Enhanced Training Config**:
   ```python
   config = EnhancedTrainingConfig(
       use_engineered_features=True,
       feature_extraction_method="crisis_specific"
   )
   ```

2. **Hybrid Model Training**:
   ```python
   trainer = HybridCrisisTrainer(config, feature_extractor)
   trainer.train()
   ```

3. **Ensemble Integration**:
   ```python
   ensemble = create_crisis_ensemble(config)
   results = train_crisis_ensemble(config, training_config, train_loader, val_loader)
   ```

### Advanced Usage

#### Custom Feature Extraction
```python
# Add custom crisis patterns
custom_patterns = {
    "custom_crisis": {
        "terms": ["custom_term1", "custom_term2"],
        "severity": 4,
        "category": "custom"
    }
}

extractor.add_custom_patterns(custom_patterns)
```

#### Feature Importance Analysis
```python
# Get top features
top_features = extractor.select_top_features(X, y, top_k=50)

# Analyze feature distribution
stats = analyzer.analyze_feature_distribution(X)
```

#### Crisis-Specific Weighting
```python
# Configure crisis weighting
config = HybridClassifierConfig(
    crisis_weighting=True,
    humanitarian_boost=1.2,
    critical_crisis_weight=2.0
)
```

### Monitoring and Maintenance

1. **Feature Drift Detection**: Monitor feature distribution changes
2. **Lexicon Updates**: Regular updates based on new crisis patterns
3. **Performance Tracking**: Track feature contribution to model performance
4. **Version Control**: Version feature engineering pipeline

### Troubleshooting

**Common Issues:**
- SpaCy model not found: `python -m spacy download en_core_web_sm`
- Memory issues: Use batch processing for large datasets
- Feature dimension mismatch: Ensure consistent feature extraction

**Performance Tips:**
- Use batch processing for feature extraction
- Cache extracted features for repeated use
- Parallelize NLP components where possible

---

**Complete 3-Feature Pipeline:**
1. **Hyperparameter Optimization**: `python scripts/training/optimize_hyperparameters.py`
2. **Feature Engineering**: Use crisis-specific features in training
3. **Ensemble Training**: `python scripts/training/test_ensemble_performance.py --comprehensive --verbose`

This comprehensive feature engineering pipeline is designed to achieve 87%+ accuracy through domain knowledge integration while maintaining interpretability for emergency responders.
