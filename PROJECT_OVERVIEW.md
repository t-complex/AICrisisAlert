# AICrisisAlert - Quick Project Overview

## 🎯 What This Project Does

AICrisisAlert is an **AI-powered crisis management system** that classifies social media posts and text messages during emergencies into actionable categories for emergency responders.

**Input**: Text like "URGENT: People trapped in building collapse!"
**Output**: Classification like "urgent" with confidence score

## 🏗️ Project Architecture (Quick Reference)

```
AICrisisAlert/
├── 📁 src/                    # Main source code
│   ├── api/                  # FastAPI application (main.py)
│   ├── models/               # ML models (BERTweet, ensemble)
│   ├── training/             # Training scripts & hyperparameter optimization
│   ├── utils/                # Feature engineering & evaluation
│   └── data_processing/      # Data preprocessing
├── 📁 scripts/               # Executable scripts
│   ├── run_all_tests.py      # Test runner
│   ├── start_api.py          # API startup
│   └── test_api.py           # API testing
├── 📁 tests/                 # Test suite (unit, integration, api)
├── 📁 configs/               # Configuration files
├── 📁 data/                  # Training data and datasets
└── 📁 docs/                  # Documentation
```

## 🚀 Key Components

### 1. **API Layer** (`src/api/`)
- **FastAPI application** with crisis classification endpoints
- **Endpoints**: `/classify`, `/classify/batch`, `/classify/emergency`
- **Health check**: `/health`
- **Model info**: `/model/info`

### 2. **ML Models** (`src/models/`)
- **BERTweet**: Fine-tuned transformer for crisis classification
- **Hybrid Classifier**: Combines BERTweet with engineered features
- **Ensemble Models**: Multiple classifiers for better performance
- **Feature Engineering**: Crisis-specific pattern extraction

### 3. **Training Pipeline** (`src/training/`)
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Enhanced Training**: Advanced training with feature engineering
- **Ensemble Training**: Multi-model training and evaluation

### 4. **Feature Engineering** (`src/utils/`)
- **Crisis Features**: Emergency keywords, geolocation, urgency markers
- **Text Processing**: NER, sentiment analysis, preprocessing
- **Evaluation**: Comprehensive model evaluation metrics

## 🔧 Technology Stack

- **ML/AI**: PyTorch, Transformers (BERTweet), scikit-learn, Optuna
- **API**: FastAPI, uvicorn
- **Data**: PostgreSQL, Redis
- **Deployment**: Docker, Docker Compose
- **Testing**: pytest, flake8, autoflake

## 📊 Current Performance

- **Accuracy**: ~84% (BERTweet model)
- **Macro F1**: ~78%
- **Target**: 87%+ (with feature engineering)

## 🎯 Use Cases

1. **Emergency Response**: Real-time crisis classification during disasters
2. **Social Media Monitoring**: Automated analysis of emergency posts
3. **Resource Allocation**: Help emergency responders prioritize responses
4. **Situational Awareness**: Provide overview of crisis situations

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start API (simple mode)
python scripts/start_api.py

# Test API
python scripts/test_api.py

# Run all tests
python scripts/run_all_tests.py
```

## 📚 Key Files to Understand

1. **`src/api/main.py`** - Main API application
2. **`src/models/hybrid_classifier.py`** - Core ML model
3. **`src/training/enhanced_train.py`** - Training pipeline
4. **`src/utils/crisis_features.py`** - Feature engineering
5. **`scripts/run_all_tests.py`** - Test runner
6. **`README.md`** - Comprehensive documentation

## 🔍 For Claude Analysis

When analyzing this project, focus on:
1. **API endpoints** and their functionality
2. **ML model architecture** and training process
3. **Feature engineering** pipeline
4. **Testing strategy** and quality assurance
5. **Deployment** and scalability considerations

This project demonstrates **production-ready AI/ML development** with proper testing, documentation, and deployment practices. 