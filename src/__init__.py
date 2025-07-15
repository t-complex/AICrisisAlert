"""
AICrisisAlert - AI-Powered Crisis Management and Alert System

This package provides a comprehensive crisis classification system designed for
real-time emergency response coordination. It processes social media posts and
other text inputs to classify them into actionable crisis categories.

Key Components:
- api/          : FastAPI application with crisis classification endpoints
- models/       : ML model components (BERTweet, ensemble, hybrid classifiers)
- training/     : Training scripts and hyperparameter optimization
- utils/        : Utility functions for feature engineering and evaluation
- data_processing/ : Data preprocessing and pipeline components

Main Features:
- Real-time crisis text classification
- Automated hyperparameter optimization with Optuna
- Advanced feature engineering for crisis-specific patterns
- Ensemble model training and evaluation
- Comprehensive API with emergency endpoints
- Docker containerization for easy deployment

Usage:
    from src.api.main import app
    from src.models.hybrid_classifier import HybridCrisisClassifierWrapper
    
    # Start API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Use classifier directly
    classifier = HybridCrisisClassifierWrapper()
    result = classifier.classify("URGENT: People trapped in building collapse!")

Author: [Your Name]
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"
__description__ = "AI-powered crisis management and alert system"