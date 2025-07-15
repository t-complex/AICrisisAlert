# 🏗️ AICrisisAlert Project Structure

This document outlines the professional organization of the AICrisisAlert project.

## 📁 Root Directory Structure

```
AICrisisAlert/
├── 📁 src/                    # Source code
│   ├── 📁 api/               # FastAPI application
│   ├── 📁 models/            # ML model components
│   ├── 📁 training/          # Training scripts and utilities
│   ├── 📁 utils/             # Utility functions
│   ├── 📁 data_processing/   # Data processing modules
│   └── 📁 inference/         # Inference components
├── 📁 scripts/               # Executable scripts
│   ├── 📁 training/          # Model training scripts
│   ├── 📁 data/              # Data processing scripts
│   ├── 📁 deployment/        # Deployment scripts
│   ├── start_api.py          # API startup script
│   └── test_api.py           # API testing script
├── 📁 tests/                 # Test suite
│   ├── 📁 unit/              # Unit tests
│   ├── 📁 integration/       # Integration tests
│   ├── 📁 api/               # API tests
│   └── conftest.py           # Pytest configuration
├── 📁 configs/               # Configuration files
├── 📁 data/                  # Data files
│   ├── 📁 raw/               # Raw datasets
│   └── 📁 processed/         # Processed datasets
├── 📁 outputs/               # Output files
│   └── 📁 models/            # Trained models
├── 📁 docs/                  # Documentation
├── 📁 notebooks/             # Jupyter notebooks
├── 📁 logs/                  # Log files
├── 📁 .github/               # GitHub configuration
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker services
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview
├── DEPLOYMENT.md            # Deployment guide
├── PROJECT_STATUS.md        # Current status
└── env.example              # Environment template
```

## 🔧 Source Code (`src/`)

### API Layer (`src/api/`)
- **`main_simple.py`**: Simplified API with mock model
- **`main.py`**: Full API with trained model integration
- **`models.py`**: Pydantic request/response models
- **`services.py`**: Business logic layer
- **`config.py`**: Configuration management
- **`dependencies.py`**: Dependency injection
- **`middleware.py`**: Request logging and monitoring

### Models (`src/models/`)
- **`model_loader.py`**: Model loading utilities
- **`config.py`**: Model configuration
- **`lora_setup.py`**: LoRA fine-tuning setup
- **`ensemble_classifier.py`**: Ensemble methods
- **`model_persistence.py`**: Model saving/loading

### Training (`src/training/`)
- **`enhanced_train.py`**: Main training pipeline
- **`train.py`**: Basic training script
- **`dataset_utils.py`**: Dataset utilities
- **`trainer_utils.py`**: Training utilities
- **`ensemble_trainer.py`**: Ensemble training

### Utils (`src/utils/`)
- **`evaluation.py`**: Evaluation metrics
- **`enhanced_evaluation.py`**: Advanced evaluation
- **`crisis_features.py`**: Feature engineering

## 📜 Scripts (`scripts/`)

### Training Scripts (`scripts/training/`)
- **`enhanced_feature_engineering.py`**: Main training script for Windows GPU
- **`optimize_hyperparameters.py`**: Hyperparameter optimization
- **`optimize_hyperparameters_minimal.py`**: Minimal optimization
- **`test_enhanced_setup.py`**: Training setup testing
- **`test_enhanced_setup_comprehensive.py`**: Comprehensive testing
- **`test_ensemble_performance.py`**: Ensemble performance testing
- **`dry_run_enhanced_train.py`**: Training dry run
- **`run_enhanced_training.sh`**: Training shell script
- **`run_training.sh`**: Basic training script

### Data Scripts (`scripts/data/`)
- **`create_balanced_leak_free_dataset.py`**: Dataset creation
- **`create_dry_run_dataset.py`**: Dry run dataset
- **`clean_and_merge.py`**: Data cleaning and merging

### Deployment Scripts (`scripts/deployment/`)
- **`init-db.sql`**: Database initialization

### Root Scripts
- **`start_api.py`**: API server startup
- **`test_api.py`**: API testing

## 🧪 Tests (`tests/`)

### Unit Tests (`tests/unit/`)
- **`test_hyperopt.py`**: Hyperparameter optimization tests
- **`test_hyperopt_simple.py`**: Simple optimization tests
- **`test_imports.py`**: Import validation tests

### API Tests (`tests/api/`)
- **`test_api.py`**: Complete API endpoint testing

### Integration Tests (`tests/integration/`)
- End-to-end testing scenarios

### Test Configuration
- **`conftest.py`**: Pytest fixtures and configuration
- **`test_output/`**: Test output files

## ⚙️ Configuration (`configs/`)

- **`enhanced_training_config.json`**: Main training configuration
- **`dry_run_config.json`**: Dry run configuration
- **`best_hyperparams.json`**: Optimized hyperparameters
- **`best_hyperparams_minimal.json`**: Minimal hyperparameters

## 📊 Data (`data/`)

### Raw Data (`data/raw/`)
- **`Crisis_Benchmarks_Dataset/`**: Original crisis datasets
- **`humaid_data_all/`**: Humanitarian aid data
- **`Kaggle/`**: Kaggle datasets

### Processed Data (`data/processed/`)
- **`train_balanced_leak_free.csv`**: Training dataset
- **`validation_balanced_leak_free.csv`**: Validation dataset
- **`test_balanced_leak_free.csv`**: Test dataset
- **`merged_dataset.csv`**: Combined dataset

## 📤 Outputs (`outputs/`)

### Models (`outputs/models/`)
- **`bertweet_enhanced/`**: Trained BERTweet model
- **`dry_run_test/`**: Test model outputs
- **`test_bertweet_enhanced/`**: Test model results

## 🐳 Docker Configuration

- **`Dockerfile`**: Multi-stage Docker build
- **`docker-compose.yml`**: Full stack services
- **`env.example`**: Environment variables template

## 📚 Documentation

- **`README.md`**: Project overview and quick start
- **`DEPLOYMENT.md`**: Comprehensive deployment guide
- **`PROJECT_STATUS.md`**: Current project status
- **`PROJECT_STRUCTURE.md`**: This file

## 🚀 Key Files for Development

### Quick Start
```bash
# Start API
python scripts/start_api.py

# Test API
python scripts/test_api.py

# Run tests
pytest tests/
```

### Training (Windows GPU)
```bash
# Main training script
python scripts/training/enhanced_feature_engineering.py

# Hyperparameter optimization
python scripts/training/optimize_hyperparameters.py
```

### Docker Deployment
```bash
# Development
docker-compose --profile dev up -d

# Production
docker-compose --profile production up -d
```

## 🎯 Benefits of This Structure

### ✅ Professional Organization
- **Clear separation** of concerns
- **Logical grouping** of related files
- **Standard conventions** followed
- **Easy navigation** for new developers

### ✅ Scalability
- **Modular design** allows easy expansion
- **Clear boundaries** between components
- **Reusable components** across modules
- **Testable architecture** at all levels

### ✅ Maintainability
- **Consistent naming** conventions
- **Documented structure** for easy understanding
- **Organized scripts** by purpose
- **Proper test coverage** structure

### ✅ Deployment Ready
- **Docker configuration** for all environments
- **Environment management** with templates
- **CI/CD pipeline** integration
- **Production deployment** guides

This structure follows industry best practices and makes the project professional, maintainable, and ready for production deployment. 