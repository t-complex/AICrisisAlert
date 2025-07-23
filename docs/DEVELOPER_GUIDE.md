# AICrisisAlert Developer Guide

## Overview

This guide provides comprehensive information for developers working on the AICrisisAlert system. It covers development setup, coding standards, testing procedures, and contribution guidelines.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [API Development](#api-development)
- [Model Development](#model-development)
- [Database Management](#database-management)
- [Contributing Guidelines](#contributing-guidelines)
- [Troubleshooting](#troubleshooting)

## Development Environment Setup

### Prerequisites

**System Requirements**:
- Python 3.9+
- Node.js 16+ (for frontend development)
- Docker 20.10+ & Docker Compose 2.0+
- Git 2.30+
- PostgreSQL 14+ (optional, can use Docker)
- Redis 7+ (optional, can use Docker)

**Recommended IDE Setup**:
- **VS Code** with extensions:
  - Python (Microsoft)
  - Docker (Microsoft)
  - REST Client
  - GitLens
  - Pylance
  - Black Formatter

### Initial Setup

1. **Clone the Repository**:
```bash
git clone https://github.com/your-org/AICrisisAlert.git
cd AICrisisAlert
```

2. **Create Python Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**:
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

4. **Environment Configuration**:
```bash
# Copy environment template
cp env.example .env

# Edit .env with your configuration
SECRET_KEY=your-development-secret-key-32-chars
API_KEY=your-development-api-key-32-chars
DATABASE_URL=postgresql://postgres:password@localhost:5432/aicrisisalert_dev
REDIS_URL=redis://localhost:6379
USE_SIMPLE_API=true  # For development
LOG_LEVEL=DEBUG
```

5. **Database Setup**:
```bash
# Option 1: Using Docker (Recommended)
docker-compose up -d postgres redis

# Option 2: Local Installation
# Install PostgreSQL and Redis locally
# Create database: createdb aicrisisalert_dev

# Run database migrations
alembic upgrade head
```

6. **Verify Installation**:
```bash
# Run basic tests
python -m pytest tests/unit/test_imports.py

# Start API in simple mode
python scripts/start_api.py

# Test API (in another terminal)
curl http://localhost:8000/health
```

### Development Tools Setup

**Pre-commit Hooks**:
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

**IDE Configuration** (VS Code):
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

## Project Structure

### Source Code Organization

```
src/
├── api/                    # FastAPI application
│   ├── __init__.py
│   ├── main.py            # Main FastAPI app
│   ├── config.py          # Configuration management
│   ├── dependencies.py    # Dependency injection
│   ├── models.py          # Pydantic models
│   ├── services.py        # Business logic services
│   ├── database.py        # Database models and connection
│   └── middleware.py      # Custom middleware
├── models/                 # ML model components
│   ├── __init__.py
│   ├── model_loader.py    # Model loading utilities
│   ├── ensemble_classifier.py  # Ensemble methods
│   ├── hybrid_classifier.py    # Hybrid text+features
│   ├── lora_setup.py      # LoRA fine-tuning
│   └── config.py          # Model configuration
├── training/               # Training pipeline
│   ├── __init__.py
│   ├── enhanced_train.py  # Main training script
│   ├── configs.py         # Training configurations
│   ├── dataset_utils.py   # Dataset utilities
│   ├── trainer_utils.py   # Training utilities
│   ├── losses.py          # Custom loss functions
│   └── metrics.py         # Evaluation metrics
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── config.py          # Global configuration
│   ├── crisis_features.py # Feature engineering
│   ├── evaluation.py      # Model evaluation
│   └── preprocessing.py   # Data preprocessing
└── inference/              # Inference pipeline
    ├── __init__.py
    └── predictor.py       # Crisis prediction logic
```

### Key Components

**API Layer** (`src/api/`):
- **main.py**: FastAPI application with endpoints
- **services.py**: Business logic and model integration
- **models.py**: Request/response schemas
- **dependencies.py**: Authentication and rate limiting

**ML Pipeline** (`src/models/`, `src/training/`):
- **model_loader.py**: Model loading and management
- **enhanced_train.py**: Training orchestration
- **crisis_features.py**: Domain-specific feature engineering
- **predictor.py**: Inference pipeline

**Configuration Management**:
- Environment-based configuration
- Type-safe settings with Pydantic
- Development/staging/production profiles

## Development Workflow

### Git Workflow

**Branch Strategy**:
```bash
# Main branches
main          # Production-ready code
develop       # Integration branch
release/*     # Release preparation
hotfix/*      # Emergency fixes

# Feature branches
feature/add-new-endpoint
feature/improve-model-accuracy
bugfix/fix-memory-leak
```

**Development Process**:
1. Create feature branch from `develop`
2. Implement changes with tests
3. Run pre-commit checks
4. Create pull request to `develop`
5. Code review and testing
6. Merge to `develop`
7. Release preparation and testing
8. Merge to `main` for production

### Code Review Process

**Pull Request Guidelines**:
- **Title**: Clear, descriptive title
- **Description**: Explain changes and rationale
- **Tests**: Include relevant tests
- **Documentation**: Update docs if needed
- **Breaking Changes**: Clearly marked

**Review Checklist**:
- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is adequate
- [ ] Security considerations addressed
- [ ] Performance impact considered
- [ ] Documentation updated
- [ ] API changes are backward compatible

## Coding Standards

### Python Style Guide (PEP 8)

**Code Formatting**:
```python
# Use Black formatter (line length: 88)
# Configure in pyproject.toml
[tool.black]
line-length = 88
target-version = ['py39']

# Import organization (isort)
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
```

**Naming Conventions**:
```python
# Variables and functions: snake_case
user_id = "12345"
def process_classification_request():
    pass

# Classes: PascalCase
class CrisisClassifier:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_TEXT_LENGTH = 10000
DEFAULT_BATCH_SIZE = 32

# Private methods: leading underscore
def _internal_helper_method():
    pass
```

**Documentation Standards**:
```python
def classify_crisis_text(
    text: str,
    model: Any,
    features: Optional[Dict[str, float]] = None
) -> CrisisClassificationResponse:
    """
    Classify crisis text using trained model.
    
    Args:
        text: Input text to classify (3-10,000 characters)
        model: Trained crisis classification model
        features: Optional feature dictionary for hybrid classification
        
    Returns:
        CrisisClassificationResponse with prediction and confidence
        
    Raises:
        ValueError: If text is empty or too long
        ModelNotLoadedError: If model is not properly initialized
        
    Example:
        >>> result = classify_crisis_text("Help needed urgently!")
        >>> print(result.predicted_class)
        'urgent_help'
    """
    if not text or len(text.strip()) < 3:
        raise ValueError("Text must be at least 3 characters")
    
    # Implementation...
    return result
```

### Type Hints

**Required Type Annotations**:
```python
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Function signatures
async def process_batch_classification(
    texts: List[str],
    batch_size: int = 32,
    timeout: Optional[float] = None
) -> List[CrisisClassificationResponse]:
    """Process multiple texts with proper typing."""
    pass

# Class definitions
class ModelConfiguration:
    """Type-safe model configuration."""
    model_name: str
    max_length: int
    learning_rate: float
    use_gpu: bool = True
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
```

### Error Handling

**Exception Hierarchy**:
```python
# Custom exception classes
class AICrisisAlertError(Exception):
    """Base exception for AICrisisAlert."""
    pass

class ModelNotLoadedError(AICrisisAlertError):
    """Raised when model is not properly loaded."""
    pass

class ClassificationError(AICrisisAlertError):
    """Raised when classification fails."""
    pass

class ValidationError(AICrisisAlertError):
    """Raised when input validation fails."""
    pass

# Usage
try:
    result = await classify_text(text)
except ModelNotLoadedError:
    logger.error("Model not loaded, using fallback")
    result = get_fallback_classification(text)
except ClassificationError as e:
    logger.error(f"Classification failed: {e}")
    raise HTTPException(status_code=500, detail="Classification failed")
```

**Logging Standards**:
```python
import structlog

# Configure structured logging
logger = structlog.get_logger(__name__)

# Log levels and usage
logger.debug("Detailed debug information", extra_field="value")
logger.info("General information", user_id=user_id, action="classify")
logger.warning("Warning condition", threshold_exceeded=True)
logger.error("Error occurred", error=str(e), traceback=True)
logger.critical("Critical system error", system="api", component="auth")

# Context logging
with structlog.contextvars.bound_contextvars(request_id="req_123"):
    logger.info("Processing request")  # Includes request_id automatically
```

## Testing Guidelines

### Test Structure

**Test Organization**:
```
tests/
├── unit/                   # Unit tests
│   ├── test_api_models.py
│   ├── test_classification.py
│   ├── test_features.py
│   └── test_utils.py
├── integration/            # Integration tests
│   ├── test_api_endpoints.py
│   ├── test_database.py
│   └── test_model_pipeline.py
├── api/                    # API tests
│   ├── test_api.py
│   └── test_auth.py
├── conftest.py            # Pytest configuration
└── fixtures/              # Test fixtures
    ├── sample_data.json
    └── mock_responses.py
```

### Unit Testing

**Test Writing Standards**:
```python
import pytest
from unittest.mock import Mock, patch
from src.api.services import ModelService
from src.api.models import CrisisClassificationRequest

class TestModelService:
    """Test model service functionality."""
    
    @pytest.fixture
    def model_service(self):
        """Create model service instance for testing."""
        service = ModelService()
        service.is_initialized = True
        return service
    
    @pytest.fixture
    def sample_request(self):
        """Create sample classification request."""
        return CrisisClassificationRequest(
            text="URGENT: People trapped in building collapse!",
            source="twitter",
            location="Miami, FL"
        )
    
    async def test_classify_text_success(self, model_service, sample_request):
        """Test successful text classification."""
        # Arrange
        with patch.object(model_service, '_predict') as mock_predict:
            mock_predict.return_value = ("urgent_help", {"urgent_help": 0.92})
            
            # Act
            result = await model_service.classify_text(sample_request.text)
            
            # Assert
            assert result.predicted_class == "urgent_help"
            assert result.confidence == 0.92
            assert result.processing_time_ms > 0
    
    async def test_classify_text_not_initialized(self, model_service, sample_request):
        """Test classification when model not initialized."""
        # Arrange
        model_service.is_initialized = False
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Model service not initialized"):
            await model_service.classify_text(sample_request.text)
    
    @pytest.mark.parametrize("text,expected_error", [
        ("", "Text cannot be empty"),
        ("ab", "Text must be at least 3 characters"),
        ("a" * 10001, "Text too long"),
    ])
    def test_text_validation(self, text, expected_error):
        """Test text validation with various inputs."""
        with pytest.raises(ValueError, match=expected_error):
            CrisisClassificationRequest(text=text)
```

### Integration Testing

**API Integration Tests**:
```python
import pytest
from httpx import AsyncClient
from src.api.main import app

@pytest.mark.asyncio
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
    
    async def test_classify_endpoint_success(self, api_key_header):
        """Test successful classification."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/classify",
                headers=api_key_header,
                json={
                    "text": "URGENT: People trapped in building collapse!",
                    "source": "twitter"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert data["confidence"] >= 0.0 and data["confidence"] <= 1.0
    
    async def test_classify_endpoint_auth_required(self):
        """Test that authentication is required."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/classify",
                json={"text": "Test text"}
            )
            
            assert response.status_code == 401
```

### Test Configuration

**pytest Configuration** (`conftest.py`):
```python
import pytest
import asyncio
from unittest.mock import Mock
from src.api.dependencies import verify_api_key

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def api_key_header():
    """Provide valid API key header for tests."""
    return {"Authorization": "Bearer test-api-key"}

@pytest.fixture
def mock_model_service():
    """Mock model service for testing."""
    service = Mock()
    service.is_initialized = True
    service.classify_text.return_value = Mock(
        predicted_class="urgent_help",
        confidence=0.85,
        processing_time_ms=150.0
    )
    return service

@pytest.fixture(autouse=True)
def override_dependencies(mock_model_service):
    """Override FastAPI dependencies for testing."""
    from src.api.main import app
    from src.api.dependencies import get_model_service_dependency
    
    app.dependency_overrides[get_model_service_dependency] = lambda: mock_model_service
    app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
    
    yield
    
    # Cleanup
    app.dependency_overrides.clear()
```

### Test Execution

**Running Tests**:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_api_models.py

# Run tests matching pattern
pytest -k "test_classify"

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto

# Run only failed tests from last run
pytest --lf
```

**Test Coverage Requirements**:
- Minimum 85% overall coverage
- 90% coverage for critical components (API, models, security)
- 100% coverage for new features

## API Development

### FastAPI Best Practices

**Endpoint Design**:
```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="AICrisisAlert API",
    description="AI-powered crisis management system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Request/Response models with examples
class CrisisClassificationRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=10000)
    source: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "URGENT: People trapped in building collapse!",
                "source": "twitter"
            }
        }

# Endpoint with proper documentation
@app.post(
    "/classify",
    response_model=CrisisClassificationResponse,
    tags=["Classification"],
    summary="Classify crisis text",
    description="Classify a single text input for crisis classification.",
    responses={
        200: {"description": "Classification successful"},
        400: {"description": "Invalid input"},
        401: {"description": "Authentication required"},
        500: {"description": "Classification failed"}
    }
)
async def classify_crisis(
    request: CrisisClassificationRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service_dependency),
    _: None = Depends(check_rate_limit)
):
    """Classify crisis text with comprehensive error handling."""
    try:
        result = await model_service.classify_text(request.text)
        
        # Background logging
        background_tasks.add_task(
            log_classification,
            request.text,
            result.predicted_class,
            result.confidence
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")
```

**Middleware Development**:
```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with timing and security info."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to context
        request.state.request_id = request_id
        
        # Log request start
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host,
            user_agent=request.headers.get("user-agent", "Unknown")
        )
        
        try:
            response = await call_next(request)
            
            # Log successful completion
            processing_time = time.time() - start_time
            logger.info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                processing_time=processing_time
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str(processing_time)
            
            return response
            
        except Exception as e:
            # Log errors
            processing_time = time.time() - start_time
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                processing_time=processing_time
            )
            raise

# Add middleware to app
app.add_middleware(RequestLoggingMiddleware)
```

### API Testing

**Manual Testing with HTTPie**:
```bash
# Health check
http GET localhost:8000/health

# Authentication test
http POST localhost:8000/classify \
    Authorization:"Bearer your-api-key" \
    text="URGENT: People trapped!"

# Batch classification
http POST localhost:8000/classify/batch \
    Authorization:"Bearer your-api-key" \
    texts:='["URGENT: Help needed!", "Power outage downtown"]'
```

**Automated API Testing**:
```python
# tests/api/test_endpoints.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
class TestClassificationEndpoints:
    """Test classification API endpoints."""
    
    async def test_classify_endpoint_comprehensive(self, client: AsyncClient):
        """Comprehensive test of classify endpoint."""
        test_cases = [
            {
                "input": {"text": "URGENT: People trapped in building!"},
                "expected_category": "urgent_help",
                "min_confidence": 0.7
            },
            {
                "input": {"text": "Power lines down on Main Street"},
                "expected_category": "infrastructure_damage",
                "min_confidence": 0.6
            }
        ]
        
        for case in test_cases:
            response = await client.post("/classify", json=case["input"])
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["predicted_class"] == case["expected_category"]
            assert data["confidence"] >= case["min_confidence"]
            assert "processing_time_ms" in data
            assert data["processing_time_ms"] > 0
```

## Model Development

### Training Pipeline

**Training Script Structure**:
```python
# src/training/enhanced_train.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.training.configs import EnhancedTrainingConfig
from src.training.dataset_utils import CrisisDataset
from src.utils.crisis_features import CrisisFeatureExtractor

class CrisisTrainer:
    """Enhanced crisis classification trainer."""
    
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.feature_extractor = CrisisFeatureExtractor()
    
    def setup(self):
        """Initialize model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.config.crisis_labels)
        ).to(self.device)
    
    def train(self, train_dataset, val_dataset):
        """Main training loop with comprehensive logging."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        best_f1 = 0.0
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_dataset, optimizer)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_dataset)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}",
                train_loss=train_loss,
                val_loss=val_loss,
                val_f1=val_metrics["f1_macro"]
            )
            
            # Save best model
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                self._save_checkpoint(epoch, val_metrics)
    
    def _train_epoch(self, dataset, optimizer):
        """Single training epoch with proper error handling."""
        self.model.train()
        total_loss = 0.0
        predictions, targets = [], []
        
        try:
            for batch in dataset:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item()
                predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                targets.extend(batch["labels"].cpu().numpy())
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("GPU out of memory during training")
                torch.cuda.empty_cache()
                raise
            else:
                raise
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        return total_loss / len(dataset), metrics
```

### Model Evaluation

**Comprehensive Evaluation**:
```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self, model, tokenizer, feature_extractor):
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.crisis_labels = [
            "urgent_help", "infrastructure_damage", "casualty_info",
            "resource_availability", "general_info"
        ]
    
    def evaluate_comprehensive(self, test_dataset):
        """Comprehensive model evaluation."""
        predictions, targets, confidences = [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataset:
                outputs = self.model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                batch_preds = torch.argmax(outputs.logits, dim=-1)
                batch_confs = torch.max(probs, dim=-1)[0]
                
                predictions.extend(batch_preds.cpu().numpy())
                targets.extend(batch["labels"].cpu().numpy())
                confidences.extend(batch_confs.cpu().numpy())
        
        # Generate comprehensive report
        report = {
            "classification_report": classification_report(
                targets, predictions, 
                target_names=self.crisis_labels,
                output_dict=True
            ),
            "confusion_matrix": confusion_matrix(targets, predictions),
            "confidence_stats": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences)
            }
        }
        
        # Visualizations
        self._plot_confusion_matrix(report["confusion_matrix"])
        self._plot_confidence_distribution(confidences)
        
        return report
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', 
            xticklabels=self.crisis_labels,
            yticklabels=self.crisis_labels,
            cmap='Blues'
        )
        plt.title('Crisis Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
```

## Database Management

### Alembic Migrations

**Creating Migrations**:
```bash
# Generate new migration
alembic revision --autogenerate -m "Add new crisis categories"

# Review migration file
# Edit alembic/versions/001_add_crisis_categories.py if needed

# Apply migration
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

**Migration Example**:
```python
# alembic/versions/001_add_crisis_categories.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

def upgrade():
    """Add new crisis categories table."""
    op.create_table(
        'crisis_categories',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(50), nullable=False, unique=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('severity_level', sa.Integer, nullable=False, default=1),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now())
    )
    
    # Create index
    op.create_index(
        'idx_crisis_categories_severity',
        'crisis_categories',
        ['severity_level']
    )
    
    # Insert default categories
    crisis_categories_table = sa.table(
        'crisis_categories',
        sa.column('id'),
        sa.column('name'),
        sa.column('description'),
        sa.column('severity_level')
    )
    
    op.bulk_insert(crisis_categories_table, [
        {'name': 'urgent_help', 'description': 'Immediate assistance required', 'severity_level': 5},
        {'name': 'infrastructure_damage', 'description': 'Infrastructure damage reports', 'severity_level': 3},
        {'name': 'casualty_info', 'description': 'Casualty and injury information', 'severity_level': 4},
        {'name': 'resource_availability', 'description': 'Available resources and volunteers', 'severity_level': 2},
        {'name': 'general_info', 'description': 'General crisis information', 'severity_level': 1}
    ])

def downgrade():
    """Remove crisis categories table."""
    op.drop_index('idx_crisis_categories_severity')
    op.drop_table('crisis_categories')
```

### Database Testing

**Database Test Fixtures**:
```python
# conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.api.database import Base

@pytest.fixture(scope="session")
def test_db_engine():
    """Create test database engine."""
    engine = create_engine("sqlite:///./test.db", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()

@pytest.fixture
def test_db_session(test_db_engine):
    """Create test database session."""
    Session = sessionmaker(bind=test_db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def sample_classification_data():
    """Sample classification data for testing."""
    return {
        "text": "URGENT: People trapped in building collapse!",
        "category": "urgent_help",
        "confidence": 0.92,
        "model_name": "bertweet-crisis-v1",
        "processing_time_ms": 150.5
    }
```

## Contributing Guidelines

### Pull Request Process

1. **Fork and Clone**:
```bash
git clone https://github.com/your-username/AICrisisAlert.git
cd AICrisisAlert
git remote add upstream https://github.com/original-org/AICrisisAlert.git
```

2. **Create Feature Branch**:
```bash
git checkout -b feature/improve-classification-accuracy
```

3. **Implement Changes**:
- Write code following style guidelines
- Add comprehensive tests
- Update documentation
- Run pre-commit checks

4. **Test Changes**:
```bash
# Run all tests
pytest

# Run specific tests
pytest tests/unit/test_new_feature.py

# Check coverage
pytest --cov=src --cov-report=term-missing
```

5. **Commit Changes**:
```bash
git add .
git commit -m "feat: improve classification accuracy with ensemble methods

- Add ensemble classifier with voting mechanism
- Implement confidence-based weighting
- Add comprehensive tests for ensemble methods
- Update API to support ensemble predictions

Closes #123"
```

6. **Push and Create PR**:
```bash
git push origin feature/improve-classification-accuracy
# Create pull request on GitHub
```

### Code Review Guidelines

**Reviewer Checklist**:
- [ ] Code follows project style guidelines
- [ ] All tests pass and coverage is adequate
- [ ] Security implications considered
- [ ] Performance impact assessed
- [ ] Documentation updated appropriately
- [ ] Breaking changes clearly documented
- [ ] Error handling is comprehensive

**Common Review Comments**:
- "Consider adding type hints for better code clarity"
- "This could benefit from additional error handling"
- "Please add a test case for the edge case scenario"
- "Documentation should include usage examples"
- "Performance: Consider caching this expensive operation"

## Troubleshooting

### Common Development Issues

**Issue**: ImportError for local modules
```bash
# Solution: Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use pip install in development mode
pip install -e .
```

**Issue**: Database connection errors
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check connection string
echo $DATABASE_URL

# Reset database
docker-compose down
docker-compose up -d postgres
alembic upgrade head
```

**Issue**: Model loading errors
```bash
# Check model files exist
ls -la outputs/models/bertweet_enhanced/

# Use simple API mode for development
export USE_SIMPLE_API=true
python scripts/start_api.py
```

**Issue**: GPU out of memory during training
```python
# Add to training script
torch.cuda.empty_cache()

# Reduce batch size in config
batch_size = 8  # Instead of 32

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### Debugging Tips

**API Debugging**:
```python
# Add debug logging
import logging
logging.getLogger("src.api").setLevel(logging.DEBUG)

# Use uvicorn with reload
uvicorn src.api.main:app --reload --log-level debug

# Add breakpoints
import pdb; pdb.set_trace()
```

**Model Debugging**:
```python
# Inspect model outputs
with torch.no_grad():
    outputs = model(**inputs)
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Logits values: {outputs.logits}")
    
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### Performance Optimization

**Profiling Code**:
```python
import cProfile
import pstats

# Profile function
def profile_classification():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    result = classify_text("Sample text")
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass
```

**Database Query Optimization**:
```python
# Use query explain
query = session.query(Classification).filter(
    Classification.category == 'urgent_help'
)
print(query.statement.compile(compile_kwargs={"literal_binds": True}))

# Add indexes for frequent queries
# In migration file:
op.create_index('idx_classification_category_created', 'classifications', ['category', 'created_at'])
```

This developer guide provides a comprehensive foundation for contributing to the AICrisisAlert project. Regular updates to this guide ensure it remains current with project evolution and best practices.