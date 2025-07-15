"""
AICrisisAlert FastAPI Application

Main API server for crisis classification and emergency response coordination.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
from datetime import datetime

from .config import get_settings
from .dependencies import get_model_service_dependency
from .models import (
    CrisisClassificationRequest,
    CrisisClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    HealthCheckResponse,
    ModelInfoResponse
)
from .services import ModelService
from .middleware import RequestLoggingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting AICrisisAlert API server")
    settings = get_settings()
    
    # Initialize model service
    model_service = get_model_service_dependency()
    await model_service.initialize()
    
    logger.info("AICrisisAlert API server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AICrisisAlert API server")
    await model_service.cleanup()
    logger.info("AICrisisAlert API server shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="AICrisisAlert API",
    description="AI-powered crisis management and alert system for emergency response coordination",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure appropriately
app.add_middleware(RequestLoggingMiddleware)

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

# Model information endpoint
@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info(model_service: ModelService = Depends(get_model_service_dependency)):
    """Get information about the loaded crisis classification model."""
    try:
        info = await model_service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get model information")

# Single classification endpoint
@app.post("/classify", response_model=CrisisClassificationResponse, tags=["Classification"])
async def classify_crisis(
    request: CrisisClassificationRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service_dependency)
):
    """
    Classify a single text input for crisis classification.
    
    Returns the predicted crisis category and confidence scores.
    """
    try:
        logger.info("Processing classification request", text_length=len(request.text))
        
        # Add background task for logging/analytics
        background_tasks.add_task(model_service.log_classification, request.text)
        
        # Perform classification
        result = await model_service.classify_text(request.text)
        
        logger.info("Classification completed", 
                   predicted_class=result.predicted_class,
                   confidence=result.confidence)
        
        return result
        
    except Exception as e:
        logger.error("Classification failed", error=str(e), text=request.text[:100])
        raise HTTPException(status_code=500, detail="Classification failed")

# Batch classification endpoint
@app.post("/classify/batch", response_model=BatchClassificationResponse, tags=["Classification"])
async def classify_crisis_batch(
    request: BatchClassificationRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service_dependency)
):
    """
    Classify multiple text inputs for crisis classification.
    
    Returns predictions for all inputs in the batch.
    """
    try:
        logger.info("Processing batch classification request", 
                   batch_size=len(request.texts))
        
        # Add background task for logging/analytics
        background_tasks.add_task(model_service.log_batch_classification, len(request.texts))
        
        # Perform batch classification
        results = await model_service.classify_batch(request.texts)
        
        logger.info("Batch classification completed", 
                   batch_size=len(results),
                   avg_confidence=sum(r.confidence for r in results) / len(results))
        
        return BatchClassificationResponse(results=results)
        
    except Exception as e:
        logger.error("Batch classification failed", error=str(e))
        raise HTTPException(status_code=500, detail="Batch classification failed")

# Emergency endpoint for high-priority classifications
@app.post("/classify/emergency", response_model=CrisisClassificationResponse, tags=["Emergency"])
async def classify_emergency(
    request: CrisisClassificationRequest,
    model_service: ModelService = Depends(get_model_service_dependency)
):
    """
    Emergency classification endpoint with priority processing.
    
    Use this endpoint for urgent crisis situations requiring immediate response.
    """
    try:
        logger.warning("Emergency classification request received", 
                      text_length=len(request.text))
        
        # Perform classification with emergency priority
        result = await model_service.classify_emergency(request.text)
        
        logger.warning("Emergency classification completed", 
                      predicted_class=result.predicted_class,
                      confidence=result.confidence)
        
        return result
        
    except Exception as e:
        logger.error("Emergency classification failed", error=str(e))
        raise HTTPException(status_code=500, detail="Emergency classification failed")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 