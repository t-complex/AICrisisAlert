"""
AICrisisAlert FastAPI Application

Main API server for crisis classification and emergency response coordination.
"""

import logging
import asyncio
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

from .config import get_settings
from .dependencies import get_model_service_dependency, verify_api_key, check_rate_limit
from .models import (
    CrisisClassificationRequest,
    CrisisClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    HealthCheckResponse,
    ModelInfoResponse
)
from .database import engine, Base, get_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting AICrisisAlert API...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Load models
    try:
        model_service = get_model_service_dependency()
        await model_service.initialize()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # In production, we might want to fail fast
        if os.getenv("ENVIRONMENT") == "production":
            raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AICrisisAlert API...")
    # Cleanup resources
    await asyncio.sleep(0.1)  # Allow pending tasks to complete

# Create FastAPI application
app = FastAPI(
    title="AICrisisAlert API",
    description="AI-powered crisis management and alert system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to AICrisisAlert API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check(db=Depends(get_db)):
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "uptime_seconds": None  # Could be calculated if needed
    }
    
    # Check database
    try:
        db.execute("SELECT 1")
        health_status["status"] = "healthy"
    except Exception as e:
        health_status["status"] = "degraded"
        logger.error(f"Database health check failed: {e}")
    
    return HealthCheckResponse(**health_status)

# Single classification endpoint
@app.post("/classify", response_model=CrisisClassificationResponse, tags=["Classification"])
async def classify_crisis(
    request: CrisisClassificationRequest,
    background_tasks: BackgroundTasks,
    model_service=Depends(get_model_service_dependency),
    _: None = Depends(check_rate_limit)  # Requires API key and rate limit check
):
    """
    Classify a single text input for crisis classification.
    
    Returns the predicted crisis category and confidence scores.
    """
    try:
        logger.info(f"Processing classification request, text length: {len(request.text)}")
        
        # Add background task for logging/analytics
        background_tasks.add_task(log_classification, request.text)
        
        # Perform classification
        result = await model_service.classify_text(request.text)
        
        logger.info(f"Classification completed - predicted_class: {result.predicted_class}, confidence: {result.confidence}")
        
        return result
        
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}, text: {request.text[:100]}")
        raise HTTPException(status_code=500, detail="Classification failed")

# Batch classification endpoint
@app.post("/classify/batch", response_model=BatchClassificationResponse, tags=["Classification"])
async def classify_crisis_batch(
    request: BatchClassificationRequest,
    background_tasks: BackgroundTasks,
    model_service=Depends(get_model_service_dependency),
    _: None = Depends(check_rate_limit)  # Requires API key and rate limit check
):
    """
    Classify multiple text inputs for crisis classification.
    
    Returns predictions for all inputs in the batch.
    """
    try:
        logger.info(f"Processing batch classification request, batch_size: {len(request.texts)}")
        
        # Validate batch size
        if len(request.texts) > settings.batch_size_limit:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum of {settings.batch_size_limit}"
            )
        
        # Add background task for logging/analytics
        background_tasks.add_task(log_batch_classification, len(request.texts))
        
        # Perform batch classification
        results = await model_service.classify_batch(request.texts)
        
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
        logger.info(f"Batch classification completed - batch_size: {len(results)}, avg_confidence: {avg_confidence}")
        
        return BatchClassificationResponse(
            results=results,
            total_processing_time_ms=0.0,  # Could be calculated
            batch_size=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch classification failed")

# Emergency endpoint for high-priority classifications
@app.post("/classify/emergency", response_model=CrisisClassificationResponse, tags=["Emergency"])
async def classify_emergency(
    request: CrisisClassificationRequest,
    model_service=Depends(get_model_service_dependency),
    api_key: str = Depends(verify_api_key)  # Requires API key but no rate limit for emergencies
):
    """
    Emergency classification endpoint with priority processing.
    
    Use this endpoint for urgent crisis situations requiring immediate response.
    """
    try:
        logger.warning(f"Emergency classification request received, text length: {len(request.text)}")
        
        # Perform classification with emergency priority
        result = await model_service.classify_emergency(request.text)
        
        logger.warning(f"Emergency classification completed - predicted_class: {result.predicted_class}, confidence: {result.confidence}")
        
        return result
        
    except Exception as e:
        logger.error(f"Emergency classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Emergency classification failed")

# Model information endpoint  
@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info(
    model_service=Depends(get_model_service_dependency),
    api_key: str = Depends(verify_api_key)  # Requires API key
):
    """Get information about the loaded crisis classification model."""
    try:
        info = await model_service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

# Background tasks
async def log_classification(text: str):
    """Log classification to database."""
    try:
        # Implementation for database logging
        logger.info(f"Logging classification for text: {text[:100]}...")
    except Exception as e:
        logger.error(f"Failed to log classification: {e}")

async def log_batch_classification(batch_size: int):
    """Log batch classification to database."""
    try:
        # Implementation for batch database logging
        logger.info(f"Logging batch classification for {batch_size} texts")
    except Exception as e:
        logger.error(f"Failed to log batch classification: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 