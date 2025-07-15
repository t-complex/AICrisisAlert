"""
Simplified AICrisisAlert FastAPI Application

A working version without complex dependencies for initial testing.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Pydantic models
class CrisisClassificationRequest(BaseModel):
    text: str

class CrisisClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    processing_time_ms: float
    model_version: str

class BatchClassificationRequest(BaseModel):
    texts: List[str]

class BatchClassificationResponse(BaseModel):
    results: List[CrisisClassificationResponse]
    total_processing_time_ms: float
    batch_size: int

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    classes: List[str]
    loaded_at: str

# Create FastAPI application
app = FastAPI(
    title="AICrisisAlert API",
    description="AI-powered crisis management and alert system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock model service for testing
class MockModelService:
    def __init__(self):
        self.classes = ["urgent_help", "infrastructure_damage", "casualty_info", "resource_availability", "general_info"]
        self.model_version = "1.0.0"
    
    async def classify_text(self, text: str) -> CrisisClassificationResponse:
        """Mock classification for testing."""
        start_time = time.time()
        
        # Simple mock logic based on keywords
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["help", "urgent", "emergency", "trapped"]):
            predicted_class = "urgent_help"
            confidence = 0.85
        elif any(word in text_lower for word in ["damage", "destroyed", "broken", "power", "water"]):
            predicted_class = "infrastructure_damage"
            confidence = 0.80
        elif any(word in text_lower for word in ["injured", "dead", "missing", "hurt"]):
            predicted_class = "casualty_info"
            confidence = 0.75
        elif any(word in text_lower for word in ["volunteer", "donate", "available", "helping"]):
            predicted_class = "resource_availability"
            confidence = 0.70
        else:
            predicted_class = "general_info"
            confidence = 0.60
        
        # Create mock probabilities
        probabilities = {cls: 0.1 for cls in self.classes}
        probabilities[predicted_class] = confidence
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return CrisisClassificationResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=probabilities,
            processing_time_ms=processing_time_ms,
            model_version=self.model_version
        )

# Global mock service
mock_service = MockModelService()

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

# Model information endpoint
@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the model."""
    return ModelInfoResponse(
        model_name="bertweet-crisis-classifier",
        model_version="1.0.0",
        model_type="bertweet",
        classes=mock_service.classes,
        loaded_at=datetime.now().isoformat()
    )

# Single classification endpoint
@app.post("/classify", response_model=CrisisClassificationResponse)
async def classify_crisis(request: CrisisClassificationRequest):
    """Classify a single text input."""
    try:
        logger.info(f"Processing classification request: {len(request.text)} chars")
        
        result = await mock_service.classify_text(request.text)
        
        logger.info(f"Classification completed: {result.predicted_class} ({result.confidence:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")

# Batch classification endpoint
@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_crisis_batch(request: BatchClassificationRequest):
    """Classify multiple text inputs."""
    try:
        logger.info(f"Processing batch classification: {len(request.texts)} texts")
        
        start_time = time.time()
        results = []
        
        for text in request.texts:
            result = await mock_service.classify_text(text)
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Batch classification completed: {len(results)} results in {total_time:.2f}ms")
        
        return BatchClassificationResponse(
            results=results,
            total_processing_time_ms=total_time,
            batch_size=len(results)
        )
        
    except Exception as e:
        logger.error(f"Batch classification failed: {e}")
        raise HTTPException(status_code=500, detail="Batch classification failed")

# Emergency endpoint
@app.post("/classify/emergency", response_model=CrisisClassificationResponse)
async def classify_emergency(request: CrisisClassificationRequest):
    """Emergency classification with priority processing."""
    try:
        logger.warning(f"Emergency classification request: {len(request.text)} chars")
        
        result = await mock_service.classify_text(request.text)
        
        if result.confidence >= 0.8:
            logger.warning(f"High-confidence emergency: {result.predicted_class} ({result.confidence:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Emergency classification failed: {e}")
        raise HTTPException(status_code=500, detail="Emergency classification failed")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 