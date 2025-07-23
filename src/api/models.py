"""
Pydantic models for AICrisisAlert API requests and responses.
"""

from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

import re
from pydantic import BaseModel, Field, validator

class CrisisCategory(str, Enum):
    """Crisis classification categories."""
    URGENT_HELP = "urgent_help"
    INFRASTRUCTURE_DAMAGE = "infrastructure_damage"
    CASUALTY_INFO = "casualty_info"
    RESOURCE_AVAILABILITY = "resource_availability"
    GENERAL_INFO = "general_info"

class CrisisClassificationRequest(BaseModel):
    """Request model for crisis classification."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    source: Optional[str] = Field(None, description="Source of the text (e.g., 'twitter', 'facebook')")
    location: Optional[str] = Field(None, description="Geographic location if known")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the text")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', v.strip())
        
        # Check for potentially malicious content
        if re.search(r'<script|javascript:|data:|vbscript:', cleaned_text, re.IGNORECASE):
            raise ValueError('Text contains potentially malicious content')
        
        # Check for excessively repeated characters (potential spam/DoS)
        if re.search(r'(.)\1{100,}', cleaned_text):
            raise ValueError('Text contains excessive repeated characters')
        
        # Basic length validation (additional to Field constraint)
        if len(cleaned_text) < 3:
            raise ValueError('Text must be at least 3 characters long')
        
        return cleaned_text
    
    @validator('source')
    def validate_source(cls, v):
        if v is not None:
            v = v.strip()
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError('Source must contain only alphanumeric characters, underscores, and hyphens')
        return v
    
    @validator('location')
    def validate_location(cls, v):
        if v is not None:
            v = v.strip()
            # Basic location validation - allow letters, numbers, spaces, commas, periods
            if not re.match(r'^[a-zA-Z0-9\s,.-]+$', v):
                raise ValueError('Location contains invalid characters')
            if len(v) > 200:
                raise ValueError('Location must be less than 200 characters')
        return v

class CrisisClassificationResponse(BaseModel):
    """Response model for crisis classification."""
    predicted_class: CrisisCategory = Field(..., description="Predicted crisis category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    class_probabilities: Dict[str, float] = Field(..., description="Probability scores for all classes")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Version of the model used")
    features_used: Optional[List[str]] = Field(None, description="Features used in classification")
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_class": "urgent_help",
                "confidence": 0.92,
                "class_probabilities": {
                    "urgent_help": 0.92,
                    "infrastructure_damage": 0.05,
                    "casualty_info": 0.02,
                    "resource_availability": 0.01,
                    "general_info": 0.00
                },
                "processing_time_ms": 150.5,
                "model_version": "1.0.0",
                "features_used": ["text_length", "urgency_keywords", "location_indicators"]
            }
        }

class BatchClassificationRequest(BaseModel):
    """Request model for batch crisis classification."""
    texts: List[str] = Field(..., description="List of texts to classify")
    sources: Optional[List[str]] = Field(None, description="Sources for each text")
    locations: Optional[List[str]] = Field(None, description="Locations for each text")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not all(text.strip() for text in v):
            raise ValueError('All texts must be non-empty')
        return [text.strip() for text in v]

class BatchClassificationResponse(BaseModel):
    """Response model for batch crisis classification."""
    results: List[CrisisClassificationResponse] = Field(..., description="Classification results")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    batch_size: int = Field(..., description="Number of texts processed")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "predicted_class": "urgent_help",
                        "confidence": 0.92,
                        "class_probabilities": {
                            "urgent_help": 0.92,
                            "infrastructure_damage": 0.05,
                            "casualty_info": 0.02,
                            "resource_availability": 0.01,
                            "general_info": 0.00
                        },
                        "processing_time_ms": 150.5,
                        "model_version": "1.0.0"
                    }
                ],
                "total_processing_time_ms": 150.5,
                "batch_size": 1
            }
        }

class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "uptime_seconds": 3600.5
            }
        }

class ModelInfoResponse(BaseModel):
    """Response model for model information endpoint."""
    model_name: str = Field(..., description="Name of the loaded model")
    model_version: str = Field(..., description="Version of the model")
    model_type: str = Field(..., description="Type of model (e.g., 'bertweet', 'ensemble')")
    classes: List[str] = Field(..., description="Available classification classes")
    accuracy: Optional[float] = Field(None, description="Model accuracy on test set")
    f1_score: Optional[float] = Field(None, description="Model F1 score on test set")
    loaded_at: str = Field(..., description="When the model was loaded")
    last_updated: Optional[str] = Field(None, description="When the model was last updated")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "bertweet-crisis-classifier",
                "model_version": "1.0.0",
                "model_type": "bertweet",
                "classes": ["urgent_help", "infrastructure_damage", "casualty_info", "resource_availability", "general_info"],
                "accuracy": 0.84,
                "f1_score": 0.78,
                "loaded_at": "2024-01-15T10:00:00Z",
                "last_updated": "2024-01-15T09:30:00Z"
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Classification failed",
                "detail": "Model not loaded",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123456"
            }
        } 