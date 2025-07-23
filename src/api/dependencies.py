"""
Dependency injection for AICrisisAlert API.
"""

import time
from typing import Dict
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .services import get_model_service, ModelService
from .config import get_settings

security = HTTPBearer()
settings = get_settings()

# Simple rate limiting store (in production, use Redis)
rate_limit_store: Dict[str, Dict[str, int]] = {}

def get_model_service_dependency() -> ModelService:
    """Get model service dependency."""
    return get_model_service()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify API key authentication.
    
    Args:
        credentials: Bearer token credentials
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials or credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication scheme. Use Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key = credentials.credentials
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key

def check_rate_limit(api_key: str = Depends(verify_api_key)) -> None:
    """
    Check rate limiting for API requests.
    
    Args:
        api_key: Verified API key
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    current_time = int(time.time())
    window_start = current_time - (current_time % settings.rate_limit_period)
    
    if api_key not in rate_limit_store:
        rate_limit_store[api_key] = {}
    
    user_requests = rate_limit_store[api_key]
    
    # Clean old windows
    for window in list(user_requests.keys()):
        if window < window_start:
            del user_requests[window]
    
    # Check current window
    if window_start not in user_requests:
        user_requests[window_start] = 0
    
    if user_requests[window_start] >= settings.rate_limit_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {settings.rate_limit_requests} requests per {settings.rate_limit_period} seconds.",
            headers={"Retry-After": str(settings.rate_limit_period)}
        )
    
    user_requests[window_start] += 1 