"""
Dependency injection for AICrisisAlert API.
"""

from .services import get_model_service, ModelService

def get_model_service_dependency() -> ModelService:
    """Get model service dependency."""
    return get_model_service() 