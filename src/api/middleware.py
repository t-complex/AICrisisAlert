"""
Middleware for AICrisisAlert API.
"""

import time
import uuid
from typing import Callable
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started - ID: {request_id}, Method: {request.method}, URL: {request.url}, "
            f"Client IP: {request.client.host if request.client else 'Unknown'}, "
            f"User Agent: {request.headers.get('user-agent', 'Unknown')}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log request completion
            logger.info(
                f"Request completed - ID: {request_id}, Status: {response.status_code}, "
                f"Processing Time: {processing_time:.3f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str(processing_time)
            
            return response
            
        except Exception as e:
            # Log request error
            processing_time = time.time() - start_time
            logger.error(
                f"Request failed - ID: {request_id}, Error: {str(e)}, "
                f"Processing Time: {processing_time:.3f}s"
            )
            raise 