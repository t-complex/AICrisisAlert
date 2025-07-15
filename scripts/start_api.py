#!/usr/bin/env python3
"""
Startup script for AICrisisAlert API

This script starts the FastAPI server with proper configuration.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Start the API server."""
    print("🚨 Starting AICrisisAlert API Server...")
    print("=" * 60)
    
    # Set environment variables
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("HOST", "0.0.0.0")
    os.environ.setdefault("PORT", "8000")
    
    # Check if we should use the simple version
    use_simple = os.getenv("USE_SIMPLE_API", "true").lower() == "true"
    
    if use_simple:
        print("📝 Using SIMPLIFIED API (mock model) - Fast startup, no ML dependencies")
        print("   This mode is perfect for testing and development")
        app_module = "src.api.main_simple:app"
    else:
        print("🤖 Using FULL API (requires trained model) - May take time to load")
        print("   This mode requires the trained BERTweet model")
        app_module = "src.api.main:app"
    
    port = os.getenv("PORT", "8000")
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🌐 API will be available at: http://localhost:{port}")
    print(f"📚 Documentation: http://localhost:{port}/docs")
    print(f"🔍 Health check: http://localhost:{port}/health")
    print("=" * 60)
    
    # Start the server with timeout settings
    uvicorn.run(
        app_module,
        host=host,
        port=int(port),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        timeout_keep_alive=30,  # Keep connections alive for 30 seconds
        timeout_graceful_shutdown=30,  # Graceful shutdown timeout
        access_log=True
    )

if __name__ == "__main__":
    main() 