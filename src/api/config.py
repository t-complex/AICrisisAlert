"""
Configuration management for AICrisisAlert API.
"""

import os
from functools import lru_cache

class Settings:
    """Application settings."""
    
    def __init__(self):
        # API Configuration
        self.api_title = os.getenv("API_TITLE", "AICrisisAlert API")
        self.api_version = os.getenv("API_VERSION", "1.0.0")
        self.api_description = os.getenv("API_DESCRIPTION", "AI-powered crisis management and alert system")
        
        # Server Configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.reload = os.getenv("RELOAD", "false").lower() == "true"
        self.workers = int(os.getenv("WORKERS", "1"))
        
        # Model Configuration
        self.model_path = os.getenv("MODEL_PATH", "outputs/models/bertweet_enhanced")
        self.model_type = os.getenv("MODEL_TYPE", "bertweet")
        self.model_version = os.getenv("MODEL_VERSION", "1.0.0")
        
        # Database Configuration
        self.database_url = os.getenv("DATABASE_URL")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Security Configuration
        self.secret_key = os.getenv("SECRET_KEY")
        if not self.secret_key or self.secret_key == "your-secret-key-here":
            raise ValueError("SECRET_KEY environment variable must be set to a secure value")
        if len(self.secret_key) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
            
        self.allowed_hosts = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
        # Security: Don't allow wildcard CORS origins by default
        cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
        self.cors_origins = cors_origins_env.split(",") if cors_origins_env != "*" else []
        if not self.cors_origins:
            raise ValueError("CORS_ORIGINS must be explicitly configured - wildcard (*) not allowed for security")
        
        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_format = os.getenv("LOG_FORMAT", "json")
        
        # Monitoring Configuration
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.metrics_port = int(os.getenv("METRICS_PORT", "9090"))
        
        # AWS Configuration (for production)
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.s3_bucket = os.getenv("S3_BUCKET")
        
        # Crisis Management Configuration
        self.emergency_threshold = float(os.getenv("EMERGENCY_THRESHOLD", "0.8"))
        self.batch_size_limit = int(os.getenv("BATCH_SIZE_LIMIT", "100"))
        self.max_text_length = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
        
        # API Security Configuration
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY environment variable must be set for authentication")
        if len(self.api_key) < 32:
            raise ValueError("API_KEY must be at least 32 characters long")
            
        # Rate limiting configuration
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_period = int(os.getenv("RATE_LIMIT_PERIOD", "3600"))  # 1 hour
        
        # Notification Configuration
        self.enable_notifications = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()

# Environment-specific configurations
def get_development_settings() -> Settings:
    """Get development-specific settings."""
    settings = Settings()
    settings.reload = True
    settings.log_level = "DEBUG"
    settings.enable_metrics = False
    settings.database_url = "sqlite:///./dev.db"
    return settings

def get_production_settings() -> Settings:
    """Get production-specific settings."""
    settings = Settings()
    settings.reload = False
    settings.workers = 4
    settings.log_level = "WARNING"
    settings.enable_metrics = True
    return settings

def get_test_settings() -> Settings:
    """Get test-specific settings."""
    settings = Settings()
    settings.database_url = "sqlite:///./test.db"
    settings.log_level = "ERROR"
    settings.enable_metrics = False
    return settings 