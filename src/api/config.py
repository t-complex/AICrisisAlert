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
        self.secret_key = os.getenv("SECRET_KEY", "your-secret-key-here")
        self.allowed_hosts = os.getenv("ALLOWED_HOSTS", "*").split(",")
        self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        
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