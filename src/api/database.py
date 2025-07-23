# src/api/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid

from .config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database Models
class Classification(Base):
    """Store classification results."""
    __tablename__ = "classifications"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    text = Column(Text, nullable=False)
    category = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    processing_time_ms = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Location data
    location_extracted = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Source information
    source_platform = Column(String(50))  # twitter, facebook, etc.
    source_id = Column(String(200))
    source_timestamp = Column(DateTime)
    
    # Crisis metadata
    urgency_score = Column(Float)
    verified = Column(Boolean, default=False)
    responded = Column(Boolean, default=False)
    
    # Relationships
    features = relationship("ExtractedFeature", back_populates="classification", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="classification")
    
    __table_args__ = (
        Index('idx_created_category', 'created_at', 'category'),
        Index('idx_urgency_responded', 'urgency_score', 'responded'),
    )

class ExtractedFeature(Base):
    """Store extracted features for each classification."""
    __tablename__ = "extracted_features"
    
    id = Column(Integer, primary_key=True)
    classification_id = Column(String, ForeignKey("classifications.id"), nullable=False)
    
    # Feature categories
    emergency_keywords_count = Column(Integer, default=0)
    location_mentions_count = Column(Integer, default=0)
    urgency_indicators_count = Column(Integer, default=0)
    casualty_indicators_count = Column(Integer, default=0)
    infrastructure_keywords_count = Column(Integer, default=0)
    
    # Detailed features
    features_json = Column(Text)  # JSON string of all features
    
    classification = relationship("Classification", back_populates="features")

class Alert(Base):
    """Store generated alerts."""
    __tablename__ = "alerts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    classification_id = Column(String, ForeignKey("classifications.id"), nullable=False)
    
    alert_type = Column(String(50), nullable=False)  # urgent, infrastructure, casualty, etc.
    severity = Column(String(20), nullable=False)  # critical, high, medium, low
    
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Alert status
    status = Column(String(50), default="pending")  # pending, sent, acknowledged, resolved
    sent_at = Column(DateTime)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Response tracking
    responder_id = Column(String(100))
    response_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    classification = relationship("Classification", back_populates="alerts")
    
    __table_args__ = (
        Index('idx_alert_status', 'status', 'severity'),
    )

class ModelMetrics(Base):
    """Track model performance metrics."""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    humanitarian_f1 = Column(Float)
    
    # Timing metrics
    avg_inference_time_ms = Column(Float)
    total_predictions = Column(Integer, default=0)
    
    # Evaluation details
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    test_set_size = Column(Integer)
    confusion_matrix_json = Column(Text)
    
    __table_args__ = (
        Index('idx_model_version', 'model_name', 'model_version'),
    )

# Create tables
Base.metadata.create_all(bind=engine)