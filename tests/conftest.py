"""
Pytest configuration for AICrisisAlert tests.
"""

import pytest
import asyncio
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from src.api.main import app
from src.api.database import Base, get_db
from src.utils.config import get_settings

# Override settings for testing
get_settings.cache_clear()

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def test_db():
    """
    Create a test database and override the get_db dependency.
    Yields a SQLAlchemy session factory (not a session).
    Cleans up and resets dependency overrides after test.
    """
    SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal
    app.dependency_overrides = {}  # Reset overrides after test
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(test_db):
    """Create a test client using the overridden test DB."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_crisis_text():
    """Sample crisis text for testing."""
    return {
        "urgent": "URGENT: People trapped in collapsed building on Main St! Need immediate help!",
        "infrastructure": "Power outage affecting downtown area, traffic lights not working",
        "casualty": "3 people injured in car accident near hospital, ambulance needed",
        "resource": "We have food and water available for 50 people at community center",
        "info": "Hurricane update: Category 3, expected landfall in 6 hours"
    } 