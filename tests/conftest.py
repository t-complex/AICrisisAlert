"""
Pytest configuration for AICrisisAlert tests.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

@pytest.fixture
def sample_crisis_texts():
    """Sample crisis texts for testing."""
    return [
        "URGENT: People trapped in building collapse! Need immediate help!",
        "Hurricane caused major damage to power lines and water systems",
        "Several people injured in the accident, need medical assistance",
        "Volunteers available to help with cleanup efforts",
        "General update on the current situation in the affected area"
    ]

@pytest.fixture
def sample_crisis_labels():
    """Sample crisis labels for testing."""
    return [
        "urgent_help",
        "infrastructure_damage", 
        "casualty_info",
        "resource_availability",
        "general_info"
    ]

@pytest.fixture
def api_base_url():
    """Base URL for API testing."""
    return "http://localhost:8000" 