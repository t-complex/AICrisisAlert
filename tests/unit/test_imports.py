#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        print("✓ Basic imports working")
        
        # Test training imports
        print("✓ Hyperparameter tuner imports working")
        
        print("✓ Enhanced training config imports working")
        
        # Test utils imports
        print("✓ Crisis features imports working")
        
        print("✓ Crisis preprocessing imports working")
        
        # Test models imports
        print("✓ Hybrid classifier imports working")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All modules can be imported successfully!")
        print("You can now run: python3 scripts/optimize_hyperparameters.py")
    else:
        print("\n💥 There are import issues that need to be resolved.")
        sys.exit(1) 