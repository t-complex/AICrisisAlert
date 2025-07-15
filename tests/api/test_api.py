#!/usr/bin/env python3
"""
Test script for AICrisisAlert API

This script tests the API endpoints to ensure they're working correctly.
"""

import requests
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🏥 Testing health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("🤖 Testing model info...")
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Model info: {data['model_name']} v{data['model_version']}")
        print(f"   Classes: {', '.join(data['classes'])}")
        return True
        
    except Exception as e:
        print(f"❌ Model info failed: {e}")
        return False

def test_single_classification():
    """Test single text classification."""
    print("📝 Testing single classification...")
    
    test_cases = [
        "URGENT: People trapped in building collapse! Need immediate help!",
        "Hurricane caused major damage to power lines and water systems",
        "Several people injured in the accident, need medical assistance",
        "Volunteers available to help with cleanup efforts",
        "General update on the current situation in the affected area"
    ]
    
    try:
        for i, text in enumerate(test_cases, 1):
            print(f"   Test {i}: {text[:50]}...")
            
            response = requests.post(
                f"{BASE_URL}/classify",
                json={"text": text}
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"      → {data['predicted_class']} (confidence: {data['confidence']:.3f})")
            
            time.sleep(0.1)  # Small delay between requests
        
        print("✅ Single classification tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Single classification failed: {e}")
        return False

def test_batch_classification():
    """Test batch classification."""
    print("📦 Testing batch classification...")
    
    texts = [
        "Emergency situation requires immediate response",
        "Infrastructure damage reported in downtown area",
        "Casualties reported at the scene",
        "Resources and volunteers are available",
        "General information about the crisis"
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/classify/batch",
            json={"texts": texts}
        )
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Batch classification: {data['batch_size']} texts processed")
        print(f"   Total time: {data['total_processing_time_ms']:.2f}ms")
        
        for i, result in enumerate(data['results'], 1):
            print(f"   Result {i}: {result['predicted_class']} ({result['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch classification failed: {e}")
        return False

def test_emergency_classification():
    """Test emergency classification endpoint."""
    print("🚨 Testing emergency classification...")
    
    emergency_text = "URGENT: Multiple people trapped in collapsed building! Immediate rescue needed!"
    
    try:
        response = requests.post(
            f"{BASE_URL}/classify/emergency",
            json={"text": emergency_text}
        )
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Emergency classification: {data['predicted_class']} ({data['confidence']:.3f})")
        
        if data['confidence'] >= 0.8:
            print("   ⚠️  High-confidence emergency detected!")
        
        return True
        
    except Exception as e:
        print(f"❌ Emergency classification failed: {e}")
        return False

def main():
    """Run all API tests."""
    print("🧪 Starting AICrisisAlert API Tests...")
    print("=" * 50)
    
    tests = [
        test_health_check,
        test_model_info,
        test_single_classification,
        test_batch_classification,
        test_emergency_classification
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the API server.")
    
    return passed == total

if __name__ == "__main__":
    main() 