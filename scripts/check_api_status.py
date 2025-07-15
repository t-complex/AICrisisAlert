#!/usr/bin/env python3
"""
Quick API Status Checker for AICrisisAlert

This script provides a simple way to check if the API is running and responding.
Useful for troubleshooting and verification.
"""

import requests
import sys
import time

def check_api_status(base_url: str = "http://127.0.0.1:8000", timeout: int = 10):
    """
    Check if the API is running and responding.
    
    Args:
        base_url: API base URL
        timeout: Request timeout in seconds
    
    Returns:
        dict: Status information
    """
    print("🔍 Checking AICrisisAlert API Status...")
    print("=" * 50)
    
    status = {
        "api_running": False,
        "health_check": False,
        "classification": False,
        "response_time": None,
        "errors": []
    }
    
    # Check if API is reachable
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/health", timeout=timeout)
        response_time = time.time() - start_time
        status["response_time"] = response_time
        
        if response.status_code == 200:
            status["api_running"] = True
            status["health_check"] = True
            data = response.json()
            print(f"✅ API is running and healthy")
            print(f"   Response time: {response_time:.3f} seconds")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Version: {data.get('version', 'unknown')}")
        else:
            status["errors"].append(f"Health check failed with status {response.status_code}")
            print(f"❌ API responded but health check failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        status["errors"].append("Connection refused - API is not running")
        print("❌ API is not running")
        print("💡 Start the API with: python scripts/start_api.py")
        return status
        
    except requests.exceptions.Timeout:
        status["errors"].append(f"Request timed out after {timeout} seconds")
        print(f"⏰ API request timed out after {timeout} seconds")
        return status
        
    except Exception as e:
        status["errors"].append(f"Unexpected error: {str(e)}")
        print(f"❌ Unexpected error: {e}")
        return status
    
    # Test classification endpoint
    if status["api_running"]:
        try:
            test_data = {"text": "URGENT: People trapped in building collapse!"}
            response = requests.post(
                f"{base_url}/classify",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            
            if response.status_code == 200:
                status["classification"] = True
                data = response.json()
                print(f"✅ Classification endpoint working")
                print(f"   Test result: {data.get('predicted_class', 'unknown')} ({data.get('confidence', 0):.3f})")
            else:
                status["errors"].append(f"Classification failed with status {response.status_code}")
                print(f"❌ Classification endpoint failed: {response.status_code}")
                
        except requests.exceptions.Timeout:
            status["errors"].append(f"Classification request timed out after {timeout} seconds")
            print(f"⏰ Classification request timed out after {timeout} seconds")
            
        except Exception as e:
            status["errors"].append(f"Classification error: {str(e)}")
            print(f"❌ Classification error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 API Status Summary:")
    print(f"   API Running: {'✅' if status['api_running'] else '❌'}")
    print(f"   Health Check: {'✅' if status['health_check'] else '❌'}")
    print(f"   Classification: {'✅' if status['classification'] else '❌'}")
    
    if status["response_time"]:
        print(f"   Response Time: {status['response_time']:.3f}s")
    
    if status["errors"]:
        print(f"\n⚠️  Issues Found:")
        for error in status["errors"]:
            print(f"   - {error}")
    
    if status["api_running"] and status["health_check"] and status["classification"]:
        print("\n🎉 API is working correctly!")
        print("   Available endpoints:")
        print("   - GET  /health")
        print("   - GET  /model/info")
        print("   - POST /classify")
        print("   - POST /classify/batch")
        print("   - POST /classify/emergency")
        print("   - GET  /docs (API documentation)")
    else:
        print("\n💡 Troubleshooting Tips:")
        print("   1. Start the API: python scripts/start_api.py")
        print("   2. Check if port 8000 is available")
        print("   3. Use simple mode: USE_SIMPLE_API=true")
        print("   4. Check logs for errors")
    
    return status

def main():
    """Main function."""
    # Parse command line arguments
    base_url = "http://127.0.0.1:8000"
    timeout = 10
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    if len(sys.argv) > 2:
        timeout = int(sys.argv[2])
    
    status = check_api_status(base_url, timeout)
    
    # Exit with appropriate code
    if status["api_running"] and status["health_check"] and status["classification"]:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main() 