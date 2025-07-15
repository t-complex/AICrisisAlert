#!/usr/bin/env python3
"""
API Testing Script for AICrisisAlert

Tests all API endpoints with proper timeout handling and error reporting.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import aiohttp
import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class APITester:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        timeout_config = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout_config)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> Dict[str, Any]:
        """Test health check endpoint."""
        print("🔍 Testing health check endpoint...")
        if not self.session:
            return {"status": "error", "message": "Session not initialized"}
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed: {data}")
                    return {"status": "success", "data": data}
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return {"status": "error", "status_code": response.status}
        except asyncio.TimeoutError:
            print(f"⏰ Health check timed out after {self.timeout} seconds")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def test_model_info(self) -> Dict[str, Any]:
        """Test model info endpoint."""
        print("🤖 Testing model info endpoint...")
        if not self.session:
            return {"status": "error", "message": "Session not initialized"}
        try:
            async with self.session.get(f"{self.base_url}/model/info") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Model info: {data}")
                    return {"status": "success", "data": data}
                else:
                    print(f"❌ Model info failed: {response.status}")
                    return {"status": "error", "status_code": response.status}
        except asyncio.TimeoutError:
            print(f"⏰ Model info timed out after {self.timeout} seconds")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def test_single_classification(self) -> Dict[str, Any]:
        """Test single classification endpoint."""
        print("📝 Testing single classification endpoint...")
        if not self.session:
            return {"status": "error", "message": "Session not initialized"}
        test_data = {
            "text": "URGENT: People trapped in building collapse! Need immediate help!"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/classify",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Classification successful: {data['predicted_class']} ({data['confidence']:.3f})")
                    return {"status": "success", "data": data}
                else:
                    print(f"❌ Classification failed: {response.status}")
                    return {"status": "error", "status_code": response.status}
        except asyncio.TimeoutError:
            print(f"⏰ Classification timed out after {self.timeout} seconds")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def test_batch_classification(self) -> Dict[str, Any]:
        """Test batch classification endpoint."""
        print("📦 Testing batch classification endpoint...")
        if not self.session:
            return {"status": "error", "message": "Session not initialized"}
        test_data = {
            "texts": [
                "URGENT: People trapped in building collapse!",
                "Power lines down on Main Street",
                "Volunteers needed for cleanup efforts"
            ]
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/classify/batch",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Batch classification successful: {len(data['results'])} results")
                    return {"status": "success", "data": data}
                else:
                    print(f"❌ Batch classification failed: {response.status}")
                    return {"status": "error", "status_code": response.status}
        except asyncio.TimeoutError:
            print(f"⏰ Batch classification timed out after {self.timeout} seconds")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Batch classification error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def test_emergency_classification(self) -> Dict[str, Any]:
        """Test emergency classification endpoint."""
        print("🚨 Testing emergency classification endpoint...")
        if not self.session:
            return {"status": "error", "message": "Session not initialized"}
        test_data = {
            "text": "EMERGENCY: Multiple casualties reported at downtown explosion!"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/classify/emergency",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Emergency classification successful: {data['predicted_class']} ({data['confidence']:.3f})")
                    return {"status": "success", "data": data}
                else:
                    print(f"❌ Emergency classification failed: {response.status}")
                    return {"status": "error", "status_code": response.status}
        except asyncio.TimeoutError:
            print(f"⏰ Emergency classification timed out after {self.timeout} seconds")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Emergency classification error: {e}")
            return {"status": "error", "message": str(e)}

def test_api_sync():
    """Synchronous API testing using requests."""
    print("🧪 Running synchronous API tests...")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8000"
    timeout = 30
    
    # Test health check
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.Timeout:
        print(f"⏰ Health check timed out after {timeout} seconds")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test single classification
    print("\n📝 Testing single classification endpoint...")
    test_data = {"text": "URGENT: People trapped in building collapse!"}
    try:
        response = requests.post(
            f"{base_url}/classify",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Classification successful: {data['predicted_class']} ({data['confidence']:.3f})")
        else:
            print(f"❌ Classification failed: {response.status_code}")
    except requests.exceptions.Timeout:
        print(f"⏰ Classification timed out after {timeout} seconds")
    except Exception as e:
        print(f"❌ Classification error: {e}")

async def test_api_async():
    """Asynchronous API testing."""
    print("🧪 Running asynchronous API tests...")
    print("=" * 60)
    
    async with APITester() as tester:
        results = []
        
        # Test all endpoints
        results.append(await tester.test_health_check())
        results.append(await tester.test_model_info())
        results.append(await tester.test_single_classification())
        results.append(await tester.test_batch_classification())
        results.append(await tester.test_emergency_classification())
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 API Test Summary:")
        
        success_count = sum(1 for r in results if r["status"] == "success")
        timeout_count = sum(1 for r in results if r["status"] == "timeout")
        error_count = sum(1 for r in results if r["status"] == "error")
        
        print(f"✅ Successful: {success_count}/5")
        print(f"⏰ Timeouts: {timeout_count}/5")
        print(f"❌ Errors: {error_count}/5")
        
        if timeout_count > 0:
            print("\n⚠️  Timeout issues detected. This might be due to:")
            print("   - Model loading taking too long")
            print("   - Network connectivity issues")
            print("   - Server resource constraints")
            print("\n💡 Recommendations:")
            print("   - Use simple mode for testing: USE_SIMPLE_API=true")
            print("   - Increase timeout settings")
            print("   - Check server logs for errors")

def main():
    """Main function to run API tests."""
    print("🚨 AICrisisAlert API Testing")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding properly")
            print("💡 Start the API first: python scripts/start_api.py")
            return
    except:
        print("❌ API is not running")
        print("💡 Start the API first: python scripts/start_api.py")
        return
    
    # Run tests
    test_api_sync()
    
    # Run async tests
    asyncio.run(test_api_async())

if __name__ == "__main__":
    main() 