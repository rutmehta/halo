#!/usr/bin/env python3
"""
Test Railway Deployed Halo Face Search API
"""

import requests
import json
import time
from pathlib import Path

# Replace with your actual Railway URL
RAILWAY_URL = "https://your-project-name.railway.app"

def test_health_check():
    """Test basic health check"""
    print("🏥 Testing Health Check...")
    try:
        response = requests.get(f"{RAILWAY_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint for API info"""
    print("\n🏠 Testing Root Endpoint...")
    try:
        response = requests.get(f"{RAILWAY_URL}/", timeout=10)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"API Status: {data.get('status')}")
        print(f"Database Faces: {data.get('database_faces', 'Unknown')}")
        print(f"Version: {data.get('version')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_stats_endpoint():
    """Test database statistics"""
    print("\n📊 Testing Stats Endpoint...")
    try:
        response = requests.get(f"{RAILWAY_URL}/stats", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Total Faces: {data['database_stats']['total_faces']}")
            print(f"Embedding Dimension: {data['database_stats']['embedding_dimension']}")
            print(f"Sample Faces: {len(data.get('sample_faces', []))}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Stats endpoint failed: {e}")
        return False

def test_face_search():
    """Test face search with a sample image"""
    print("\n🔍 Testing Face Search...")
    
    # Check if we have any test images
    test_images = list(Path("data/synthetic_faces").glob("*.jpg"))[:1]
    if not test_images:
        test_images = list(Path("data/lfw_faces").glob("*.jpg"))[:1]
    
    if not test_images:
        print("⚠️ No test images found, skipping face search test")
        return True
    
    test_image = test_images[0]
    print(f"Using test image: {test_image}")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            data = {'top_k': 5}
            
            response = requests.post(
                f"{RAILWAY_URL}/search", 
                files=files, 
                data=data,
                timeout=30  # Face search takes longer
            )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Results Found: {result['query']['results_found']}")
            
            for i, match in enumerate(result['results'][:3], 1):
                print(f"  {i}. {match['person_name']} (similarity: {match['similarity_score']})")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Face search failed: {e}")
        return False

def test_api_performance():
    """Test API response times"""
    print("\n⚡ Testing API Performance...")
    
    start_time = time.time()
    health_ok = test_health_check()
    health_time = time.time() - start_time
    
    print(f"Health check time: {health_time:.2f}s")
    
    if health_time > 2.0:
        print("⚠️ Health check slower than 2s (cold start expected)")
    else:
        print("✅ Health check within 2s requirement")
    
    return health_ok

def main():
    """Run all tests"""
    print("🚀 Testing Halo Face Search API on Railway")
    print(f"🌐 Target URL: {RAILWAY_URL}")
    print("=" * 50)
    
    # Update URL prompt
    if "your-project-name" in RAILWAY_URL:
        print("⚠️  IMPORTANT: Update RAILWAY_URL in this script with your actual Railway URL!")
        print("   Find it in Railway dashboard under your project")
        return
    
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint), 
        ("Stats Endpoint", test_stats_endpoint),
        ("Face Search", test_face_search),
        ("Performance", test_api_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 TEST SUMMARY:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is ready for production!")
    else:
        print("⚠️  Some tests failed. Check Railway logs for details.")

if __name__ == "__main__":
    main() 