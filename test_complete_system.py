#!/usr/bin/env python3
"""
Complete System Test for Halo Face Search API
This tests all functionality required by the Halo take-home project
"""

import requests
import json
import time
import os
from pathlib import Path

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

def test_api_health():
    """Test API health and basic functionality"""
    print("🏥 Testing API Health...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ API Status: {data.get('status')}")
        print(f"📊 Database Faces: {data.get('database_faces', 0)}")
        print(f"🔧 Capabilities: {data.get('capabilities', {}).get('face_recognition')}")
        
        return True
    except Exception as e:
        print(f"❌ API Health Check Failed: {e}")
        return False

def test_health_endpoint():
    """Test detailed health endpoint"""
    print("\n🔍 Testing Health Endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Health Status: {data.get('status')}")
        print(f"🔗 Milvus Connected: {data.get('services', {}).get('milvus_connected')}")
        print(f"💾 Total Faces: {data.get('database', {}).get('total_faces', 0)}")
        
        return True
    except Exception as e:
        print(f"❌ Health Endpoint Failed: {e}")
        return False

def test_stats_endpoint():
    """Test statistics endpoint"""
    print("\n📊 Testing Stats Endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats Retrieved:")
            print(f"   Total Faces: {data.get('database_stats', {}).get('total_faces', 0)}")
            print(f"   Embedding Dimension: {data.get('database_stats', {}).get('embedding_dimension')}")
            print(f"   Face Model: {data.get('api_info', {}).get('face_model')}")
            return True
        else:
            print(f"⚠️  Stats endpoint returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats Endpoint Failed: {e}")
        return False

def test_add_face(image_path=None, person_name="Test Person"):
    """Test adding a face to the database"""
    print(f"\n➕ Testing Add Face...")
    
    # Create a simple test image if none provided
    if not image_path or not os.path.exists(image_path):
        print("⚠️  No test image provided, skipping add face test")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {'person_name': person_name}
            
            response = requests.post(f"{API_BASE_URL}/add_face", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Face Added:")
            print(f"   Face ID: {result.get('face_id')}")
            print(f"   Person: {result.get('person_name')}")
            
            return result.get('face_id')
    except Exception as e:
        print(f"❌ Add Face Failed: {e}")
        return False

def test_search_faces(image_path=None, top_k=5):
    """Test face search functionality"""
    print(f"\n🔍 Testing Face Search (top {top_k})...")
    
    if not image_path or not os.path.exists(image_path):
        print("⚠️  No test image provided, skipping search test")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            params = {'top_k': top_k}
            
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/search", files=files, params=params)
            response.raise_for_status()
            search_time = time.time() - start_time
            
            result = response.json()
            
            print(f"✅ Search Completed in {search_time:.2f}s")
            print(f"   Results Found: {result.get('query', {}).get('results_found', 0)}")
            
            # Show top results
            for i, match in enumerate(result.get('results', [])[:3]):
                print(f"   {i+1}. {match.get('person_name')} (similarity: {match.get('similarity_score', 0):.3f})")
            
            # Check performance requirement (< 2 seconds)
            if search_time < 2.0:
                print(f"🚀 Performance: PASSED (< 2s)")
            else:
                print(f"⚠️  Performance: SLOW ({search_time:.2f}s > 2s)")
            
            return len(result.get('results', []))
    except Exception as e:
        print(f"❌ Face Search Failed: {e}")
        return False

def test_performance(image_path=None, num_requests=10):
    """Test API performance requirements (20 RPS, <2s latency)"""
    print(f"\n🚀 Testing Performance ({num_requests} requests)...")
    
    if not image_path or not os.path.exists(image_path):
        print("⚠️  No test image provided, skipping performance test")
        return
    
    successful_requests = 0
    total_time = 0
    latencies = []
    
    print("Running requests...")
    overall_start = time.time()
    
    for i in range(num_requests):
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (f'test_{i}.jpg', f, 'image/jpeg')}
                
                start_time = time.time()
                response = requests.post(f"{API_BASE_URL}/search", files=files, timeout=5)
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    successful_requests += 1
                    latencies.append(latency)
                
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    overall_time = time.time() - overall_start
    
    if successful_requests > 0:
        avg_latency = sum(latencies) / len(latencies)
        rps = successful_requests / overall_time
        
        print(f"\n📊 Performance Results:")
        print(f"   Successful Requests: {successful_requests}/{num_requests}")
        print(f"   Average Latency: {avg_latency:.3f}s")
        print(f"   Requests Per Second: {rps:.1f}")
        
        # Check Halo requirements
        latency_ok = avg_latency < 2.0
        rps_ok = rps >= 20
        
        print(f"\n🎯 Halo Requirements Check:")
        print(f"   Latency < 2s: {'✅ PASS' if latency_ok else '❌ FAIL'} ({avg_latency:.3f}s)")
        print(f"   RPS >= 20: {'✅ PASS' if rps_ok else '❌ FAIL'} ({rps:.1f})")
        
        return latency_ok and rps_ok
    else:
        print("❌ No successful requests")
        return False

def find_test_image():
    """Find a test image in the data directories"""
    possible_paths = [
        "data/synthetic_faces",
        "data/sample_faces", 
        "data/lfw_faces",
        "."
    ]
    
    for dir_path in possible_paths:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(dir_path, file)
    
    return None

def main():
    """Run complete system tests for Halo Face Search"""
    
    print("🔥 HALO FACE SEARCH - COMPLETE SYSTEM TEST")
    print("=" * 50)
    print(f"🎯 Testing API: {API_BASE_URL}")
    
    # Find test image
    test_image = find_test_image()
    if test_image:
        print(f"📸 Using test image: {test_image}")
    else:
        print("⚠️  No test images found")
    
    results = {}
    
    # Test 1: API Health
    results['health'] = test_api_health()
    
    # Test 2: Health Endpoint
    results['health_endpoint'] = test_health_endpoint()
    
    # Test 3: Stats Endpoint
    results['stats'] = test_stats_endpoint()
    
    # Test 4: Add Face (if image available)
    if test_image:
        results['add_face'] = test_add_face(test_image, "Test Person") is not False
    
    # Test 5: Search Faces (if image available)
    if test_image:
        results['search'] = test_search_faces(test_image, top_k=5) is not False
    
    # Test 6: Performance Test (if image available)
    if test_image:
        results['performance'] = test_performance(test_image, num_requests=5)  # Reduced for testing
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 HALO REQUIREMENTS SUMMARY")
    print("=" * 50)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED! Ready for Halo team review!")
        print(f"🔗 API URL for Halo team: {API_BASE_URL}")
        print("📚 API Documentation: {}/docs".format(API_BASE_URL))
    else:
        print("\n⚠️  Some tests failed. Please review and fix issues.")
    
    return all(results.values())

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1) 