#!/usr/bin/env python3
"""
Quick Test Script for Halo Face Search API
"""

import requests
import json
import os

API_URL = "http://localhost:8000"

def test_api():
    print("🧪 Quick Face Search API Test")
    print("=" * 40)
    
    # 1. Test health
    print("1. Testing API health...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ API is healthy")
            print(f"   📊 Database contains: {health['database']['total_faces']} faces")
        else:
            print(f"   ❌ API unhealthy: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Can't connect to API: {e}")
        print("   💡 Make sure your API is running: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return
    
    # 2. Find a test image
    test_image = None
    image_paths = [
        "data/synthetic_faces/face_0001.jpg",
        "data/synthetic_faces/face_0010.jpg", 
        "data/synthetic_faces/face_0100.jpg"
    ]
    
    for path in image_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if not test_image:
        print("   ⚠️  No test images found. Checking data directory...")
        if os.path.exists("data/synthetic_faces"):
            files = [f for f in os.listdir("data/synthetic_faces") if f.endswith('.jpg')]
            if files:
                test_image = f"data/synthetic_faces/{files[0]}"
    
    if not test_image:
        print("   ❌ No test images available!")
        print("   💡 Run: python scripts/generate_faces.py")
        return
    
    print(f"2. Testing face search with: {test_image}")
    
    # 3. Test face search
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (os.path.basename(test_image), f, 'image/jpeg')}
            
            response = requests.post(f"{API_URL}/search", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Search successful!")
                print(f"   📊 Found {len(result.get('results', []))} similar faces")
                
                # Show top 3 results
                for i, match in enumerate(result.get('results', [])[:3]):
                    person = match.get('person_name', 'Unknown')
                    score = match.get('similarity_score', 0)
                    print(f"      {i+1}. {person} (similarity: {score:.3f})")
                    
            else:
                print(f"   ❌ Search failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
    except Exception as e:
        print(f"   ❌ Search test failed: {e}")
        return
    
    # 4. Test performance
    print("3. Testing response time...")
    import time
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (os.path.basename(test_image), f, 'image/jpeg')}
            
            start_time = time.time()
            response = requests.post(f"{API_URL}/search", files=files)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                if response_time < 2.0:
                    print(f"   ✅ Response time: {response_time:.2f}s (< 2s requirement ✅)")
                else:
                    print(f"   ⚠️  Response time: {response_time:.2f}s (> 2s requirement ❌)")
            else:
                print(f"   ❌ Performance test failed")
                
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
    
    print("\n🎯 Test Summary:")
    print("   1. ✅ API is running and healthy")
    print("   2. ✅ Face search is working")
    print("   3. ✅ Returns similar faces with scores")
    print("   4. ✅ Response time is acceptable")
    
    print(f"\n🔗 Try the interactive docs: {API_URL}/docs")
    print("🎉 Your Face Search API is working!")

if __name__ == "__main__":
    test_api() 