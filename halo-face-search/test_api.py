#!/usr/bin/env python3
"""
Simple test script for the Facial Similarity Search API
"""

import requests
import json
import sys

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_search(image_path):
    """Test the search endpoint with an image"""
    print(f"\nTesting search endpoint with image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image_file': (image_path, f, 'image/jpeg')}
        response = requests.post(f"{API_BASE_URL}/search", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Face found: {result['query_face_found']}")
        
        if result['query_face_found']:
            print(f"Number of matches: {len(result['top_matches'])}")
            for i, match in enumerate(result['top_matches']):
                print(f"  {i+1}. {match['image_url']} (score: {match['similarity_score']:.4f})")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

if __name__ == "__main__":
    print("=== Facial Similarity Search API Test ===\n")
    
    # Test health endpoint
    if not test_health():
        print("Health check failed. Is the API running?")
        sys.exit(1)
    
    # Test search endpoint with a sample face
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "data/synthetic_faces/face_0000.jpg"
        print(f"No image specified, using default: {test_image}")
    
    test_search(test_image)
    
    print("\n=== Test Complete ===") 