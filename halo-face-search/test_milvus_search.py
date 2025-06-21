#!/usr/bin/env python3
"""
Test Milvus search functionality to debug the API issue
"""

import os
from pymilvus import MilvusClient
import numpy as np

# Initialize the same way as your API
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "face_embeddings"

print(f"Connecting to Milvus at: {MILVUS_URI}")
milvus_client = MilvusClient(uri=MILVUS_URI)

try:
    # Check if collection exists
    if milvus_client.has_collection(COLLECTION_NAME):
        print(f"✅ Collection '{COLLECTION_NAME}' exists")
        
        # Get collection stats
        stats = milvus_client.get_collection_stats(COLLECTION_NAME)
        print(f"📊 Database contains {stats.get('row_count', 0)} records")
        
        if stats.get('row_count', 0) > 0:
            # Create a test embedding (same dimensions as your ArcFace embeddings)
            test_embedding = np.random.random(512).tolist()
            print("🔍 Testing search with random embedding...")
            
            # Try the search with minimal parameters
            try:
                search_results = milvus_client.search(
                    collection_name=COLLECTION_NAME,
                    data=[test_embedding],
                    limit=5
                )
                print(f"✅ Search successful! Found {len(search_results[0])} results")
                
                # Try with output fields
                search_results2 = milvus_client.search(
                    collection_name=COLLECTION_NAME,
                    data=[test_embedding],
                    limit=5,
                    output_fields=["face_id", "person_name"]
                )
                print(f"✅ Search with output fields successful!")
                
            except Exception as e:
                print(f"❌ Search failed: {e}")
                print(f"Error type: {type(e)}")
                
        else:
            print("⚠️  No data in collection to search")
            
    else:
        print(f"❌ Collection '{COLLECTION_NAME}' does not exist")
        
except Exception as e:
    print(f"❌ Connection failed: {e}") 