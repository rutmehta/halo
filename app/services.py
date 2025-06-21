# file: app/services.py
import os
import sys
from typing import List, Dict, Optional
from pymilvus import Collection
import numpy as np

# Add scripts directory to path to access our modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))

# Set environment variables before importing
os.environ['MILVUS_HOST'] = os.getenv('MILVUS_HOST', 'localhost')
os.environ['MILVUS_PORT'] = os.getenv('MILVUS_PORT', '19530')

from embedding_generator import get_face_embedding
from milvus_manager import (
    connect_to_milvus, 
    create_milvus_collection, 
    search_similar_faces
)

class FaceSearchService:
    """Service class for handling face search operations."""
    
    def __init__(self):
        self.collection: Optional[Collection] = None
        self._connect_to_milvus()
    
    def _connect_to_milvus(self):
        """Initialize connection to Milvus and get collection."""
        try:
            connect_to_milvus()
            self.collection = create_milvus_collection()
            
            # Load collection into memory for searching
            if self.collection.num_entities > 0:
                self.collection.load()
                print(f"Loaded collection with {self.collection.num_entities} faces.")
            else:
                print("Warning: Collection is empty. Please run data ingestion first.")
        except Exception as e:
            print(f"Error connecting to Milvus: {e}")
            self.collection = None
    
    def find_similar_faces(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Find the most similar faces to the given image.
        
        Args:
            image_path: Path to the query image
            top_k: Number of similar faces to return
            
        Returns:
            List of dictionaries containing similar faces
        """
        # Generate embedding for the query image
        query_embedding = get_face_embedding(image_path)
        
        if query_embedding is None:
            return []
        
        if self.collection is None:
            raise RuntimeError("Milvus collection not available")
        
        # Search for similar faces
        try:
            results = search_similar_faces(
                self.collection, 
                query_embedding.tolist(), 
                top_k=top_k
            )
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def close(self):
        """Release resources."""
        if self.collection is not None:
            self.collection.release()

# Singleton instance
_face_search_service: Optional[FaceSearchService] = None

def get_face_search_service() -> FaceSearchService:
    """Get or create the face search service instance."""
    global _face_search_service
    if _face_search_service is None:
        _face_search_service = FaceSearchService()
    return _face_search_service 