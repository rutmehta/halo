#!/usr/bin/env python3
"""
Generate face embeddings from LFW dataset and store in Milvus
This populates the face search database with real face data
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from deepface import DeepFace
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class FaceEmbeddingGenerator:
    """Generate and store face embeddings for the Halo face search system"""
    
    def __init__(self, milvus_uri="http://localhost:19530"):
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = "face_embeddings"
        self.dimension = 512  # ArcFace embedding dimension
        
        print("🚀 Initializing Halo Face Embedding Generator")
        print(f"🔗 Connecting to Milvus: {milvus_uri}")
        
    def extract_face_embedding(self, image_path: str) -> tuple[np.ndarray, bool]:
        """
        Extract face embedding using ArcFace model
        Returns (embedding, success)
        """
        try:
            # Generate embedding with ArcFace
            embedding_obj = DeepFace.represent(
                img_path=image_path,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False,
                align=True,
                normalization="base"
            )
            
            # Extract and normalize embedding
            embedding = np.array(embedding_obj[0]["embedding"])
            embedding = embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity
            
            return embedding, True
            
        except Exception as e:
            print(f"⚠️  Failed to process {os.path.basename(image_path)}: {str(e)}")
            return np.zeros(self.dimension), False
    
    def ensure_collection_exists(self):
        """Ensure the Milvus collection exists and is properly configured"""
        
        if not self.milvus_client.has_collection(self.collection_name):
            print(f"📦 Creating collection: {self.collection_name}")
            
            # Create schema
            schema = CollectionSchema([
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="face_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="person_name", dtype=DataType.VARCHAR, max_length=200)
            ])
            
            # Create collection with optimized index
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params={
                    "field_name": "embedding",
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200}
                }
            )
            print(f"✅ Created collection with HNSW index")
        else:
            print(f"✅ Collection {self.collection_name} already exists")
        
        # Load collection into memory
        self.milvus_client.load_collection(self.collection_name)
        print(f"⚡ Collection loaded into memory for fast search")
    
    def get_existing_faces(self) -> set:
        """Get set of face_ids already in database to avoid duplicates"""
        try:
            existing = self.milvus_client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["face_id"],
                limit=50000  # Get all existing face_ids
            )
            return {item['face_id'] for item in existing}
        except:
            return set()
    
    def process_face_dataset(self, faces_dir: str, batch_size: int = 50, max_faces: int = None):
        """
        Process face dataset and generate embeddings
        
        Args:
            faces_dir: Directory containing face images
            batch_size: Number of faces to process in each batch
            max_faces: Maximum number of faces to process (None for all)
        """
        
        print(f"\n🎯 Processing faces from: {faces_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in os.listdir(faces_dir)
            if Path(f).suffix.lower() in image_extensions
        ]
        
        if max_faces:
            image_files = image_files[:max_faces]
        
        print(f"📸 Found {len(image_files)} face images to process")
        
        # Check for existing faces
        existing_faces = self.get_existing_faces()
        print(f"🔄 {len(existing_faces)} faces already in database")
        
        # Filter out already processed faces
        new_faces = []
        for img_file in image_files:
            face_id = Path(img_file).stem  # filename without extension
            if face_id not in existing_faces:
                new_faces.append(img_file)
        
        print(f"🆕 {len(new_faces)} new faces to process")
        
        if not new_faces:
            print("✅ All faces already processed!")
            return
        
        # Process faces in batches
        successful_embeddings = 0
        failed_embeddings = 0
        batch_data = []
        
        with tqdm(total=len(new_faces), desc="Generating embeddings") as pbar:
            for i, img_file in enumerate(new_faces):
                image_path = os.path.join(faces_dir, img_file)
                
                # Extract person name and face_id from filename
                # Format: PersonName_imageX.jpg -> PersonName, PersonName_imageX
                face_id = Path(img_file).stem
                person_name = face_id.split('_')[0] if '_' in face_id else face_id
                
                # Generate embedding
                embedding, success = self.extract_face_embedding(image_path)
                
                if success:
                    # Add to batch
                    batch_data.append({
                        "face_id": face_id,
                        "embedding": embedding.tolist(),
                        "image_path": f"lfw_faces/{img_file}",
                        "person_name": person_name
                    })
                    successful_embeddings += 1
                else:
                    failed_embeddings += 1
                
                # Insert batch when full or at end
                if len(batch_data) >= batch_size or i == len(new_faces) - 1:
                    if batch_data:
                        try:
                            self.milvus_client.insert(
                                collection_name=self.collection_name,
                                data=batch_data
                            )
                            pbar.set_postfix({
                                'Success': successful_embeddings,
                                'Failed': failed_embeddings,
                                'Batch': len(batch_data)
                            })
                            batch_data = []
                        except Exception as e:
                            print(f"\n❌ Batch insert failed: {e}")
                
                pbar.update(1)
        
        print(f"\n🎯 Embedding generation completed!")
        print(f"✅ Successfully processed: {successful_embeddings} faces")
        print(f"❌ Failed to process: {failed_embeddings} faces")
        print(f"📊 Success rate: {successful_embeddings/(successful_embeddings+failed_embeddings)*100:.1f}%")
        
        # Get final database stats
        stats = self.milvus_client.get_collection_stats(self.collection_name)
        total_faces = stats.get('row_count', 0)
        print(f"💾 Total faces in database: {total_faces}")
        
        return successful_embeddings, failed_embeddings
    
    def verify_database(self):
        """Verify the database is working correctly"""
        
        print("\n🔍 Verifying face search database...")
        
        try:
            # Get database stats
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            total_faces = stats.get('row_count', 0)
            
            # Get sample records
            sample_records = self.milvus_client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["face_id", "person_name"],
                limit=10
            )
            
            print(f"📊 Database verification:")
            print(f"   Total faces: {total_faces}")
            print(f"   Sample people: {', '.join(set([r['person_name'] for r in sample_records[:5]]))}")
            
            # Test search functionality
            if total_faces > 0 and sample_records:
                test_embedding = np.random.rand(self.dimension).tolist()
                search_results = self.milvus_client.search(
                    collection_name=self.collection_name,
                    data=[test_embedding],
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"ef": 64}},
                    limit=5,
                    output_fields=["face_id", "person_name"]
                )
                
                if search_results and len(search_results[0]) > 0:
                    print(f"🔍 Search test: Found {len(search_results[0])} similar faces")
                    print("✅ Database is ready for face search!")
                else:
                    print("⚠️  Search test returned no results")
            
        except Exception as e:
            print(f"❌ Database verification failed: {e}")

def main():
    """Main function to populate the Halo face search database"""
    
    print("🔥 HALO FACE SEARCH - Database Population")
    print("=" * 50)
    
    # Initialize embedding generator
    generator = FaceEmbeddingGenerator()
    
    # Ensure collection exists
    generator.ensure_collection_exists()
    
    # Look for face datasets
    possible_dirs = [
        'data/sample_faces',
        'data/lfw_faces', 
        'data/real_faces',
        'data/synthetic_faces'
    ]
    
    faces_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path) and os.listdir(dir_path):
            faces_dir = dir_path
            break
    
    if not faces_dir:
        print("❌ No face dataset found!")
        print("Please run: python scripts/download_lfw_dataset.py first")
        return
    
    print(f"📂 Using face dataset: {faces_dir}")
    
    # Process faces (limit to 1000 for demo)
    generator.process_face_dataset(
        faces_dir=faces_dir,
        batch_size=50,
        max_faces=1000  # Halo requirement: 1000 unique faces
    )
    
    # Verify database
    generator.verify_database()
    
    print("\n🎯 HALO FACE SEARCH DATABASE READY!")
    print("🚀 You can now test the API:")
    print("   - GET  http://localhost:8000/ (health check)")
    print("   - POST http://localhost:8000/search (upload image to search)")
    print("   - GET  http://localhost:8000/docs (interactive API docs)")

if __name__ == '__main__':
    main() 