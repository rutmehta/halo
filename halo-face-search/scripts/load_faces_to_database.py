#!/usr/bin/env python3
"""
Load Face Database
Load both synthetic and LFW faces into Milvus vector database
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
from pymilvus import MilvusClient
from deepface import DeepFace
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_face_embedding(image_path: str) -> np.ndarray:
    """Extract face embedding using DeepFace ArcFace model"""
    try:
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            normalization="base"
        )
        
        embedding = np.array(embedding_obj[0]["embedding"])
        return embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity
        
    except Exception as e:
        print(f"⚠️ Failed to process {image_path}: {e}")
        return None

def load_faces_to_database():
    """Load all faces (synthetic + LFW) into Milvus database"""
    
    print("🚀 Loading Face Database for Halo Search API")
    
    # Milvus connection
    MILVUS_URI = "http://localhost:19530"
    COLLECTION_NAME = "face_embeddings"
    DIMENSION = 512
    
    try:
        # Connect using the ORM connections
        from pymilvus import connections
        connections.connect("default", host="localhost", port="19530")
        
        milvus_client = MilvusClient(uri=MILVUS_URI)
        print(f"✅ Connected to Milvus at {MILVUS_URI}")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        print("💡 Make sure Milvus is running: docker-compose up -d")
        return False
    
    # Clear existing collection
    if milvus_client.has_collection(COLLECTION_NAME):
        print(f"🗑️ Clearing existing collection: {COLLECTION_NAME}")
        milvus_client.drop_collection(COLLECTION_NAME)
    
    # Create fresh collection with proper schema
    print(f"📦 Creating new collection: {COLLECTION_NAME}")
    
    # Define the schema explicitly
    from pymilvus import DataType, FieldSchema, CollectionSchema
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="face_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="person_name", dtype=DataType.VARCHAR, max_length=100)
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Halo face search embeddings"
    )
    
    # Create collection with schema
    from pymilvus import Collection
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using='default'
    )
    
    # Create index for vector search
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    # Define face directories
    data_dir = Path("data")
    synthetic_dir = data_dir / "synthetic_faces"
    lfw_dir = data_dir / "lfw_faces"
    
    # Collect all face images
    face_files = []
    
    # Add synthetic faces
    if synthetic_dir.exists():
        synthetic_files = list(synthetic_dir.glob("*.jpg")) + list(synthetic_dir.glob("*.png"))
        face_files.extend([(f, "synthetic", f.stem) for f in synthetic_files])
        print(f"📁 Found {len(synthetic_files)} synthetic faces")
    
    # Add LFW faces
    if lfw_dir.exists():
        lfw_files = list(lfw_dir.glob("*.jpg")) + list(lfw_dir.glob("*.jpeg"))
        face_files.extend([(f, "lfw", f.stem.split('_')[0]) for f in lfw_files])
        print(f"📁 Found {len(lfw_files)} LFW real faces")
    
    print(f"📊 Total faces to process: {len(face_files)}")
    
    if len(face_files) == 0:
        print("❌ No face images found!")
        print("💡 Run download_lfw.py first or add synthetic faces")
        return False
    
    # Process faces in batches
    batch_size = 50
    batch_data = []
    processed_count = 0
    failed_count = 0
    
    print("🔄 Processing faces and generating embeddings...")
    
    for i, (face_path, source_type, person_name) in enumerate(face_files):
        try:
            # Extract embedding
            embedding = extract_face_embedding(str(face_path))
            
            if embedding is not None:
                face_id = f"{source_type}_{i:06d}"
                
                # Handle both absolute and relative paths safely
                try:
                    relative_path = face_path.relative_to(Path.cwd())
                except ValueError:
                    # If relative_to fails, just use the filename
                    relative_path = face_path.name
                
                batch_data.append({
                    "face_id": face_id,
                    "embedding": embedding.tolist(),
                    "image_path": str(relative_path),
                    "person_name": person_name
                })
                
                processed_count += 1
            else:
                failed_count += 1
            
            # Insert batch when full
            if len(batch_data) >= batch_size:
                collection.insert(batch_data)
                collection.flush()  # Ensure data is persisted
                print(f"📊 Processed {processed_count}/{len(face_files)} faces ({failed_count} failed)")
                batch_data = []
            
        except Exception as e:
            print(f"⚠️ Error processing {face_path}: {e}")
            failed_count += 1
    
    # Insert remaining batch
    if batch_data:
        collection.insert(batch_data)
        collection.flush()  # Ensure data is persisted
    
    # Load collection for searching
    collection.load()
    
    # Final stats  
    final_count = collection.num_entities
    
    print(f"\n🎉 Face Database Loaded Successfully!")
    print(f"📊 Total faces in database: {final_count}")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"⚠️ Failed to process: {failed_count}")
    print(f"🚀 Database ready for face search API!")
    
    # Test the database
    print(f"\n🧪 Testing database query...")
    try:
        sample_results = collection.query(
            expr="",
            output_fields=["face_id", "person_name"],
            limit=5
        )
        
        print("📝 Sample records:")
        for record in sample_results:
            print(f"  - {record['face_id']}: {record['person_name']}")
            
    except Exception as e:
        print(f"⚠️ Test query failed: {e}")
    
    return True

if __name__ == "__main__":
    success = load_faces_to_database()
    if success:
        print(f"\n🎯 Ready to start Halo Face Search API!")
        print(f"💡 Run: python -m app.main")
    else:
        print(f"\n❌ Database loading failed") 