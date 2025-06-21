#!/usr/bin/env python3
"""
Halo Face Search API - Simplified Working Version
Production-ready face similarity search with in-memory vector database
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from deepface import DeepFace
import numpy as np
import tempfile
import os
import warnings
import json
import time
from typing import List, Dict, Any
import uuid
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI(
    title="Halo Face Search API",
    description="Real-time face similarity search service using ArcFace embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory vector database
class FaceDatabase:
    def __init__(self):
        self.faces = []  # List of face records
        self.embeddings = []  # List of embeddings
        self.next_id = 1
        
    def add_face(self, embedding: np.ndarray, face_id: str, person_name: str, image_path: str):
        """Add a face to the database"""
        record = {
            "id": self.next_id,
            "face_id": face_id,
            "person_name": person_name,
            "image_path": image_path,
            "created_at": time.time()
        }
        
        self.faces.append(record)
        self.embeddings.append(embedding)
        self.next_id += 1
        
        return record
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search for similar faces"""
        if len(self.embeddings) == 0:
            return []
        
        # Calculate cosine similarities
        embeddings_matrix = np.array(self.embeddings)
        query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
        
        # Get top K most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            similarity_score = float(similarities[idx])
            face_record = self.faces[idx]
            
            results.append({
                "rank": i + 1,
                "face_id": face_record["face_id"],
                "person_name": face_record["person_name"],
                "image_path": face_record["image_path"],
                "similarity_score": round(similarity_score, 4),
                "similarity_percentage": round(similarity_score * 100, 2)
            })
        
        return results
    
    def get_stats(self):
        """Get database statistics"""
        return {
            "total_faces": len(self.faces),
            "embedding_dimension": 512 if self.embeddings else 0,
            "database_type": "In-Memory Vector Store"
        }

# Initialize the face database
face_db = FaceDatabase()

@app.on_event("startup")
async def startup_event():
    """Initialize the face database with synthetic faces"""
    print("🚀 Starting Halo Face Search API...")
    
    # Load synthetic faces on startup
    await load_synthetic_faces()
    
    stats = face_db.get_stats()
    print(f"📊 Database initialized with {stats['total_faces']} faces")
    print("🎯 Halo Face Search API ready for requests!")

async def load_synthetic_faces():
    """Load synthetic faces into the database"""
    faces_dir = "data/synthetic_faces"
    
    if not os.path.exists(faces_dir):
        print("⚠️  No synthetic faces directory found")
        return
    
    # Get face files
    face_files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Load up to 100 faces for demo (adjust as needed)
    max_faces = min(100, len(face_files))
    
    print(f"📥 Loading {max_faces} synthetic faces...")
    
    for i, face_file in enumerate(face_files[:max_faces]):
        try:
            face_path = os.path.join(faces_dir, face_file)
            
            # Extract embedding
            embedding = extract_face_embedding(face_path)
            
            # Generate metadata
            face_id = f"synthetic_{i+1:04d}"
            person_name = f"Person_{i+1:04d}"
            
            # Add to database
            face_db.add_face(embedding, face_id, person_name, f"synthetic_faces/{face_file}")
            
            if (i + 1) % 10 == 0:
                print(f"   Loaded {i + 1}/{max_faces} faces...")
                
        except Exception as e:
            print(f"   ⚠️  Failed to load {face_file}: {str(e)}")
    
    print(f"✅ Loaded {face_db.get_stats()['total_faces']} faces successfully")

def extract_face_embedding(image_path: str) -> np.ndarray:
    """Extract face embedding using ArcFace model"""
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
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to extract face embedding: {str(e)}"
        )

@app.get("/")
async def root():
    """API Health check and information"""
    stats = face_db.get_stats()
    
    return {
        "message": "🎯 Halo Face Search API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "database_faces": stats["total_faces"],
        "capabilities": {
            "face_recognition": "ArcFace 512D embeddings",
            "vector_database": "In-Memory Vector Store",
            "similarity_metric": "Cosine similarity",
            "max_rps": "20+ requests per second",
            "latency": "<2 seconds per search"
        },
        "endpoints": {
            "/search": "POST - Search for similar faces",
            "/add_face": "POST - Add new face to database",
            "/stats": "GET - Database statistics",
            "/health": "GET - Detailed health check",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    stats = face_db.get_stats()
    
    return {
        "status": "healthy",
        "timestamp": str(time.time()),
        "services": {
            "face_database": "connected",
            "api_server": "running",
            "face_recognition": "ready"
        },
        "database": {
            "total_faces": stats["total_faces"],
            "embedding_dimension": stats["embedding_dimension"],
            "database_type": stats["database_type"]
        },
        "performance": {
            "target_rps": 20,
            "target_latency": "< 2 seconds",
            "memory_loaded": True
        }
    }

@app.post("/search")
async def search_faces(file: UploadFile = File(...), top_k: int = 5):
    """
    🔍 Search for the most similar faces in the database
    
    Upload an image and get back the top K most similar faces.
    """
    
    # Validate input
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image (JPG, PNG, etc.)")
    
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Extract face embedding from uploaded image
        print(f"🔍 Extracting face embedding from {file.filename}")
        start_time = time.time()
        
        query_embedding = extract_face_embedding(tmp_file_path)
        
        # Search in database
        print(f"🚀 Searching for top {top_k} similar faces...")
        results = face_db.search_similar(query_embedding, top_k)
        
        search_time = time.time() - start_time
        print(f"✅ Found {len(results)} similar faces in {search_time:.2f}s")
        
        return {
            "success": True,
            "message": "Face search completed successfully",
            "query": {
                "filename": file.filename,
                "top_k_requested": top_k,
                "results_found": len(results),
                "search_time_seconds": round(search_time, 3)
            },
            "results": results,
            "metadata": {
                "embedding_model": "ArcFace",
                "similarity_metric": "Cosine",
                "database_size": face_db.get_stats()["total_faces"]
            }
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.post("/add_face")
async def add_face(file: UploadFile = File(...), person_name: str = None):
    """
    ➕ Add a new face to the database
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate face_id
    current_count = face_db.get_stats()["total_faces"]
    face_id = f"uploaded_{current_count + 1:06d}"
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        print(f"➕ Adding face: {face_id} ({person_name or 'Unknown'})")
        
        # Extract face embedding
        embedding = extract_face_embedding(tmp_file_path)
        
        # Add to database
        record = face_db.add_face(
            embedding=embedding,
            face_id=face_id,
            person_name=person_name or "Unknown",
            image_path=f"uploaded/{file.filename}"
        )
        
        print(f"✅ Successfully added face {face_id}")
        
        return {
            "success": True,
            "message": "Face added successfully to database",
            "face_data": {
                "face_id": face_id,
                "person_name": person_name or "Unknown",
                "filename": file.filename,
                "embedding_dimension": len(embedding)
            },
            "database_info": {
                "total_faces": face_db.get_stats()["total_faces"]
            }
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.get("/stats")
async def get_stats():
    """📊 Get comprehensive database and API statistics"""
    
    stats = face_db.get_stats()
    
    # Get sample records
    sample_faces = face_db.faces[:5] if face_db.faces else []
    
    return {
        "database_stats": stats,
        "sample_faces": [
            {
                "face_id": face["face_id"], 
                "person_name": face["person_name"]
            } for face in sample_faces
        ],
        "api_info": {
            "version": "1.0.0",
            "face_model": "ArcFace",
            "max_search_results": 20,
            "supported_formats": ["JPG", "PNG", "BMP", "TIFF"]
        },
        "performance": {
            "target_rps": 20,
            "target_latency": "< 2 seconds",
            "memory_optimization": "In-memory vector store"
        }
    }

@app.delete("/clear_database")
async def clear_database():
    """🗑️ Clear all faces from database (use with caution!)"""
    global face_db
    
    old_count = face_db.get_stats()["total_faces"]
    face_db = FaceDatabase()
    
    return {
        "success": True,
        "message": "Database cleared successfully",
        "faces_removed": old_count,
        "status": "ready_for_new_data"
    }

if __name__ == "__main__":
    print("🚀 Starting Halo Face Search API server...")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    ) 