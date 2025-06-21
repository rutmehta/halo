#!/usr/bin/env python3
"""
Halo Face Search API - Complete Implementation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from deepface import DeepFace
import numpy as np
import tempfile
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI(
    title="Halo Face Search API",
    description="Real-time face similarity search service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
milvus_client = MilvusClient(uri=MILVUS_URI)
COLLECTION_NAME = "face_embeddings"
DIMENSION = 512

@app.on_event("startup")
async def startup_event():
    try:
        print("🚀 Starting Halo Face Search API...")
        if not milvus_client.has_collection(COLLECTION_NAME):
            schema = CollectionSchema([
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="face_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="person_name", dtype=DataType.VARCHAR, max_length=200)
            ])
            
            milvus_client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema,
                index_params={
                    "field_name": "embedding",
                    "index_type": "HNSW", 
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200}
                }
            )
            print(f"✅ Created collection: {COLLECTION_NAME}")
        
        milvus_client.load_collection(COLLECTION_NAME)
        stats = milvus_client.get_collection_stats(COLLECTION_NAME)
        print(f"📊 Database contains {stats.get('row_count', 0)} face embeddings")
        print("�� Halo Face Search API ready!")
        
    except Exception as e:
        print(f"❌ Error initializing: {e}")

def extract_face_embedding(image_path: str) -> np.ndarray:
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
        return embedding / np.linalg.norm(embedding)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face embedding failed: {str(e)}")

@app.get("/")
async def root():
    try:
        stats = milvus_client.get_collection_stats(COLLECTION_NAME)
        return {
            "message": "🎯 Halo Face Search API is running!",
            "status": "healthy",
            "version": "1.0.0",
            "database_faces": stats.get('row_count', 0),
            "capabilities": {
                "face_recognition": "ArcFace 512D embeddings",
                "vector_database": "Milvus with HNSW indexing",
                "similarity_metric": "Cosine similarity"
            },
            "endpoints": {
                "/search": "POST - Search for similar faces",
                "/add_face": "POST - Add new face to database",
                "/stats": "GET - Database statistics", 
                "/docs": "GET - API documentation"
            }
        }
    except:
        return {"message": "🎯 Halo Face Search API", "status": "healthy", "database": "connecting"}

@app.get("/health")
async def health_check():
    try:
        stats = milvus_client.get_collection_stats(COLLECTION_NAME)
        return {
            "status": "healthy",
            "services": {"milvus_connected": True, "api_server": "running"},
            "database": {"total_faces": stats.get('row_count', 0), "embedding_dimension": DIMENSION}
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

@app.post("/search") 
async def search_faces(file: UploadFile = File(...), top_k: int = 5):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        query_embedding = extract_face_embedding(tmp_file_path)
        
        search_results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding.tolist()],
            anns_field="embedding", 
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["face_id", "image_path", "person_name"]
        )
        
        results = []
        if search_results and len(search_results[0]) > 0:
            for hit in search_results[0]:
                results.append({
                    "rank": len(results) + 1,
                    "face_id": hit.entity.get("face_id"),
                    "person_name": hit.entity.get("person_name", "Unknown"),
                    "similarity_score": round(float(hit.score), 4)
                })
        
        return {
            "success": True,
            "message": "Face search completed",
            "query": {"filename": file.filename, "results_found": len(results)},
            "results": results
        }
        
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.post("/add_face")
async def add_face(file: UploadFile = File(...), person_name: str = None):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    current_count = milvus_client.get_collection_stats(COLLECTION_NAME).get('row_count', 0)
    face_id = f"face_{current_count + 1:06d}"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        embedding = extract_face_embedding(tmp_file_path)
        
        data = [{
            "face_id": face_id,
            "embedding": embedding.tolist(),
            "image_path": f"uploaded/{file.filename}",
            "person_name": person_name or "Unknown"
        }]
        
        result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
        
        return {
            "success": True,
            "message": "Face added successfully",
            "face_id": face_id,
            "person_name": person_name or "Unknown"
        }
        
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.get("/stats")
async def get_stats():
    try:
        stats = milvus_client.get_collection_stats(COLLECTION_NAME)
        sample_records = milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter="",
            output_fields=["face_id", "person_name"],
            limit=5
        )
        
        return {
            "database_stats": {
                "total_faces": stats.get('row_count', 0),
                "embedding_dimension": DIMENSION
            },
            "sample_faces": sample_records,
            "api_info": {"version": "1.0.0", "face_model": "ArcFace"}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
