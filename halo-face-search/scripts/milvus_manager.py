# file: milvus_manager.py
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import json
import numpy as np

# --- Connection and Configuration ---
MILVUS_HOST = "localhost"  # This will be the service name in Docker Compose
MILVUS_PORT = "19530"
COLLECTION_NAME = "face_recognition_db"
DIMENSION = 512  # Dimension of ArcFace embeddings
INDEX_TYPE = "HNSW"
METRIC_TYPE = "L2"  # Euclidean distance. Can also be "IP" for inner product.

def connect_to_milvus():
    """Establishes a connection to the Milvus server."""
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Successfully connected to Milvus.")

def create_milvus_collection():
    """Creates the Milvus collection with a defined schema if it doesn't exist."""
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return Collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    # Define the schema for our collection
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    schema = CollectionSchema(fields, description="Face Similarity Search Collection")
    
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print("Collection created successfully.")
    return collection

def insert_data_into_milvus(collection: Collection, data_records: list):
    """Inserts data records into the specified Milvus collection."""
    if not data_records:
        print("No data to insert.")
        return
        
    print(f"Inserting {len(data_records)} records into '{COLLECTION_NAME}'...")
    # Milvus expects data in columnar format (lists of values for each field)
    entities = [
        [record['id'] for record in data_records],
        [record['image_path'] for record in data_records],
        [record['embedding'] for record in data_records]
    ]
    
    insert_result = collection.insert(entities)
    collection.flush()  # Flushes data to disk
    print(f"Successfully inserted {insert_result.insert_count} records.")

def build_milvus_index(collection: Collection):
    """Builds an HNSW index on the 'embedding' field for fast searching."""
    if len(collection.indexes) > 0:
        print(f"Index already exists on collection '{COLLECTION_NAME}'.")
        return

    print(f"Creating '{INDEX_TYPE}' index for collection '{COLLECTION_NAME}'...")
    index_params = {
        "metric_type": METRIC_TYPE,
        "index_type": INDEX_TYPE,
        "params": {"M": 16, "efConstruction": 256}  # M: graph connectivity, efConstruction: search depth during build
    }
    
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created successfully.")

def search_similar_faces(collection: Collection, query_vector: list, top_k: int = 5) -> list:
    """
    Searches for the top_k most similar faces to the query_vector.
    
    Returns:
        A list of dictionaries, each containing the id, distance, and image_path of a match.
    """
    collection.load()  # Load collection into memory for searching
    
    search_params = {
        "metric_type": METRIC_TYPE,
        "params": {"ef": 128}  # ef: search depth during query, higher is more accurate but slower
    }
    
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "image_path"]  # Specify fields to return
    )
    
    hits = results[0]
    search_results = []
    for hit in hits:
        search_results.append({
            "id": hit.entity.get("id"),
            "distance": hit.distance,
            "image_path": hit.entity.get("image_path")
        })
        
    collection.release()  # Release collection from memory
    return search_results

if __name__ == '__main__':
    # --- Full Ingestion and Search Workflow ---
    connect_to_milvus()
    face_collection = create_milvus_collection()
    
    # Load data from the JSON file created earlier
    try:
        with open('data/face_database.json', 'r') as f:
            db_records = json.load(f)
    except FileNotFoundError:
        print("Error: face_database.json not found. Please run create_database_records.py first.")
        exit(1)
    
    # Check if data is already inserted to avoid duplicates
    if face_collection.num_entities == 0:
        insert_data_into_milvus(face_collection, db_records)
        build_milvus_index(face_collection)
    else:
        print(f"Data already present in collection ({face_collection.num_entities} entities). Skipping insertion and index building.")

    # Example search
    if db_records:
        print("\n--- Performing Example Search ---")
        sample_query_vector = db_records[0]['embedding']
        
        results = search_similar_faces(face_collection, sample_query_vector)
        
        print(f"Found {len(results)} similar faces for sample vector:")
        for result in results:
            print(f"  ID: {result['id']}, Path: {result['image_path']}, Distance: {result['distance']:.4f}") 