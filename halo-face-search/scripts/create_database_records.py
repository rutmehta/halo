# file: create_database_records.py
import os
import sys
import uuid
import json

# Add the scripts directory to Python path to import embedding_generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from embedding_generator import get_face_embedding

def process_image_directory(image_dir: str, output_file: str):
    """
    Processes all images in a directory to generate face embeddings and saves them
    as a structured JSON file.

    Args:
        image_dir (str): Directory containing the face images.
        output_file (str): Path to save the JSON file with database records.
    """
    database_records = []
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory '{image_dir}' does not exist.")
        return
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    total_files = len(image_files)
    print(f"Found {total_files} images to process in '{image_dir}'.")

    for i, filename in enumerate(image_files):
        image_path = os.path.join(image_dir, filename)
        
        # Generate the embedding for the current face
        embedding = get_face_embedding(image_path)
        
        if embedding is not None:
            record = {
                'id': str(uuid.uuid4()),  # Generate a unique ID for each record
                'image_path': image_path,
                'embedding': embedding.tolist()  # Convert numpy array to list for JSON serialization
            }
            database_records.append(record)
            print(f"Processed {i+1}/{total_files}: {filename}")
        else:
            print(f"Skipped {i+1}/{total_files}: {filename} (no face detected or error)")
            
    # Save the records to a JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(database_records, f, indent=4)
        
    print(f"\nProcessing complete. Saved {len(database_records)} records to '{output_file}'.")

if __name__ == '__main__':
    IMAGE_DIRECTORY = 'data/synthetic_faces'
    OUTPUT_JSON_PATH = 'data/face_database.json'
    process_image_directory(IMAGE_DIRECTORY, OUTPUT_JSON_PATH) 