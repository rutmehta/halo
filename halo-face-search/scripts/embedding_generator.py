# file: embedding_generator.py
from deepface import DeepFace
import numpy as np

# Specify the ArcFace model for generating embeddings.
# DeepFace will automatically download the model weights on the first run.
MODEL_NAME = "ArcFace"

def get_face_embedding(image_path: str) -> np.ndarray | None:
    """
    Generates a 512-dimensional facial embedding for a given image using ArcFace.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray | None: A 512-dimensional numpy array representing the face embedding,
                           or None if no face is detected.
    """
    try:
        # The represent function handles face detection, alignment, and embedding generation.
        # It returns a list of dictionaries, one for each face found in the image.
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            enforce_detection=True  # Ensures that an exception is raised if no face is found.
        )
        
        # We assume one face per image as per the project spec.
        embedding = embedding_objs[0]['embedding']
        return np.array(embedding)
    except ValueError as e:
        # This error is typically raised by deepface if no face is detected.
        print(f"Error processing image {image_path}: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    # Replace 'path/to/your/face.jpg' with an actual image path.
    test_image_path = 'path/to/your/face.jpg'
    embedding = get_face_embedding(test_image_path)
    
    if embedding is not None:
        print(f"Successfully generated embedding for {test_image_path}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding data type: {embedding.dtype}")
        # print(f"First 5 dimensions: {embedding[:5]}")
    else:
        print(f"Could not generate embedding for {test_image_path}. No face detected or other error.") 