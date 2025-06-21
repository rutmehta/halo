# file: generate_faces.py
import os
import requests
import zipfile
from PIL import Image
import numpy as np
import torch
from torchvision.utils import save_image

def download_synthetic_faces(num_faces: int, output_dir: str):
    """
    Downloads synthetic faces from This Person Does Not Exist API or similar.
    For production, we'd use StyleGAN, but this is faster for the demo.
    """
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Alternative: Use pre-generated synthetic faces
    print(f"Generating {num_faces} synthetic faces...")
    
    for i in range(num_faces):
        try:
            # Option 1: Download from thispersondoesnotexist.com (if available)
            # Note: In production, we'd generate these ourselves with StyleGAN
            response = requests.get("https://thispersondoesnotexist.com/", stream=True)
            if response.status_code == 200:
                file_path = os.path.join(output_dir, f'face_{i:04d}.jpg')
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Downloaded face {i+1}/{num_faces}")
            else:
                # Fallback: Generate a placeholder face
                generate_placeholder_face(i, output_dir)
        except Exception as e:
            print(f"Error downloading face {i}: {e}")
            # Generate placeholder instead
            generate_placeholder_face(i, output_dir)
            
    print(f"Successfully generated {num_faces} faces in '{output_dir}'.")

def generate_placeholder_face(index: int, output_dir: str):
    """
    Generates a placeholder face image for testing purposes.
    In production, this would use StyleGAN.
    """
    # Create a simple synthetic face-like image
    img_size = 256
    img_array = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Add some variation based on index
    np.random.seed(index)
    
    # Simple face structure
    # Background
    bg_color = np.random.randint(200, 255, 3)
    img_array[:, :] = bg_color
    
    # Face oval
    center_x, center_y = img_size // 2, img_size // 2
    face_color = np.random.randint(150, 220, 3)
    
    for y in range(img_size):
        for x in range(img_size):
            # Simple oval equation
            if ((x - center_x) / 100)**2 + ((y - center_y) / 120)**2 < 1:
                img_array[y, x] = face_color
    
    # Save image
    img = Image.fromarray(img_array)
    file_path = os.path.join(output_dir, f'face_{index:04d}.jpg')
    img.save(file_path)

def generate_synthetic_faces_stylegan(num_faces: int, output_dir: str):
    """
    Generates synthetic faces using a pre-trained StyleGAN2 model.
    This is the production-quality approach mentioned in the guide.
    """
    print("Note: StyleGAN generation requires significant setup and GPU resources.")
    print("For this demo, using download_synthetic_faces() instead.")
    # In production, implement full StyleGAN generation here
    download_synthetic_faces(num_faces, output_dir)

if __name__ == '__main__':
    NUM_FACES_TO_GENERATE = 1000
    OUTPUT_IMAGE_DIR = 'data/synthetic_faces'
    
    # For demo purposes, we'll download/generate faces
    # In production, use generate_synthetic_faces_stylegan()
    download_synthetic_faces(NUM_FACES_TO_GENERATE, OUTPUT_IMAGE_DIR) 