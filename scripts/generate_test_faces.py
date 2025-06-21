#!/usr/bin/env python3
"""
Generate a small set of test faces for quick testing
"""

import os
import requests
import time
from PIL import Image
import numpy as np
import cv2

def generate_test_faces(num_faces=10, output_dir='data/synthetic_faces'):
    """Generate a small set of test faces"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_faces} test faces...")
    
    for i in range(num_faces):
        print(f"Generating face {i+1}/{num_faces}...")
        
        # Try to download from thispersondoesnotexist.com
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/", 
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=10
            )
            
            if response.status_code == 200:
                file_path = os.path.join(output_dir, f'face_{i:04d}.jpg')
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"  Downloaded face {i}")
            else:
                print(f"  Failed to download, generating procedural face {i}")
                generate_simple_face(i, output_dir)
                
            # Wait a bit between requests
            time.sleep(1)
            
        except Exception as e:
            print(f"  Error: {e}, generating procedural face {i}")
            generate_simple_face(i, output_dir)
    
    print(f"Generated {num_faces} test faces in {output_dir}")

def generate_simple_face(index, output_dir):
    """Generate a simple procedural face"""
    # Create a simple face image
    img_size = 256
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Random seed for consistency
    np.random.seed(index)
    
    # Face color
    face_color = (200 + np.random.randint(-20, 20),
                  180 + np.random.randint(-20, 20), 
                  160 + np.random.randint(-20, 20))
    
    # Draw face oval
    center = (img_size // 2, img_size // 2)
    cv2.ellipse(img, center, (100, 120), 0, 0, 360, face_color, -1)
    
    # Add eyes
    eye_y = img_size // 2 - 20
    cv2.circle(img, (img_size // 2 - 30, eye_y), 10, (50, 50, 50), -1)
    cv2.circle(img, (img_size // 2 + 30, eye_y), 10, (50, 50, 50), -1)
    
    # Add mouth
    mouth_y = img_size // 2 + 40
    cv2.ellipse(img, (img_size // 2, mouth_y), (30, 15), 0, 0, 180, (100, 50, 50), 2)
    
    # Save
    file_path = os.path.join(output_dir, f'face_{index:04d}.jpg')
    cv2.imwrite(file_path, img)

if __name__ == '__main__':
    generate_test_faces(10) 