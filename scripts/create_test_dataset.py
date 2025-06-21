#!/usr/bin/env python3
"""
Create a test face dataset for the face recognition system
"""

import os
import json
import shutil
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

def create_test_faces(num_faces=100, output_dir='data/real_faces'):
    """
    Create simple test faces for the demo
    In production, you'd use real datasets like VGGFace2 or LFW
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_faces} test face images...")
    print("Note: These are simple geometric faces for demo purposes")
    print("For production, use real datasets like VGGFace2 or LFW\n")
    
    metadata = {
        'dataset': 'geometric_test_faces',
        'total_faces': num_faces,
        'description': 'Simple geometric faces for testing face recognition system'
    }
    
    # Create diverse face variations
    face_colors = [
        (255, 219, 172),  # Light skin
        (241, 194, 125),  # Medium light
        (224, 172, 105),  # Medium
        (198, 134, 66),   # Medium dark
        (141, 85, 36),    # Dark
    ]
    
    hair_colors = [
        (101, 67, 33),    # Brown
        (62, 43, 31),     # Black
        (255, 255, 0),    # Blonde
        (165, 42, 42),    # Auburn
        (128, 128, 128),  # Gray
    ]
    
    for i in range(num_faces):
        # Create 256x256 image
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        
        # Random variations
        face_color = random.choice(face_colors)
        hair_color = random.choice(hair_colors)
        
        # Face oval
        face_x1 = 64 + random.randint(-10, 10)
        face_y1 = 80 + random.randint(-10, 10)
        face_x2 = 192 + random.randint(-10, 10)
        face_y2 = 220 + random.randint(-10, 10)
        draw.ellipse([face_x1, face_y1, face_x2, face_y2], fill=face_color)
        
        # Hair
        hair_y = face_y1 - random.randint(20, 40)
        draw.ellipse([face_x1-10, hair_y, face_x2+10, face_y1+30], fill=hair_color)
        
        # Eyes
        eye_y = face_y1 + 40 + random.randint(-5, 5)
        eye_size = random.randint(8, 12)
        draw.ellipse([face_x1+30, eye_y, face_x1+30+eye_size, eye_y+eye_size], fill='black')
        draw.ellipse([face_x2-30-eye_size, eye_y, face_x2-30, eye_y+eye_size], fill='black')
        
        # Nose
        nose_x = (face_x1 + face_x2) // 2
        nose_y = eye_y + 20 + random.randint(-5, 5)
        draw.ellipse([nose_x-3, nose_y, nose_x+3, nose_y+6], fill=(200, 150, 100))
        
        # Mouth
        mouth_y = nose_y + 25 + random.randint(-5, 5)
        mouth_width = random.randint(15, 25)
        draw.ellipse([nose_x-mouth_width//2, mouth_y, nose_x+mouth_width//2, mouth_y+8], fill='red')
        
        # Save image
        img_path = os.path.join(output_dir, f'face_{i:05d}.jpg')
        img.save(img_path, 'JPEG', quality=95)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_faces} faces...")
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Created {num_faces} test face images in {output_dir}")
    print("\nFor production use:")
    print("1. VGGFace2: 3.31M images of 9,131 people")
    print("2. LFW: 13,233 images for face verification")
    print("3. CASIA-WebFace: 500K images of 10,575 people")
    
    return num_faces

if __name__ == '__main__':
    create_test_faces(100) 