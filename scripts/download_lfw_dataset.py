#!/usr/bin/env python3
"""
Download and organize LFW (Labeled Faces in the Wild) dataset
Real face dataset for building production face recognition system
"""

import os
import requests
import tarfile
import shutil
from tqdm import tqdm
import json

def download_lfw_dataset(data_dir='data'):
    """
    Download LFW dataset - 13,233 images of 5,749 people
    Perfect for real face recognition testing
    """
    
    print("🔥 Downloading LFW Dataset for Halo Face Search")
    print("📊 Dataset info: 13,233 images from 5,749 people")
    print("🎯 Real faces for production-quality face recognition\n")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    lfw_dir = os.path.join(data_dir, 'lfw_faces')
    os.makedirs(lfw_dir, exist_ok=True)
    
    # LFW dataset URL
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    tar_path = os.path.join(data_dir, 'lfw.tgz')
    
    # Download if not exists
    if not os.path.exists(tar_path):
        print(f"📥 Downloading LFW dataset (173MB)...")
        
        response = requests.get(lfw_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print("✅ Download completed!")
    else:
        print("✅ LFW dataset already downloaded")
    
    # Extract if not already extracted
    extracted_dir = os.path.join(data_dir, 'lfw')
    if not os.path.exists(extracted_dir):
        print("📂 Extracting dataset...")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        print("✅ Extraction completed!")
    else:
        print("✅ Dataset already extracted")
    
    # Organize faces for easier processing
    organize_faces(extracted_dir, lfw_dir)
    
    # Create metadata
    create_metadata(lfw_dir)
    
    print(f"\n🎯 LFW dataset ready at: {lfw_dir}")
    return lfw_dir

def organize_faces(source_dir, target_dir):
    """Organize faces into a flat structure for easier processing"""
    
    print("🗂️  Organizing faces...")
    
    face_count = 0
    person_count = 0
    
    # Walk through person directories
    for person_name in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person_name)
        
        if not os.path.isdir(person_path):
            continue
            
        person_count += 1
        
        # Copy all images for this person
        for image_file in os.listdir(person_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                source_image = os.path.join(person_path, image_file)
                
                # Create unique filename: personname_imagename.jpg
                new_filename = f"{person_name}_{image_file}"
                target_image = os.path.join(target_dir, new_filename)
                
                if not os.path.exists(target_image):
                    shutil.copy2(source_image, target_image)
                
                face_count += 1
        
        if person_count % 100 == 0:
            print(f"   Processed {person_count} people, {face_count} faces...")
    
    print(f"✅ Organized {face_count} faces from {person_count} people")

def create_metadata(lfw_dir):
    """Create metadata file with dataset information"""
    
    print("📝 Creating metadata...")
    
    # Count files and people
    face_files = [f for f in os.listdir(lfw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Extract person names
    people = set()
    for filename in face_files:
        person_name = filename.split('_')[0]
        people.add(person_name)
    
    metadata = {
        "dataset": "LFW (Labeled Faces in the Wild)",
        "description": "Real face dataset for face recognition",
        "total_faces": len(face_files),
        "total_people": len(people),
        "source": "http://vis-www.cs.umass.edu/lfw/",
        "purpose": "Production face recognition system",
        "sample_people": sorted(list(people))[:20],  # First 20 people as sample
        "image_format": "JPG",
        "organized_date": str(os.path.getctime(lfw_dir))
    }
    
    metadata_path = os.path.join(lfw_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {metadata_path}")
    print(f"📊 Dataset: {len(face_files)} faces from {len(people)} people")

def sample_faces_for_demo(lfw_dir, sample_dir='data/sample_faces', max_faces=1000):
    """Create a sample subset for faster demo/testing"""
    
    print(f"\n🎯 Creating sample dataset ({max_faces} faces) for demo...")
    
    os.makedirs(sample_dir, exist_ok=True)
    
    # Get all face files
    face_files = [f for f in os.listdir(lfw_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sample faces (take every nth face to ensure diversity)
    step = max(1, len(face_files) // max_faces)
    sampled_faces = face_files[::step][:max_faces]
    
    print(f"📥 Copying {len(sampled_faces)} faces to sample directory...")
    
    for i, face_file in enumerate(sampled_faces):
        source = os.path.join(lfw_dir, face_file)
        target = os.path.join(sample_dir, face_file)
        
        if not os.path.exists(target):
            shutil.copy2(source, target)
        
        if (i + 1) % 100 == 0:
            print(f"   Copied {i + 1}/{len(sampled_faces)} faces...")
    
    # Create sample metadata
    sample_metadata = {
        "dataset": "LFW Sample for Halo Demo",
        "total_faces": len(sampled_faces),
        "source_dataset": "LFW",
        "purpose": "Demo and testing subset"
    }
    
    with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
        json.dump(sample_metadata, f, indent=2)
    
    print(f"✅ Sample dataset ready: {sample_dir}")
    print(f"🚀 {len(sampled_faces)} faces ready for embedding generation!")
    
    return sample_dir

if __name__ == '__main__':
    # Download and organize LFW dataset
    lfw_dir = download_lfw_dataset()
    
    # Create sample for demo (1000 faces)
    sample_dir = sample_faces_for_demo(lfw_dir, max_faces=1000)
    
    print("\n🎯 Ready for next step: Generate embeddings!")
    print("Run: python scripts/generate_embeddings.py") 