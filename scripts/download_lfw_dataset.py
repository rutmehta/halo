#!/usr/bin/env python3
"""
Download LFW (Labeled Faces in the Wild) dataset
Real face dataset for production face recognition system
"""

import os
import requests
import tarfile
import shutil
from tqdm import tqdm
import json
import subprocess

def download_lfw_dataset(data_dir='data'):
    """Download LFW dataset - 13,233 images of 5,749 people"""
    
    print("🔥 Downloading LFW Dataset for Halo Face Search")
    print("📊 Dataset: 13,233 images from 5,749 real people")
    print("🎯 Perfect for production face recognition testing\n")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    lfw_dir = os.path.join(data_dir, 'lfw_faces')
    os.makedirs(lfw_dir, exist_ok=True)
    
    # LFW dataset URL (official mirror)
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    lfw_tar_path = os.path.join(data_dir, 'lfw.tgz')
    
    try:
        # Download LFW dataset
        print(f"📥 Downloading LFW dataset from: {lfw_url}")
        print("⏳ This may take a few minutes (145MB download)...")
        
        response = requests.get(lfw_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(lfw_tar_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded LFW dataset: {lfw_tar_path}")
        
        # Extract the dataset
        print("📦 Extracting LFW dataset...")
        with tarfile.open(lfw_tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        # Move extracted files to proper location
        extracted_lfw = os.path.join(data_dir, 'lfw')
        if os.path.exists(extracted_lfw):
            print("📁 Organizing LFW faces...")
            
            # Count and organize faces
            face_count = 0
            for person_dir in os.listdir(extracted_lfw):
                person_path = os.path.join(extracted_lfw, person_dir)
                if os.path.isdir(person_path):
                    for face_file in os.listdir(person_path):
                        if face_file.lower().endswith(('.jpg', '.jpeg')):
                            # Copy to flat structure for easier processing
                            src = os.path.join(person_path, face_file)
                            # Create unique filename: person_image.jpg
                            dst_name = f"{person_dir}_{face_file}"
                            dst = os.path.join(lfw_dir, dst_name)
                            shutil.copy2(src, dst)
                            face_count += 1
            
            print(f"✅ Organized {face_count} LFW faces in {lfw_dir}")
            
            # Clean up
            shutil.rmtree(extracted_lfw)
            os.remove(lfw_tar_path)
            
            print("🎉 LFW dataset ready for face recognition!")
            print(f"📊 Total faces available: {len(os.listdir(os.path.join(data_dir, 'synthetic_faces')))} synthetic + {face_count} real")
            
            return True
            
    except Exception as e:
        print(f"❌ Error downloading LFW dataset: {e}")
        print("💡 Alternative: Using synthetic faces only")
        return False

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

if __name__ == "__main__":
    download_lfw_dataset()
    
    # Create sample for demo (1000 faces)
    sample_dir = sample_faces_for_demo(max_faces=1000)
    
    print("\n🎯 Ready for next step: Generate embeddings!")
    print("Run: python scripts/generate_embeddings.py") 