#!/usr/bin/env python3
"""
Download LFW (Labeled Faces in the Wild) dataset
Real face data to supplement synthetic faces for better recognition
"""

import os
import numpy as np
import cv2
from sklearn.datasets import fetch_lfw_people
import shutil

def download_lfw():
    """Download LFW dataset using scikit-learn's built-in fetcher"""
    
    print("🔥 Downloading LFW Dataset (Real Faces) via Scikit-Learn")
    print("📊 Dataset: Real faces for better face recognition")
    print("📁 This will create a separate 'lfw_faces' folder alongside 'synthetic_faces'")
    
    data_dir = 'data'
    lfw_faces_dir = os.path.join(data_dir, 'lfw_faces')
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(lfw_faces_dir):
        shutil.rmtree(lfw_faces_dir)
    os.makedirs(lfw_faces_dir, exist_ok=True)
    
    try:
        print("📥 Fetching LFW dataset... this may take a few minutes")
        
        # Fetch LFW people dataset (only people with 70+ images for quality)
        lfw_dataset = fetch_lfw_people(
            min_faces_per_person=70,  # Only people with many images
            resize=0.4,  # Resize to reasonable size
            download_if_missing=True
        )
        
        print(f"✅ Downloaded {len(lfw_dataset.images)} images of {len(lfw_dataset.target_names)} people")
        
        # Convert and save images
        face_count = 0
        for i, (image, target) in enumerate(zip(lfw_dataset.images, lfw_dataset.target)):
            person_name = lfw_dataset.target_names[target]
            # Clean person name for filename
            clean_name = person_name.replace(' ', '_').replace('.', '')
            filename = f"{clean_name}_{i:04d}.jpg"
            filepath = os.path.join(lfw_faces_dir, filename)
            
            # Convert from sklearn format (0-1 float) to standard image (0-255 uint8)
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Save image
            cv2.imwrite(filepath, image_uint8)
            face_count += 1
            
            if face_count % 100 == 0:
                print(f"📁 Saved {face_count} faces...")
        
        # Summary
        synthetic_dir = os.path.join(data_dir, 'synthetic_faces')
        synthetic_count = len(os.listdir(synthetic_dir)) if os.path.exists(synthetic_dir) else 0
        
        print(f"\n🎉 LFW dataset successfully organized!")
        print(f"📊 LFW faces: {face_count} real faces from {len(lfw_dataset.target_names)} people")
        print(f"📊 Synthetic faces: {synthetic_count} faces")
        print(f"📊 Total faces available: {synthetic_count + face_count}")
        print(f"📁 LFW faces location: {lfw_faces_dir}")
        
        return {
            'success': True,
            'lfw_faces': face_count,
            'lfw_people': len(lfw_dataset.target_names),
            'synthetic_faces': synthetic_count,
            'total_faces': synthetic_count + face_count,
            'lfw_directory': lfw_faces_dir
        }
        
    except ImportError:
        print("❌ scikit-learn not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scikit-learn'])
        print("✅ Installed scikit-learn. Please run the script again.")
        return {'success': False, 'error': 'Need to install scikit-learn'}
        
    except Exception as e:
        print(f"\n❌ Error downloading LFW dataset: {e}")
        print("💡 Will continue with synthetic faces only")
        
        synthetic_dir = os.path.join(data_dir, 'synthetic_faces')
        synthetic_count = len(os.listdir(synthetic_dir)) if os.path.exists(synthetic_dir) else 0
        
        return {
            'success': False,
            'error': str(e),
            'lfw_faces': 0,
            'synthetic_faces': synthetic_count
        }

def download_lfw_alternative():
    """Alternative: Use tensorflow/datasets or manual download instructions"""
    print("\n🔧 ALTERNATIVE METHODS:")
    print("\n📋 Method 1 - Manual Download:")
    print("   1. Visit: https://vis-www.cs.umass.edu/lfw/")
    print("   2. Download 'All images as gzipped tar file (173MB)'")
    print("   3. Extract to: data/lfw_faces/")
    print("   4. Run the face loading script")
    
    print("\n📋 Method 2 - TensorFlow Datasets:")
    print("   pip install tensorflow-datasets")
    print("   import tensorflow_datasets as tfds")
    print("   ds = tfds.load('lfw', split='train')")
    
    print("\n📋 Method 3 - Continue with Synthetic Only:")
    print("   Your 1000 synthetic faces are sufficient for testing!")
    print("   Real faces mainly improve accuracy, not core functionality.")

if __name__ == "__main__":
    result = download_lfw()
    
    if not result['success']:
        download_lfw_alternative()
        print(f"\n⚠️ Continuing with {result.get('synthetic_faces', 0)} synthetic faces")
    else:
        print(f"\n🚀 Ready to index {result['total_faces']} faces into Milvus database!")
