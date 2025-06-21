#!/bin/bash

# Setup script for Facial Similarity Search Service

echo "=== Facial Similarity Search Service Setup ==="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3.10+"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directories
echo "Creating data directories..."
mkdir -p data/synthetic_faces

# Generate faces
echo ""
echo "Step 1: Generating synthetic faces..."
echo "This will download/generate 1,000 face images."
echo "Note: For demo purposes, this uses placeholder faces."
echo "In production, use StyleGAN for truly synthetic faces."
echo ""
read -p "Press Enter to continue..."
python scripts/generate_faces.py

# Create embeddings
echo ""
echo "Step 2: Creating face embeddings..."
echo "This will process all faces and generate ArcFace embeddings."
echo "Note: This may take several minutes and will download model weights on first run."
echo ""
read -p "Press Enter to continue..."
python scripts/create_database_records.py

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Run 'docker-compose up -d' to start all services"
echo "2. Run 'docker-compose exec api python scripts/milvus_manager.py' to ingest data"
echo "3. Visit http://localhost:8000/docs to test the API"
echo "" 