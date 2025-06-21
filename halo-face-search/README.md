# Facial Similarity Search Service

A high-performance, production-grade facial similarity search service built with FastAPI, ArcFace embeddings, and Milvus vector database. This system can find the top 5 most similar faces from a database of 1,000 faces with sub-2-second latency at 20+ requests per second.

## Architecture Overview

The system is built on a modern, scalable architecture:

- **Face Recognition**: ArcFace model via DeepFace library for state-of-the-art 512-dimensional face embeddings
- **Vector Database**: Milvus with HNSW indexing for ultra-fast similarity search
- **API Framework**: FastAPI with async/await for high-performance REST and WebSocket endpoints
- **Containerization**: Docker and Docker Compose for consistent deployment
- **Performance**: Redis caching (optional), multi-worker Uvicorn for 20+ RPS

## Features

- ✅ REST API endpoint for image-based face search
- ✅ WebSocket support for real-time video stream processing
- ✅ Automatic face detection and alignment
- ✅ Sub-2-second response time
- ✅ Handles 20+ requests per second
- ✅ Docker containerized for easy deployment
- ✅ Comprehensive error handling and validation
- ✅ Interactive API documentation (Swagger UI)

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- 8GB+ RAM (for running Milvus and processing)
- (Optional) NVIDIA GPU for faster face generation

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd halo-face-search
```

### 2. Generate Synthetic Face Dataset
```bash
# Install dependencies locally for data generation
pip install -r requirements.txt

# Generate 1,000 synthetic faces
python scripts/generate_faces.py

# Process faces and create embeddings
python scripts/create_database_records.py
```

### 3. Run with Docker Compose
```bash
# Build and start all services
docker-compose up -d --build

# Check service health
docker-compose ps

# View logs
docker-compose logs -f api
```

### 4. Ingest Data into Milvus
```bash
# First, ensure Milvus is healthy
docker-compose exec api python scripts/milvus_manager.py
```

### 5. Test the API
Visit http://localhost:8000/docs for interactive API documentation.

#### Search for Similar Faces
```bash
curl -X POST "http://localhost:8000/search" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image_file=@path/to/test/face.jpg"
```

## API Endpoints

### POST /search
Find the top 5 most similar faces to an uploaded image.

**Request**: Multipart form data with image file
**Response**: JSON with similar faces and similarity scores

### GET /health
Health check endpoint.

### WebSocket /ws/video_search
Real-time video frame processing for continuous face search.

## Development Setup

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Milvus with Docker
docker-compose up -d etcd minio milvus

# Run the API locally
uvicorn app.main:app --reload
```

### Running Tests
```bash
pytest tests/
```

## Performance Optimization

### Caching with Redis
To enable Redis caching for improved performance:

1. Add Redis to docker-compose.yml
2. Configure FastAPI-cache2 in the application
3. Cache embedding results by image hash

### Scaling
- Increase Uvicorn workers: Modify the `--workers` flag in docker-compose.yml
- Horizontal scaling: Deploy multiple API containers behind a load balancer
- GPU acceleration: Use NVIDIA Docker runtime for faster embedding generation

## Deployment to AWS EC2

### 1. Launch EC2 Instance
- Use Ubuntu Server 22.04 LTS
- Instance type: t3.medium or larger
- Security group: Open ports 22, 8000, 80, 443

### 2. Install Docker
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker $USER
```

### 3. Deploy Application
```bash
# Copy project to EC2
scp -r ./halo-face-search ubuntu@<EC2_IP>:~/

# SSH into instance
ssh ubuntu@<EC2_IP>

# Run the application
cd halo-face-search
docker-compose up -d
```

### 4. Access the API
Your API will be available at: `http://<EC2_PUBLIC_IP>:8000`

## Architecture Decisions

### Why ArcFace over FaceNet?
ArcFace uses Additive Angular Margin Loss which provides:
- More discriminative features
- Better performance on large-scale benchmarks
- Clearer geometric interpretation
- State-of-the-art accuracy

### Why Milvus over Pinecone?
- Open-source and self-hostable
- Greater control over indexing algorithms
- No vendor lock-in
- Better suited for on-premise deployments

### Why FastAPI over Flask?
- Native async/await support
- 3-5x better performance for I/O-bound operations
- Automatic API documentation
- Built-in data validation with Pydantic

## Troubleshooting

### Milvus Connection Error
```bash
# Check if Milvus is healthy
docker-compose ps
docker-compose logs milvus

# Restart Milvus
docker-compose restart milvus
```

### Face Detection Issues
- Ensure images contain clear, front-facing faces
- Check image format (JPEG/PNG supported)
- Verify DeepFace models are downloaded

### Performance Issues
- Monitor resource usage: `docker stats`
- Increase Docker memory allocation
- Scale up EC2 instance type
- Enable Redis caching

## License

This project is developed as a technical demonstration for the Halo internship program. 