# Deployment Guide: Facial Similarity Search Service

This guide provides detailed instructions for deploying the Facial Similarity Search Service from development to production.

## Table of Contents
1. [Local Development Setup](#local-development-setup)
2. [Docker Deployment](#docker-deployment)
3. [AWS EC2 Deployment](#aws-ec2-deployment)
4. [Performance Tuning](#performance-tuning)
5. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

## Local Development Setup

### Prerequisites
- Python 3.10+
- Docker Desktop installed
- 8GB+ RAM available
- 10GB+ disk space

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd halo-face-search

# Run the automated setup script
./setup.sh
```

### Step 2: Start Services with Docker
```bash
# Start all services (Milvus, API, etc.)
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f
```

### Step 3: Ingest Data into Milvus
```bash
# Wait for Milvus to be fully initialized (check logs)
# Then run the ingestion script
docker-compose exec api python scripts/milvus_manager.py
```

### Step 4: Test the API
```bash
# Test with the provided script
./test_api.py

# Or test with curl
curl -X POST "http://localhost:8000/search" \
  -F "image_file=@data/synthetic_faces/face_0000.jpg"

# View API documentation
open http://localhost:8000/docs
```

## Docker Deployment

### Building Images
```bash
# Build the API image
docker build -t face-search-api:latest .

# Or use docker-compose
docker-compose build
```

### Running with Docker Compose
```bash
# Start in detached mode
docker-compose up -d

# Scale the API service
docker-compose up -d --scale api=3

# Stop all services
docker-compose down

# Remove all data (careful!)
docker-compose down -v
```

### Docker Resource Limits
Edit `docker-compose.yml` to set resource limits:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## AWS EC2 Deployment

### Step 1: Launch EC2 Instance
1. Go to AWS Console → EC2 → Launch Instance
2. Select Ubuntu Server 22.04 LTS
3. Choose instance type:
   - Development: t3.medium (2 vCPU, 4GB RAM)
   - Production: t3.xlarge (4 vCPU, 16GB RAM) or larger
4. Configure storage: 30GB+ GP3 SSD
5. Security Group settings:
   ```
   - SSH (22): Your IP
   - HTTP (80): 0.0.0.0/0
   - HTTPS (443): 0.0.0.0/0
   - Custom TCP (8000): 0.0.0.0/0
   ```

### Step 2: Connect and Install Dependencies
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login for group changes
exit
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 3: Deploy Application
```bash
# Copy files to EC2 (from your local machine)
scp -i your-key.pem -r ./halo-face-search ubuntu@<EC2-PUBLIC-IP>:~/

# On EC2: Navigate to project
cd ~/halo-face-search

# Start services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

### Step 4: Configure for Production
```bash
# Use production docker-compose
cp docker-compose.prod.yml docker-compose.override.yml

# Set environment variables
echo "MILVUS_HOST=milvus" > .env
echo "API_WORKERS=4" >> .env

# Restart services
docker-compose down
docker-compose up -d
```

### Step 5: Setup Domain (Optional)
```bash
# Install Nginx
sudo apt-get install -y nginx

# Configure reverse proxy
sudo nano /etc/nginx/sites-available/face-search

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/face-search /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Performance Tuning

### API Performance
1. **Increase Workers**: Edit docker-compose.yml
   ```yaml
   command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "8"]
   ```

2. **Enable Caching**: Add Redis service
   ```yaml
   redis:
     image: redis:7-alpine
     ports:
       - "6379:6379"
   ```

3. **Optimize Milvus**: Increase search parameters
   ```python
   search_params = {
       "metric_type": "L2",
       "params": {"ef": 256}  # Increase for better accuracy
   }
   ```

### System Performance
```bash
# Monitor resource usage
docker stats

# Check Milvus metrics
curl http://localhost:9091/metrics

# Monitor API logs
docker-compose logs -f api
```

### Load Testing
```bash
# Install Apache Bench
sudo apt-get install -y apache2-utils

# Test endpoint performance
ab -n 1000 -c 20 -p test_image.jpg -T multipart/form-data http://localhost:8000/search
```

## Monitoring & Troubleshooting

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Milvus health
curl http://localhost:9091/healthz

# Container status
docker-compose ps
```

### Common Issues

#### 1. Milvus Connection Failed
```bash
# Check Milvus logs
docker-compose logs milvus

# Verify network
docker network ls
docker network inspect halo-face-search_default

# Restart Milvus
docker-compose restart milvus
```

#### 2. Out of Memory
```bash
# Check memory usage
docker stats

# Increase Docker memory limit
# Docker Desktop: Preferences → Resources → Memory

# Or use swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. Slow Performance
- Check CPU usage: `htop`
- Monitor disk I/O: `iotop`
- Profile API: Add timing logs
- Scale horizontally: Add more API containers

### Backup & Recovery
```bash
# Backup Milvus data
docker-compose exec milvus tar -czf /tmp/backup.tar.gz /var/lib/milvus
docker cp milvus:/tmp/backup.tar.gz ./milvus-backup-$(date +%Y%m%d).tar.gz

# Restore Milvus data
docker cp ./milvus-backup.tar.gz milvus:/tmp/
docker-compose exec milvus tar -xzf /tmp/milvus-backup.tar.gz -C /
```

## Security Considerations

1. **API Authentication**: Add JWT tokens
2. **HTTPS**: Use Let's Encrypt with Certbot
3. **Firewall**: Configure UFW or Security Groups
4. **Secrets Management**: Use Docker secrets or AWS Secrets Manager
5. **Rate Limiting**: Implement with FastAPI middleware

## Next Steps

1. Implement Redis caching for improved performance
2. Add comprehensive logging with ELK stack
3. Set up CI/CD with GitHub Actions
4. Implement horizontal scaling with Kubernetes
5. Add GPU support for faster embedding generation

For questions or issues, please refer to the main README.md or create an issue in the repository. 