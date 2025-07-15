# 🚀 AICrisisAlert Deployment Guide

This guide covers deploying AICrisisAlert in various environments, from local development to production.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [API Documentation](#api-documentation)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- Git

### 1. Clone and Setup

```bash
git clone https://github.com/your-username/AICrisisAlert.git
cd AICrisisAlert

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the API (Simple Mode)

```bash
# Start with mock model (for testing)
python scripts/start_api.py

# Or directly with uvicorn
uvicorn src.api.main_simple:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the API

```bash
# In another terminal
python scripts/test_api.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🛠️ Local Development

### Development Setup

1. **Environment Configuration**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit environment variables
   nano .env
   ```

2. **Database Setup** (Optional)
   ```bash
   # Start PostgreSQL with Docker
   docker run -d \
     --name postgres-crisis \
     -e POSTGRES_DB=crisis_alert \
     -e POSTGRES_USER=postgres \
     -e POSTGRES_PASSWORD=password \
     -p 5432:5432 \
     postgres:15-alpine
   ```

3. **Start Development Server**
   ```bash
   # With hot reload
   python scripts/start_api.py
   
   # Or with full model (requires trained model)
   USE_SIMPLE_API=false python scripts/start_api.py
   ```

### Development Workflow

1. **API Development**
   - Edit `src/api/main_simple.py` for quick testing
   - Use `src/api/main.py` for full functionality
   - Test with `scripts/test_api.py`

2. **Model Training** (on Windows with GPU)
   ```bash
   # Train the model (run on Windows machine)
   python scripts/enhanced_feature_engineering.py
   ```

3. **Code Quality**
   ```bash
   # Format code
   black src/ scripts/
   
   # Lint code
   flake8 src/ scripts/
   
   # Type checking
   mypy src/
   ```

## 🐳 Docker Deployment

### Quick Docker Start

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### Development with Docker

```bash
# Start development environment
docker-compose --profile dev up -d

# Access development API
curl http://localhost:8001/health
```

### Production with Docker

```bash
# Start production stack
docker-compose --profile production up -d

# Access production API
curl http://localhost:8000/health
```

### Custom Docker Build

```bash
# Build development image
docker build --target development -t crisis-alert:dev .

# Build production image
docker build --target production -t crisis-alert:prod .

# Run custom container
docker run -p 8000:8000 crisis-alert:prod
```

## 🌐 Production Deployment

### AWS Deployment

1. **EC2 Setup**
   ```bash
   # Launch EC2 instance
   # Install Docker
   sudo yum update -y
   sudo yum install -y docker
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **Deploy Application**
   ```bash
   # Clone repository
   git clone https://github.com/your-username/AICrisisAlert.git
   cd AICrisisAlert
   
   # Set environment variables
   export MODEL_PATH=/app/outputs/models/bertweet_enhanced
   export LOG_LEVEL=WARNING
   export SECRET_KEY=your-production-secret-key
   
   # Start production stack
   docker-compose --profile production up -d
   ```

3. **Load Balancer Setup**
   ```bash
   # Configure ALB/ELB to point to port 8000
   # Set up SSL certificates
   # Configure auto-scaling groups
   ```

### Environment Variables

Create `.env` file for production:

```env
# API Configuration
API_TITLE=AICrisisAlert API
API_VERSION=1.0.0
HOST=0.0.0.0
PORT=8000

# Model Configuration
MODEL_PATH=/app/outputs/models/bertweet_enhanced
MODEL_TYPE=bertweet
MODEL_VERSION=1.0.0

# Database Configuration
DATABASE_URL=postgresql://user:password@host:5432/crisis_alert
REDIS_URL=redis://host:6379

# Security
SECRET_KEY=your-super-secret-key-here
ALLOWED_HOSTS=your-domain.com,api.your-domain.com

# Monitoring
LOG_LEVEL=WARNING
ENABLE_METRICS=true

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
S3_BUCKET=your-crisis-alert-bucket
```

## 📚 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/classify` | Single text classification |
| POST | `/classify/batch` | Batch classification |
| POST | `/classify/emergency` | Emergency classification |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Single classification
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT: People trapped in building collapse!"}'

# Batch classification
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Emergency situation", "Infrastructure damage"]}'
```

### Response Format

```json
{
  "predicted_class": "urgent_help",
  "confidence": 0.92,
  "class_probabilities": {
    "urgent_help": 0.92,
    "infrastructure_damage": 0.05,
    "casualty_info": 0.02,
    "resource_availability": 0.01,
    "general_info": 0.00
  },
  "processing_time_ms": 150.5,
  "model_version": "1.0.0"
}
```

## 📊 Monitoring & Logging

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check Docker services
docker-compose ps

# Check service logs
docker-compose logs api
```

### Metrics (Production)

- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000`
- **Flower** (Celery): `http://localhost:5555`

### Log Management

```bash
# View real-time logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api postgres redis

# Export logs
docker-compose logs api > api.log
```

## 🔧 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   
   # Kill process
   kill -9 <PID>
   ```

2. **Docker Build Fails**
   ```bash
   # Clean Docker cache
   docker system prune -a
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

3. **Model Loading Issues**
   ```bash
   # Check model path
   ls -la outputs/models/
   
   # Use simple API for testing
   USE_SIMPLE_API=true python scripts/start_api.py
   ```

4. **Database Connection Issues**
   ```bash
   # Check PostgreSQL status
   docker-compose ps postgres
   
   # View database logs
   docker-compose logs postgres
   
   # Reset database
   docker-compose down -v
   docker-compose up -d postgres
   ```

### Performance Optimization

1. **API Performance**
   ```bash
   # Increase workers
   export WORKERS=4
   
   # Use Gunicorn in production
   gunicorn src.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
   ```

2. **Model Optimization**
   ```bash
   # Use ONNX for faster inference
   python scripts/convert_to_onnx.py
   
   # Enable model caching
   export ENABLE_MODEL_CACHE=true
   ```

### Security Checklist

- [ ] Change default passwords
- [ ] Configure SSL/TLS certificates
- [ ] Set up firewall rules
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Use environment variables for secrets
- [ ] Regular security updates

## 📞 Support

For issues and questions:

1. Check the [troubleshooting section](#troubleshooting)
2. Review [API documentation](#api-documentation)
3. Open an issue on GitHub
4. Check the logs: `docker-compose logs`

## 🔄 Updates

To update the application:

```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose build

# Restart services
docker-compose up -d

# Check status
docker-compose ps
``` 