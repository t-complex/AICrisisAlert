# AICrisisAlert Deployment Guide

This guide covers deployment strategies for the AICrisisAlert system across different environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [AWS Deployment](#aws-deployment)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Security Configuration](#security-configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.4GHz
- RAM: 8GB
- Storage: 50GB SSD
- Network: 100Mbps internet connection

**Recommended Requirements:**
- CPU: 8 cores, 3.0GHz or NVIDIA GPU (RTX 3060+)
- RAM: 16GB
- Storage: 100GB NVMe SSD
- Network: 1Gbps internet connection

### Software Dependencies

- **Docker**: 20.10+ and Docker Compose 2.0+
- **Python**: 3.9+ (for development/testing)
- **PostgreSQL**: 14+ (if not using Docker)
- **Redis**: 7+ (if not using Docker)
- **Git**: Latest version

### Required Accounts/Services

- Docker Hub account (for pulling images)
- AWS account (for cloud deployment)
- Domain and SSL certificates (for production)

## Environment Configuration

### 1. Clone Repository

```bash
git clone https://github.com/your-org/AICrisisAlert.git
cd AICrisisAlert
```

### 2. Environment Variables

Create environment file from template:

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```bash
# Security Configuration (REQUIRED)
SECRET_KEY=your-32-character-secret-key-here
API_KEY=your-32-character-api-key-here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
CORS_ORIGINS=http://localhost:3000,https://your-frontend.com

# Database Configuration
DATABASE_URL=postgresql://postgres:secure_password@localhost:5432/aicrisisalert
REDIS_URL=redis://localhost:6379

# Docker Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_database_password
POSTGRES_DB=aicrisisalert

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=3600

# Model Configuration
MODEL_PATH=outputs/models/bertweet_enhanced
MODEL_TYPE=bertweet
MODEL_VERSION=1.0.0

# AWS Configuration (Production)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET=your-model-bucket

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

### 3. Generate Secure Keys

Generate cryptographically secure keys:

```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate API_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Docker Deployment

### Development Environment

For development and testing:

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Production Environment

For production deployment:

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d

# Scale API instances
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# Update services without downtime
docker-compose -f docker-compose.prod.yml up -d --no-deps api
```

### Service Configuration

The Docker setup includes:

- **API Service**: FastAPI application with model loading
- **Database**: PostgreSQL with persistent storage
- **Cache**: Redis for performance optimization
- **Background Workers**: Celery for async tasks
- **Monitoring**: Prometheus metrics collection
- **Reverse Proxy**: Nginx for load balancing

### Container Security

Security features enabled:

```yaml
security_opt:
  - no-new-privileges:true
read_only: true  # Where possible
tmpfs:
  - /tmp
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
```

## Production Deployment

### 1. SSL/TLS Configuration

#### Let's Encrypt with Certbot

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

#### Custom SSL Certificate

Place certificates in `ssl/` directory:

```
ssl/
├── cert.pem
├── privkey.pem
└── fullchain.pem
```

### 2. Nginx Configuration

Create `/etc/nginx/sites-available/aicrisisalert`:

```nginx
upstream aicrisisalert_backend {
    server api:8000;
    # Add more servers for load balancing
    # server api2:8000;
    # server api3:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://aicrisisalert_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    location /health {
        proxy_pass http://aicrisisalert_backend/health;
        access_log off;
    }
}
```

### 3. Systemd Service

Create `/etc/systemd/system/aicrisisalert.service`:

```ini
[Unit]
Description=AICrisisAlert Docker Compose Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/aicrisisalert
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable aicrisisalert.service
sudo systemctl start aicrisisalert.service
```

## AWS Deployment

### 1. Infrastructure Setup (Terraform)

Create `infrastructure/main.tf`:

```hcl
provider "aws" {
  region = var.aws_region
}

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "aicrisisalert-vpc"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "aicrisisalert-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# RDS Database
resource "aws_db_instance" "main" {
  identifier     = "aicrisisalert-db"
  engine         = "postgres"
  engine_version = "14.6"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  
  db_name  = "aicrisisalert"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "aicrisisalert-final-snapshot"

  tags = {
    Name = "aicrisisalert-database"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "aicrisisalert-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_cluster" "main" {
  cluster_id           = "aicrisisalert-cache"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]

  tags = {
    Name = "aicrisisalert-redis"
  }
}

# S3 Bucket for Model Storage
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models-${random_id.bucket_suffix.hex}"

  tags = {
    Name = "aicrisisalert-models"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

### 2. ECS Task Definition

Create `aws/task-definition.json`:

```json
{
  "family": "aicrisisalert-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/aicrisisalert-task-role",
  "containerDefinitions": [
    {
      "name": "aicrisisalert-api",
      "image": "your-account.dkr.ecr.us-east-1.amazonaws.com/aicrisisalert:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/outputs/models/bertweet_enhanced"
        },
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:aicrisisalert/database-url"
        },
        {
          "name": "API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:aicrisisalert/api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/aicrisisalert",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### 3. Deployment Script

Create `scripts/deploy-aws.sh`:

```bash
#!/bin/bash

set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPOSITORY="aicrisisalert"
ECS_CLUSTER="aicrisisalert-cluster"
ECS_SERVICE="aicrisisalert-service"

# Build and push Docker image
echo "Building Docker image..."
docker build -t $ECR_REPOSITORY:latest .

# Get ECR login token
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag and push image
docker tag $ECR_REPOSITORY:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Update ECS service
echo "Updating ECS service..."
aws ecs update-service \
    --cluster $ECS_CLUSTER \
    --service $ECS_SERVICE \
    --force-new-deployment \
    --region $AWS_REGION

# Wait for deployment to complete
echo "Waiting for deployment to complete..."
aws ecs wait services-stable \
    --cluster $ECS_CLUSTER \
    --services $ECS_SERVICE \
    --region $AWS_REGION

echo "Deployment completed successfully!"
```

## Monitoring & Maintenance

### 1. Health Monitoring

Set up health check endpoints:

```bash
# API health check
curl -f http://localhost:8000/health

# Database connectivity
curl -f http://localhost:8000/health | jq '.database_status'

# Model status
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/model/info
```

### 2. Log Management

Configure centralized logging:

```yaml
# docker-compose.yml logging configuration
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

For production, consider:
- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **AWS CloudWatch**: For AWS deployments
- **Fluentd**: For log aggregation

### 3. Metrics Collection

Prometheus metrics are exposed at `/metrics`:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'aicrisisalert'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

### 4. Backup Strategy

#### Database Backups

```bash
#!/bin/bash
# Automated PostgreSQL backup
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/aicrisisalert_$TIMESTAMP.sql"

# Create backup
docker exec aicrisisalert-db pg_dump -U postgres aicrisisalert > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to S3 (optional)
aws s3 cp $BACKUP_FILE.gz s3://your-backup-bucket/database/

# Cleanup old backups (keep 7 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
```

#### Model Backups

```bash
# Backup trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz outputs/models/
aws s3 cp models_backup_*.tar.gz s3://your-backup-bucket/models/
```

### 5. Update Procedures

#### Rolling Updates

```bash
# Build new image
docker build -t aicrisisalert:v1.1.0 .

# Update docker-compose.yml with new version
sed -i 's/aicrisisalert:latest/aicrisisalert:v1.1.0/g' docker-compose.prod.yml

# Rolling update
docker-compose -f docker-compose.prod.yml up -d --no-deps api

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
```

## Security Configuration

### 1. Firewall Configuration

```bash
# UFW firewall rules
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 2. Container Security

- Run containers as non-root users
- Use read-only filesystems where possible
- Implement resource limits
- Regular security updates

### 3. Secrets Management

Use AWS Secrets Manager or HashiCorp Vault:

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
    --name "aicrisisalert/api-key" \
    --description "API key for AICrisisAlert" \
    --secret-string "your-secure-api-key"
```

### 4. Network Security

- Use VPC with private subnets
- Implement security groups with minimal access
- Enable VPC Flow Logs
- Use AWS WAF for additional protection

## Troubleshooting

### Common Issues

#### 1. Model Loading Timeout

**Symptoms**: API startup takes too long or fails
**Solution**:
```bash
# Check model files
ls -la outputs/models/bertweet_enhanced/

# Increase timeout in docker-compose.yml
healthcheck:
  start_period: 120s  # Increase from 60s
```

#### 2. Database Connection Issues

**Symptoms**: 500 errors, connection refused
**Solution**:
```bash
# Check database connectivity
docker exec -it aicrisisalert-db psql -U postgres -d aicrisisalert -c "SELECT 1;"

# Check connection string
echo $DATABASE_URL
```

#### 3. Memory Issues

**Symptoms**: OOM kills, slow performance
**Solution**:
```bash
# Monitor memory usage
docker stats aicrisisalert-api

# Increase memory limits
docker-compose up -d --scale api=2  # Scale horizontally
```

#### 4. SSL Certificate Issues

**Symptoms**: SSL errors, certificate warnings
**Solution**:
```bash
# Check certificate validity
openssl x509 -in ssl/cert.pem -text -noout

# Renew Let's Encrypt certificate
sudo certbot renew --force-renewal
```

### Performance Tuning

#### 1. Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX CONCURRENTLY idx_classifications_created_category 
ON classifications(created_at, category);

-- Update statistics
ANALYZE;

-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

#### 2. API Optimization

```python
# Increase worker processes
# In docker-compose.yml
command: ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker"]
```

#### 3. Caching Strategy

```python
# Implement Redis caching
REDIS_URL = "redis://redis:6379"
CACHE_TTL = 3600  # 1 hour

# Cache model predictions for identical inputs
```

### Monitoring Commands

```bash
# System resources
htop
iostat -x 1
df -h

# Docker stats
docker stats
docker system df

# Application logs
docker-compose logs -f --tail=100 api

# Database performance
docker exec -it aicrisisalert-db psql -U postgres -c "
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation 
FROM pg_stats 
WHERE tablename = 'classifications';"
```

## Support

For deployment support:
- Check logs: `docker-compose logs -f`
- Monitor health: `curl http://localhost:8000/health`
- Review metrics: `http://localhost:9090` (Prometheus)
- Database status: Check PostgreSQL logs

Emergency contacts and escalation procedures should be documented separately for your organization.