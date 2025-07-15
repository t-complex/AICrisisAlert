# API Troubleshooting Guide

This guide helps resolve common API issues when working with AICrisisAlert.

## 🚨 Common Issues

### 1. API Timeout Errors

**Symptoms:**
- "Request timed out" errors
- API takes too long to respond
- Connection timeouts after multiple retries

**Causes:**
- API running in full mode (loading ML model)
- Network connectivity issues
- Server resource constraints
- Model loading taking too long

**Solutions:**

#### Quick Fix - Use Simple Mode
```bash
# Stop current API
# Then start in simple mode
USE_SIMPLE_API=true python scripts/start_api.py
```

#### Check API Status
```bash
python scripts/check_api_status.py
```

#### Test with Increased Timeout
```bash
# Test with 30-second timeout
python scripts/check_api_status.py http://127.0.0.1:8000 30
```

### 2. API Not Running

**Symptoms:**
- "Connection refused" errors
- "API is not running" messages

**Solutions:**

#### Start the API
```bash
# Start in simple mode (recommended for testing)
python scripts/start_api.py

# Or start in full mode (requires trained model)
USE_SIMPLE_API=false python scripts/start_api.py
```

#### Check if Port is Available
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process using port 8000 (if needed)
kill -9 $(lsof -t -i:8000)
```

### 3. Model Loading Issues

**Symptoms:**
- API starts but classification fails
- "Model not found" errors
- Long startup times

**Solutions:**

#### Use Simple Mode for Testing
```bash
# Simple mode uses mock model - no ML dependencies
USE_SIMPLE_API=true python scripts/start_api.py
```

#### Check Model Files (Full Mode)
```bash
# Check if model files exist
ls -la outputs/models/

# If no model files, use simple mode or train a model first
```

### 4. Import Errors

**Symptoms:**
- Module import errors
- Missing dependencies

**Solutions:**

#### Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### Check Python Path
```bash
# Ensure you're in the project root
cd /path/to/AICrisisAlert

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 🔧 Testing Commands

### Basic API Testing
```bash
# Quick health check
curl http://127.0.0.1:8000/health

# Test classification
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT: People trapped in building collapse!"}'
```

### Comprehensive Testing
```bash
# Run all API tests
python scripts/test_api.py

# Check API status
python scripts/check_api_status.py
```

### Manual Testing
```bash
# Start API in one terminal
python scripts/start_api.py

# Test in another terminal
python scripts/check_api_status.py
```

## 📊 API Modes Explained

### Simple Mode (Default)
- **Purpose**: Testing and development
- **Model**: Mock classifier with keyword-based logic
- **Speed**: Fast startup and response
- **Dependencies**: Minimal (no ML libraries required)
- **Use Case**: API testing, development, demos

### Full Mode
- **Purpose**: Production with real ML model
- **Model**: Trained BERTweet transformer
- **Speed**: Slower startup (model loading)
- **Dependencies**: PyTorch, Transformers, etc.
- **Use Case**: Production deployment, real classification

## 🐛 Debugging Steps

### 1. Check API Status
```bash
python scripts/check_api_status.py
```

### 2. Check Server Logs
```bash
# Look for error messages in terminal where API is running
# Common issues:
# - Import errors
# - Model loading failures
# - Port conflicts
```

### 3. Test Individual Endpoints
```bash
# Health check
curl http://127.0.0.1:8000/health

# Model info
curl http://127.0.0.1:8000/model/info

# Classification
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}'
```

### 4. Check Environment
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(fastapi|uvicorn|torch|transformers)"

# Check if port is available
netstat -an | grep 8000
```

## 🚀 Performance Optimization

### For Development/Testing
```bash
# Use simple mode for fast testing
USE_SIMPLE_API=true python scripts/start_api.py
```

### For Production
```bash
# Use full mode with proper resources
USE_SIMPLE_API=false python scripts/start_api.py

# Or use Docker for isolation
docker-compose up -d
```

## 📞 Getting Help

### Check Documentation
- [README.md](../README.md) - Main project documentation
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) - Quick project overview

### Common Solutions
1. **Always use simple mode for testing**: `USE_SIMPLE_API=true`
2. **Check if API is running**: `python scripts/check_api_status.py`
3. **Use proper timeouts**: Increase timeout values for slow connections
4. **Check logs**: Look for error messages in the terminal

### For Claude Users
If you're using Claude to analyze this project:
1. **Start with simple mode**: `USE_SIMPLE_API=true python scripts/start_api.py`
2. **Test quickly**: `python scripts/check_api_status.py`
3. **Use the mock model**: It provides realistic responses without ML dependencies
4. **Check the overview**: Read `PROJECT_OVERVIEW.md` first for quick understanding

---

**Remember**: The API is designed to work in both simple and full modes. Use simple mode for testing and development, full mode for production with real ML models. 