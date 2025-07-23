# AICrisisAlert API Reference

**Version**: 1.0.0  
**Base URL**: `http://localhost:8000`  
**Authentication**: Bearer Token (API Key)

## Overview

The AICrisisAlert API provides real-time crisis classification for social media content during emergency situations. The API analyzes text inputs and classifies them into actionable crisis categories to support emergency response coordination.

## Authentication

All API endpoints (except `/health` and `/`) require authentication using Bearer tokens.

### Request Headers

```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

### Getting an API Key

API keys must be configured via environment variables. Contact your system administrator for access credentials.

## Rate Limiting

- **Limit**: 100 requests per hour per API key
- **Headers**: Rate limit information is returned in response headers
- **Emergency Endpoint**: No rate limiting applied to `/classify/emergency`

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Crisis Categories

The API classifies text into the following categories:

| Category | Description | Use Case |
|----------|-------------|----------|
| `urgent_help` | Immediate assistance requests | Emergency rescue coordination |
| `infrastructure_damage` | Infrastructure status reports | Resource allocation planning |
| `casualty_info` | Injury/casualty information | Medical response coordination |
| `resource_availability` | Available resources/volunteers | Community response coordination |
| `general_info` | General crisis information | Situational awareness |

## Endpoints

### Health Check

Check API service health and database connectivity.

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600.5
}
```

### Model Information

Get information about the loaded crisis classification model.

```http
GET /model/info
```

**Headers**: `Authorization: Bearer YOUR_API_KEY`

**Response**:
```json
{
  "model_name": "bertweet-crisis-classifier",
  "model_version": "1.0.0",
  "model_type": "bertweet",
  "classes": ["urgent_help", "infrastructure_damage", "casualty_info", "resource_availability", "general_info"],
  "accuracy": 0.84,
  "f1_score": 0.78,
  "loaded_at": "2024-01-15T10:00:00Z",
  "last_updated": "2024-01-15T09:30:00Z"
}
```

### Single Text Classification

Classify a single text input for crisis classification.

```http
POST /classify
```

**Headers**: 
- `Authorization: Bearer YOUR_API_KEY`
- `Content-Type: application/json`

**Request Body**:
```json
{
  "text": "URGENT: People trapped in building collapse!",
  "source": "twitter",
  "location": "Miami, FL",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Parameters**:
- `text` (required): Text to classify (3-10,000 characters)
- `source` (optional): Source platform (alphanumeric, underscore, hyphen only)
- `location` (optional): Geographic location (max 200 characters)
- `timestamp` (optional): ISO 8601 timestamp

**Response**:
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
  "model_version": "1.0.0",
  "features_used": ["text_length", "urgency_keywords", "location_indicators"]
}
```

### Batch Classification

Classify multiple text inputs in a single request.

```http
POST /classify/batch
```

**Headers**: 
- `Authorization: Bearer YOUR_API_KEY`
- `Content-Type: application/json`

**Request Body**:
```json
{
  "texts": [
    "URGENT: People trapped in building collapse!",
    "Power outages reported across downtown area"
  ],
  "sources": ["twitter", "facebook"],
  "locations": ["Miami, FL", "Orlando, FL"]
}
```

**Parameters**:
- `texts` (required): Array of texts to classify (max 100 items)
- `sources` (optional): Array of source platforms for each text
- `locations` (optional): Array of locations for each text

**Response**:
```json
{
  "results": [
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
  ],
  "total_processing_time_ms": 300.2,
  "batch_size": 2
}
```

### Emergency Classification

High-priority classification endpoint with no rate limiting for urgent situations.

```http
POST /classify/emergency
```

**Headers**: 
- `Authorization: Bearer YOUR_API_KEY`
- `Content-Type: application/json`

**Request Body**: Same as `/classify`

**Response**: Same as `/classify`

**Note**: This endpoint bypasses rate limiting and receives priority processing for urgent crisis situations.

## Error Handling

### Error Response Format

```json
{
  "error": "Classification failed",
  "detail": "Model not loaded",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456"
}
```

### HTTP Status Codes

| Code | Description | Possible Causes |
|------|-------------|-----------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid input data, malformed JSON |
| 401 | Unauthorized | Missing or invalid API key |
| 422 | Validation Error | Input validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side processing error |

### Common Error Scenarios

#### Authentication Errors

```json
{
  "error": "Invalid API key",
  "status_code": 401,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Rate Limit Exceeded

```json
{
  "error": "Rate limit exceeded. Maximum 100 requests per 3600 seconds.",
  "status_code": 429,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Validation Errors

```json
{
  "error": "Validation failed",
  "detail": [
    {
      "loc": ["text"],
      "msg": "Text must be at least 3 characters long",
      "type": "value_error"
    }
  ],
  "status_code": 422,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Request Examples

### cURL Examples

#### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

#### Single Classification
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "URGENT: People trapped in building collapse!",
    "source": "twitter",
    "location": "Miami, FL"
  }'
```

#### Batch Classification
```bash
curl -X POST "http://localhost:8000/classify/batch" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "URGENT: People trapped in building collapse!",
      "Power outages reported across downtown area"
    ]
  }'
```

### Python Examples

#### Using requests library

```python
import requests

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Single classification
response = requests.post(
    f"{API_BASE_URL}/classify",
    headers=headers,
    json={
        "text": "URGENT: People trapped in building collapse!",
        "source": "twitter",
        "location": "Miami, FL"
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

#### Using aiohttp for async requests

```python
import aiohttp
import asyncio

async def classify_text(session, text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with session.post(
        f"{API_BASE_URL}/classify",
        headers=headers,
        json={"text": text}
    ) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        result = await classify_text(session, "URGENT: Need immediate help!")
        print(result)

asyncio.run(main())
```

## Response Times

| Endpoint | Typical Response Time | Max Response Time |
|----------|----------------------|-------------------|
| `/health` | < 10ms | 100ms |
| `/classify` | 150-300ms | 2s |
| `/classify/batch` | 200ms/item | 5s |
| `/classify/emergency` | < 200ms | 1s |

## Security Considerations

### Input Validation

- Text inputs are sanitized for XSS and injection attacks
- Maximum text length: 10,000 characters
- Source and location fields are validated with regex patterns
- Excessive repeated characters are rejected

### API Security

- All endpoints require authentication (except health check)
- Rate limiting prevents abuse and DoS attacks
- CORS is configured with explicit allowed origins
- Request logging includes security audit trails

### Best Practices

1. **Store API keys securely** - Never commit API keys to version control
2. **Use HTTPS in production** - Always encrypt API communications
3. **Implement retry logic** - Handle temporary failures gracefully
4. **Monitor rate limits** - Track usage to avoid hitting limits
5. **Validate responses** - Always check HTTP status codes and error messages

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **JSON Schema**: http://localhost:8000/openapi.json

## Support

For API support and issues:
- **Documentation**: See `/docs` directory
- **Health Check**: Monitor `/health` endpoint
- **Logs**: Check application logs for detailed error information
- **Issues**: Report bugs via project issue tracker