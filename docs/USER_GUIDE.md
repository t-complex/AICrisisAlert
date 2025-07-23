# AICrisisAlert User Guide

## Overview

AICrisisAlert is an AI-powered crisis management and alert system designed to classify social media content during emergency situations. This guide provides comprehensive instructions for end users, emergency responders, and system administrators.

## Table of Contents

- [Getting Started](#getting-started)
- [Web Interface](#web-interface)
- [API Usage](#api-usage)
- [Crisis Categories](#crisis-categories)
- [Best Practices](#best-practices)
- [Emergency Procedures](#emergency-procedures)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Getting Started

### System Access

**Web Interface**: https://your-domain.com/dashboard  
**API Endpoint**: https://your-domain.com/api  
**Documentation**: https://your-domain.com/docs

### User Roles

**Emergency Responder**:
- Real-time crisis monitoring
- Alert management and response
- Situation awareness dashboard

**System Administrator**:
- User management
- System configuration
- Performance monitoring

**API User**:
- Programmatic access to classification
- Integration with external systems
- Automated crisis monitoring

### Initial Setup

1. **Request Access**: Contact your system administrator for account creation
2. **Receive Credentials**: You'll receive login credentials and API keys
3. **Login**: Access the web interface at the provided URL
4. **API Configuration**: Configure your applications with provided API keys

## Web Interface

### Dashboard Overview

The main dashboard provides real-time crisis monitoring capabilities:

```
┌─────────────────────────────────────────────────────────────┐
│                    AICrisisAlert Dashboard                  │
├─────────────────────────────────────────────────────────────┤
│ 🚨 Active Alerts: 3        📊 Classifications Today: 1,247  │
│ ⚡ Urgent Help: 12         🏗️ Infrastructure: 8             │
│ 🚑 Casualties: 2          📍 Resource Requests: 15         │
├─────────────────────────────────────────────────────────────┤
│                      Recent Classifications                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 🚨 URGENT: "People trapped in collapsed building!"     │ │
│ │    Confidence: 94% | Source: Twitter | 2 min ago      │ │
│ │    Location: Downtown Miami | Status: RESPONDING       │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ 🏗️ INFRASTRUCTURE: "Power lines down on Main St"      │ │
│ │    Confidence: 87% | Source: Facebook | 5 min ago     │ │
│ │    Location: Main Street | Status: REPORTED           │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      Quick Actions                          │
│ [Classify Text] [View Map] [Generate Report] [Settings]    │
└─────────────────────────────────────────────────────────────┘
```

### Text Classification Interface

**Step 1: Input Text**
```
┌─────────────────────────────────────────┐
│             Classify Crisis Text        │
├─────────────────────────────────────────┤
│ Text to Classify:                       │
│ ┌─────────────────────────────────────┐ │
│ │ URGENT: People trapped in building  │ │
│ │ collapse on 5th Street! Need help   │ │
│ │ immediately!                        │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ Source (optional): [Twitter    ▼]      │
│ Location (optional): Miami, FL          │
│                                         │
│ [Classify Text]                         │
└─────────────────────────────────────────┘
```

**Step 2: Review Results**
```
┌─────────────────────────────────────────┐
│           Classification Result          │
├─────────────────────────────────────────┤
│ 🚨 PREDICTED CATEGORY: Urgent Help     │
│ 📊 CONFIDENCE: 94%                     │
│ ⏱️ PROCESSING TIME: 150ms              │
│                                         │
│ Category Probabilities:                 │
│ ▓▓▓▓▓▓▓▓▓▓ Urgent Help: 94%            │
│ ▓▓         Infrastructure: 4%           │
│ ▓          Casualty Info: 2%            │
│ ▓          Resources: 1%                │
│ ▓          General: 0%                  │
│                                         │
│ [Create Alert] [Save] [Classify Another]│
└─────────────────────────────────────────┘
```

### Alert Management

**Alert Dashboard**:
```
┌─────────────────────────────────────────────────────────────┐
│                     Active Alerts                          │
├─────────────────────────────────────────────────────────────┤
│ Alert ID  │ Type        │ Severity │ Location      │ Status │
├─────────────────────────────────────────────────────────────┤
│ #001      │ Urgent Help │ Critical │ Downtown      │ Active │
│ #002      │ Infrastructure│ High   │ Main St       │ Active │
│ #003      │ Casualties  │ High     │ 5th Ave       │ Resolved│
├─────────────────────────────────────────────────────────────┤
│ [View Details] [Acknowledge] [Resolve] [Export]            │
└─────────────────────────────────────────────────────────────┘
```

**Alert Details**:
```
┌─────────────────────────────────────────┐
│              Alert #001                 │
├─────────────────────────────────────────┤
│ Type: Urgent Help Request               │
│ Severity: Critical                      │
│ Created: 2024-01-15 14:30:00           │
│ Location: Downtown Miami                │
│ Confidence: 94%                         │
│                                         │
│ Original Text:                          │
│ "URGENT: People trapped in building     │
│  collapse on 5th Street! Need help     │
│  immediately!"                          │
│                                         │
│ Response Actions:                       │
│ ☐ Dispatch emergency services          │
│ ☐ Notify local authorities             │
│ ☐ Update status board                  │
│                                         │
│ [Acknowledge] [Assign] [Add Notes]      │
└─────────────────────────────────────────┘
```

### Map Visualization

The system provides geographic visualization of crisis events:

**Features**:
- Real-time crisis location mapping
- Severity-based color coding
- Cluster analysis for related incidents
- Heat maps for crisis density
- Resource allocation visualization

**Map Legend**:
- 🔴 Critical/Urgent (Red)
- 🟠 High Severity (Orange)
- 🟡 Medium Severity (Yellow)
- 🟢 Low Severity (Green)
- ⚫ Resolved (Gray)

## API Usage

### Authentication

All API requests require authentication using Bearer tokens:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://your-domain.com/api/classify
```

### Basic Classification

**Single Text Classification**:
```bash
curl -X POST "https://your-domain.com/api/classify" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "URGENT: People trapped in building collapse!",
    "source": "twitter",
    "location": "Miami, FL"
  }'
```

**Response**:
```json
{
  "predicted_class": "urgent_help",
  "confidence": 0.94,
  "class_probabilities": {
    "urgent_help": 0.94,
    "infrastructure_damage": 0.04,
    "casualty_info": 0.02,
    "resource_availability": 0.01,
    "general_info": 0.00
  },
  "processing_time_ms": 150.5,
  "model_version": "1.0.0",
  "features_used": ["urgency_keywords", "location_indicators"]
}
```

### Batch Classification

For processing multiple texts efficiently:

```bash
curl -X POST "https://your-domain.com/api/classify/batch" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "URGENT: People trapped in building collapse!",
      "Power outages reported across downtown area",
      "Volunteers needed at Red Cross shelter"
    ]
  }'
```

### Emergency Classification

For high-priority situations (no rate limiting):

```bash
curl -X POST "https://your-domain.com/api/classify/emergency" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "EMERGENCY: Building fire with people inside!"
  }'
```

### Python SDK Example

```python
import requests
import json

class CrisisAlertAPI:
    def __init__(self, api_key, base_url="https://your-domain.com/api"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def classify_text(self, text, source=None, location=None):
        """Classify a single text."""
        data = {"text": text}
        if source:
            data["source"] = source
        if location:
            data["location"] = location
        
        response = requests.post(
            f"{self.base_url}/classify",
            headers=self.headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def classify_batch(self, texts):
        """Classify multiple texts."""
        response = requests.post(
            f"{self.base_url}/classify/batch",
            headers=self.headers,
            json={"texts": texts}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

# Usage
api = CrisisAlertAPI(api_key="your-api-key-here")

# Single classification
result = api.classify_text(
    "URGENT: People trapped in building collapse!",
    source="twitter",
    location="Miami, FL"
)

print(f"Category: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch classification
results = api.classify_batch([
    "Power outages downtown",
    "Volunteers needed at shelter",
    "URGENT: Medical help required!"
])

for i, result in enumerate(results['results']):
    print(f"Text {i+1}: {result['predicted_class']} ({result['confidence']:.2%})")
```

## Crisis Categories

### Understanding Categories

**1. Urgent Help (urgent_help)**
- **Description**: Immediate assistance requests requiring emergency response
- **Examples**:
  - "URGENT: People trapped in collapsed building!"
  - "Help! Can't evacuate, need rescue immediately!"
  - "Emergency: Medical assistance needed NOW!"
- **Response Priority**: Highest (immediate dispatch)
- **Typical Confidence**: 85-95%

**2. Infrastructure Damage (infrastructure_damage)**
- **Description**: Reports of damage to critical infrastructure
- **Examples**:
  - "Power lines down on Main Street"
  - "Bridge collapsed blocking highway 95"
  - "Water main break flooding downtown"
- **Response Priority**: High (assess and repair)
- **Typical Confidence**: 80-90%

**3. Casualty Information (casualty_info)**
- **Description**: Information about injuries, casualties, or missing persons
- **Examples**:
  - "Multiple injuries reported at intersection"
  - "Missing person: John Doe, last seen downtown"
  - "Casualties from building collapse"
- **Response Priority**: High (medical response)
- **Typical Confidence**: 75-85%

**4. Resource Availability (resource_availability)**
- **Description**: Information about available resources, volunteers, or aid
- **Examples**:
  - "Red Cross shelter open at community center"
  - "Volunteers needed for cleanup efforts"
  - "Food and water available at central park"
- **Response Priority**: Medium (coordinate resources)
- **Typical Confidence**: 70-85%

**5. General Information (general_info)**
- **Description**: General crisis-related information and updates
- **Examples**:
  - "Hurricane expected to make landfall tonight"
  - "Evacuation routes remain open"
  - "Weather service issues flood warning"
- **Response Priority**: Low (situational awareness)
- **Typical Confidence**: 60-80%

### Confidence Levels

**Confidence Interpretation**:
- **90-100%**: Very high confidence, immediate action recommended
- **80-89%**: High confidence, priority action
- **70-79%**: Good confidence, standard processing
- **60-69%**: Moderate confidence, human review recommended
- **Below 60%**: Low confidence, manual verification required

## Best Practices

### For Emergency Responders

**1. Alert Prioritization**
```
Priority Level 1 (Critical):
- Urgent help requests with >90% confidence
- Multiple casualty reports
- Infrastructure collapse with people involved

Priority Level 2 (High):
- Infrastructure damage affecting safety
- Casualty information with >80% confidence
- Resource coordination for ongoing incidents

Priority Level 3 (Medium):
- General infrastructure reports
- Resource availability updates
- Lower confidence urgent requests (human review)

Priority Level 4 (Low):
- General information
- Weather updates
- Routine status reports
```

**2. Response Workflow**
```
1. Alert Received
   ↓
2. Verify Confidence Level
   ↓
3. Check Geographic Relevance
   ↓
4. Cross-reference Existing Incidents
   ↓
5. Dispatch Appropriate Resources
   ↓
6. Update Status and Documentation
   ↓
7. Monitor for Related Incidents
```

**3. Quality Assurance**
- Always verify high-impact decisions with human judgment
- Cross-reference AI classifications with other sources
- Document cases where AI classification was incorrect
- Provide feedback to improve system accuracy

### For API Users

**1. Rate Limiting Management**
```python
# Implement exponential backoff for rate limits
import time
from random import uniform

def classify_with_retry(api, text, max_retries=3):
    """Classify text with automatic retry on rate limits."""
    for attempt in range(max_retries):
        try:
            return api.classify_text(text)
        except Exception as e:
            if "429" in str(e):  # Rate limit exceeded
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = (2 ** attempt) + uniform(0, 1)
                    time.sleep(delay)
                    continue
            raise e
    
    raise Exception("Max retries exceeded")
```

**2. Error Handling**
```python
def robust_classification(api, text):
    """Robust classification with comprehensive error handling."""
    try:
        result = api.classify_text(text)
        
        # Validate result
        if result['confidence'] < 0.6:
            print(f"Warning: Low confidence ({result['confidence']:.2%})")
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to API")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timeout")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
```

**3. Batch Processing Optimization**
```python
def process_large_dataset(api, texts, batch_size=50):
    """Process large datasets efficiently."""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            batch_results = api.classify_batch(batch)
            results.extend(batch_results['results'])
            
            # Rate limiting courtesy delay
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Process individually as fallback
            for text in batch:
                try:
                    result = api.classify_text(text)
                    results.append(result)
                except Exception as single_error:
                    print(f"Error processing single text: {single_error}")
                    results.append(None)
    
    return results
```

## Emergency Procedures

### High-Priority Alert Workflow

**When Urgent Help Alert is Received**:

1. **Immediate Assessment** (0-2 minutes)
   - Review alert details and confidence level
   - Check location and accessibility
   - Verify against existing incidents

2. **Resource Dispatch** (2-5 minutes)
   - Alert emergency services (911/local equivalent)
   - Dispatch closest available units
   - Notify hospital/medical facilities if needed

3. **Situation Monitoring** (Ongoing)
   - Monitor for related incidents in area
   - Track resource deployment status
   - Update stakeholders and command center

4. **Documentation** (Throughout process)
   - Log all actions taken
   - Document response times
   - Note any classification accuracy issues

### System Failure Procedures

**If AICrisisAlert System is Down**:

1. **Immediate Actions**
   - Switch to backup monitoring systems
   - Notify IT support team
   - Activate manual monitoring procedures

2. **Backup Procedures**
   - Use alternative crisis monitoring tools
   - Increase human monitoring of social media
   - Implement manual alert processing

3. **Communication**
   - Notify all stakeholders of system status
   - Provide regular updates on restoration
   - Document impact and lessons learned

### False Positive Management

**When AI Classification Appears Incorrect**:

1. **Immediate Response**
   - Do not ignore without human verification
   - Quickly assess actual situation
   - If false positive confirmed, log details

2. **Follow-up Actions**
   - Document false positive case
   - Analyze why classification was incorrect
   - Provide feedback to system administrators

3. **System Improvement**
   - Report classification errors
   - Contribute to model retraining data
   - Suggest improvements to classification logic

## Troubleshooting

### Common Issues and Solutions

**Issue**: API returns 401 Unauthorized
```
Solution:
1. Verify API key is correct
2. Check if API key has expired
3. Ensure Bearer token format: "Bearer YOUR_API_KEY"
4. Contact administrator for new credentials
```

**Issue**: API returns 429 Rate Limit Exceeded
```
Solution:
1. Implement exponential backoff in your code
2. Reduce request frequency
3. Use batch endpoints for multiple classifications
4. Contact administrator to discuss rate limit increase
```

**Issue**: Low confidence scores consistently
```
Solution:
1. Review text quality and length
2. Ensure text is in English
3. Check for unusual characters or formatting
4. Consider providing additional context (source, location)
5. Report systematic issues to administrators
```

**Issue**: Slow API response times
```
Solution:
1. Check network connectivity
2. Use geographically closer API endpoint if available
3. Consider using batch processing for multiple items
4. Report performance issues to support team
```

**Issue**: Web interface not loading
```
Solution:
1. Clear browser cache and cookies
2. Try different browser or incognito mode
3. Check internet connection
4. Verify you're using correct URL
5. Contact IT support if issue persists
```

### Diagnostic Commands

**Check API Health**:
```bash
curl -X GET "https://your-domain.com/api/health"
```

**Test Authentication**:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "https://your-domain.com/api/model/info"
```

**Performance Test**:
```bash
time curl -X POST "https://your-domain.com/api/classify" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test emergency message"}'
```

## FAQ

**Q: How accurate is the AI classification?**
A: The system achieves approximately 85-90% accuracy on crisis classification tasks. Confidence scores help indicate reliability of individual predictions.

**Q: Can I integrate AICrisisAlert with my existing systems?**
A: Yes, the API is designed for integration. We provide REST endpoints, Python SDK, and can assist with custom integrations.

**Q: What happens if the AI classifies something incorrectly?**
A: Human oversight is always recommended for critical decisions. Report incorrect classifications to help improve the system.

**Q: How quickly does the system process new text?**
A: Typical response times are 150-300ms for single classifications, with batch processing providing better throughput for multiple items.

**Q: Is my data secure when using the API?**
A: Yes, all communications use HTTPS encryption, and we follow industry-standard security practices. See our security documentation for details.

**Q: Can the system handle non-English text?**
A: Currently, the system is optimized for English text. Support for other languages may be added in future versions.

**Q: What should I do during a major emergency event?**
A: Use the emergency classification endpoint for highest priority items, monitor the dashboard for real-time updates, and follow your organization's emergency procedures.

**Q: How often is the AI model updated?**
A: The model is retrained periodically with new data. System administrators will notify users of significant updates.

**Q: Can I customize the crisis categories for my organization?**
A: Custom categories may be possible with enterprise deployments. Contact your administrator to discuss requirements.

**Q: What backup procedures exist if the system goes down?**
A: Your organization should have documented backup procedures. Ensure you're familiar with alternative monitoring methods and escalation procedures.

---

For additional support:
- **Technical Issues**: Contact IT Support
- **User Training**: Schedule training session with administrators  
- **Feature Requests**: Submit through your organization's feedback process
- **Emergency**: Follow your organization's emergency contact procedures

This user guide serves as a comprehensive reference for effective use of the AICrisisAlert system. Regular training and practice with the system will improve response effectiveness during actual emergency situations.