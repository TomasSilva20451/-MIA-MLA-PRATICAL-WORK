# API Usage Guide

## Financial Risk Classification API

This API provides endpoints for predicting financial risk levels of small businesses using the trained Random Forest model.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Health Check

Check if the API is running and the pipeline is loaded.

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "pipeline_loaded": true,
  "num_features": 53
}
```

### 2. Get Required Features

Get the list of features required for prediction.

**Request:**
```http
GET /features
```

**Response:**
```json
{
  "features": [
    "Attr1",
    "Attr2",
    "Attr3",
    ...
    "_risk_score"
  ],
  "num_features": 53
}
```

### 3. Predict Risk Level

Predict the financial risk level for a company based on financial features.

**Request:**
```http
POST /predict
Content-Type: application/json

{
  "features": {
    "Attr1": 0.0,
    "Attr2": 0.0,
    "Attr3": 0.0,
    "Attr4": -0.2,
    "Attr5": 0.1,
    ...
    "_risk_score": -0.2
  }
}
```

**Response:**
```json
{
  "risk_level": "Low",
  "probabilities": {
    "Low": 0.85,
    "Medium": 0.10,
    "High": 0.05
  },
  "confidence": 0.85
}
```

**Response Fields:**
- `risk_level`: Predicted risk level (`Low`, `Medium`, or `High`)
- `probabilities`: Dictionary with probability for each risk level
- `confidence`: Highest probability value (confidence in prediction)

**Error Responses:**

Missing features (400):
```json
{
  "detail": "Missing required features: ['Attr1', 'Attr2']. Required features: [...]"
}
```

Invalid data type (400):
```json
{
  "detail": "Feature 'Attr1' must be numeric, got: <class 'str'>"
}
```

## Usage Examples

### Python

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Example features (all 53 features required)
features = {
    "Attr1": 0.0,
    "Attr2": 0.0,
    "Attr3": 0.0,
    "Attr4": -0.2,
    "Attr5": 0.1,
    # ... (all 53 features)
    "_risk_score": -0.2
}

# Make prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"features": features}
)

if response.status_code == 200:
    result = response.json()
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("Probabilities:")
    for level, prob in result['probabilities'].items():
        print(f"  {level}: {prob:.2%}")
else:
    print(f"Error: {response.json()}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Attr1": 0.0,
      "Attr2": 0.0,
      ...
      "_risk_score": -0.2
    }
  }'
```

### JavaScript (Fetch API)

```javascript
const features = {
  "Attr1": 0.0,
  "Attr2": 0.0,
  // ... all 53 features
  "_risk_score": -0.2
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ features: features })
})
.then(response => response.json())
.then(data => {
  console.log('Risk Level:', data.risk_level);
  console.log('Confidence:', data.confidence);
  console.log('Probabilities:', data.probabilities);
})
.catch(error => console.error('Error:', error));
```

## Web Interface

The API includes a web interface accessible at:

```
http://localhost:8000
```

The web interface allows you to:
- Enter financial features manually
- Load sample data
- Get predictions with visual feedback
- View probabilities for all risk levels

## API Documentation

Interactive API documentation (Swagger UI) is available at:

```
http://localhost:8000/docs
```

Alternative documentation (ReDoc) is available at:

```
http://localhost:8000/redoc
```

## Starting the API

1. Make sure the pipeline is trained:
   ```bash
   python -m src.pipeline.train_pipeline
   ```

2. Start the API server:
   ```bash
   uvicorn src.api.app:app --reload
   ```

   Or using Python:
   ```bash
   python -m src.api.app
   ```

The API will be available at `http://localhost:8000`

## Notes

- All 53 features must be provided for prediction
- Feature values should be numeric (float or int)
- The features should be in the same format as the training data (already scaled/normalized)
- Missing features will result in a 400 error
- The API uses the trained Random Forest pipeline which includes preprocessing steps

