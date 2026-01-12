"""
FastAPI application for financial risk classification
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, Any
import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.api.predict import predict_risk, load_pipeline
from src.api.monitoring import (
    log_prediction, record_request, get_performance_metrics,
    get_prediction_history, detect_data_drift, check_model_degradation,
    get_drift_status, load_training_statistics
)
from src.config import API_TITLE, API_DESCRIPTION, API_VERSION

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Middleware for performance monitoring
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Middleware to measure response time and record requests"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        success = response.status_code < 400
        process_time = time.time() - start_time
        
        # Record request metrics
        endpoint = f"{request.method} {request.url.path}"
        record_request(endpoint, process_time, success=success)
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        endpoint = f"{request.method} {request.url.path}"
        record_request(endpoint, process_time, success=False)
        raise

# Load pipeline and monitoring on startup
@app.on_event("startup")
async def startup_event():
    """Load pipeline and monitoring when API starts"""
    try:
        load_pipeline()
        load_training_statistics()  # Load training stats for drift detection
        print("✓ API started successfully")
        print("✓ Monitoring system initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not load pipeline: {e}")
        print("  Make sure to run train_pipeline.py first")


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for prediction"""
    # We'll use a flexible dict to accept any features
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of financial features (feature_name: value)"
    )


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    risk_level: str = Field(..., description="Predicted risk level: Low, Medium, or High")
    probabilities: Dict[str, float] = Field(..., description="Probability for each risk level")
    confidence: float = Field(..., description="Confidence score (highest probability)")


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    html_path = PROJECT_ROOT / "src" / "api" / "templates" / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>Financial Risk Classification API</title></head>
            <body>
                <h1>Financial Risk Classification API</h1>
                <p>API is running. Use <a href="/docs">/docs</a> for API documentation.</p>
                <p>Web interface not found. Please check templates/index.html</p>
            </body>
        </html>
        """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        pipeline, feature_names = load_pipeline()
        return {
            "status": "healthy",
            "pipeline_loaded": True,
            "num_features": len(feature_names)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "pipeline_loaded": False,
            "error": str(e)
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict financial risk level
    
    Accepts a dictionary of financial features and returns:
    - Predicted risk level (Low, Medium, High)
    - Probabilities for each risk level
    - Confidence score
    """
    start_time = time.time()
    
    try:
        result = predict_risk(request.features)
        response_time = time.time() - start_time
        
        # Log prediction for monitoring
        log_prediction(
            features=request.features,
            prediction=result['risk_level'],
            probabilities=result['probabilities'],
            confidence=result['confidence'],
            response_time=response_time
        )
        
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/features")
async def get_features():
    """Get list of required features"""
    try:
        _, feature_names = load_pipeline()
        return {
            "features": feature_names,
            "num_features": len(feature_names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading features: {str(e)}")


# Monitoring Endpoints
@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    try:
        metrics = get_performance_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")


@app.get("/predictions/history")
async def get_history(limit: int = 100):
    """
    Get prediction history
    
    Parameters:
    -----------
    limit: int
        Maximum number of predictions to return (default: 100)
    """
    try:
        history = get_prediction_history(limit=limit)
        return {
            "predictions": history,
            "count": len(history),
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")


@app.get("/monitoring/drift")
async def get_drift():
    """Get data drift detection status"""
    try:
        drift_status = get_drift_status()
        return drift_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking drift: {str(e)}")


@app.post("/monitoring/drift/check")
async def check_drift(request: PredictionRequest):
    """
    Check data drift for specific features
    
    Accepts features and returns drift detection results
    """
    try:
        drift_result = detect_data_drift(request.features)
        return drift_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking drift: {str(e)}")


@app.get("/monitoring/alerts")
async def get_alerts():
    """Get model degradation alerts"""
    try:
        alerts = check_model_degradation()
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)

