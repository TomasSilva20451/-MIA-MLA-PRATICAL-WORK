"""
FastAPI application for financial risk classification
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, Any
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.api.predict import predict_risk, load_pipeline
from src.config import API_TITLE, API_DESCRIPTION, API_VERSION

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Load pipeline on startup
@app.on_event("startup")
async def startup_event():
    """Load pipeline when API starts"""
    try:
        load_pipeline()
        print("✓ API started successfully")
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
    try:
        result = predict_risk(request.features)
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


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)

