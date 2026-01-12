"""
Prediction module for API
Handles loading pipeline and making predictions
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = PROJECT_ROOT / "artifacts" / "pipeline"

# Global variables for loaded pipeline and feature names
_pipeline = None
_feature_names = None


def load_pipeline():
    """
    Load the trained pipeline and feature names
    
    Returns:
    --------
    tuple: (pipeline, feature_names)
    """
    global _pipeline, _feature_names
    
    if _pipeline is None:
        pipeline_path = PIPELINE_DIR / "full_pipeline.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Pipeline not found at {pipeline_path}. "
                "Please run train_pipeline.py first."
            )
        
        _pipeline = joblib.load(pipeline_path)
        print(f"✓ Pipeline loaded from {pipeline_path}")
    
    if _feature_names is None:
        feature_names_path = PIPELINE_DIR / "feature_names.joblib"
        if not feature_names_path.exists():
            raise FileNotFoundError(
                f"Feature names not found at {feature_names_path}. "
                "Please run train_pipeline.py first."
            )
        
        _feature_names = joblib.load(feature_names_path)
        print(f"✓ Feature names loaded ({len(_feature_names)} features)")
    
    return _pipeline, _feature_names


def validate_features(input_data, feature_names):
    """
    Validate that input data contains all required features
    
    Parameters:
    -----------
    input_data: dict
        Dictionary with feature names as keys
    feature_names: list
        List of required feature names
        
    Returns:
    --------
    pd.DataFrame: Validated and ordered feature DataFrame
    """
    # Check if all features are present
    missing_features = set(feature_names) - set(input_data.keys())
    if missing_features:
        raise ValueError(
            f"Missing required features: {sorted(missing_features)}. "
            f"Required features: {sorted(feature_names)}"
        )
    
    # Check for extra features
    extra_features = set(input_data.keys()) - set(feature_names)
    if extra_features:
        print(f"Warning: Extra features provided (will be ignored): {sorted(extra_features)}")
    
    # Create DataFrame with features in correct order
    feature_dict = {feature: [input_data[feature]] for feature in feature_names}
    df = pd.DataFrame(feature_dict)
    
    # Validate data types (should be numeric)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            raise ValueError(f"Feature '{col}' must be numeric, got: {type(input_data[col])}")
    
    return df


def predict_risk(input_data):
    """
    Predict financial risk level for given input data
    
    Parameters:
    -----------
    input_data: dict
        Dictionary with feature names as keys and values
        
    Returns:
    --------
    dict: Dictionary with prediction and probabilities
        {
            'risk_level': str,  # 'Low', 'Medium', or 'High'
            'probabilities': dict,  # {'Low': 0.xx, 'Medium': 0.xx, 'High': 0.xx}
            'confidence': float  # Highest probability value
        }
    """
    # Load pipeline if not already loaded
    pipeline, feature_names = load_pipeline()
    
    # Validate and prepare input data
    X = validate_features(input_data, feature_names)
    
    # Make prediction
    prediction = pipeline.predict(X)[0]
    
    # Get probabilities
    probabilities = pipeline.predict_proba(X)[0]
    class_names = pipeline.classes_
    
    # Create probabilities dictionary
    prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
    
    # Get confidence (highest probability)
    confidence = float(max(probabilities))
    
    return {
        'risk_level': prediction,
        'probabilities': prob_dict,
        'confidence': confidence
    }

