"""
ML Pipeline definition
Complete sklearn pipeline including preprocessing and model
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_SEED


def create_pipeline():
    """
    Create a complete ML pipeline with preprocessing and model
    
    Pipeline steps:
    1. SimpleImputer: Handle missing values (median strategy)
    2. StandardScaler: Normalize features
    3. RandomForestClassifier: Final model
    
    Returns:
    --------
    Pipeline: sklearn Pipeline object
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])
    
    return pipeline


def get_pipeline_steps():
    """
    Get information about pipeline steps
    
    Returns:
    --------
    dict: Dictionary with step names and descriptions
    """
    return {
        'imputer': 'SimpleImputer with median strategy for missing values',
        'scaler': 'StandardScaler for feature normalization',
        'classifier': 'RandomForestClassifier (n_estimators=100, max_depth=10)'
    }

