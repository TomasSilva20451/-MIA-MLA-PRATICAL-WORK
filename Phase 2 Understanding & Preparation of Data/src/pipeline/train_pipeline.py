"""
Train complete ML pipeline
Includes preprocessing and model training
"""

import pandas as pd
import joblib
from pathlib import Path
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.pipeline.ml_pipeline import create_pipeline
from src.config import SPLITS_DATA_DIR, RANDOM_SEED

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = PROJECT_ROOT / "artifacts" / "pipeline"
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data():
    """
    Load training data from data/splits/
    
    Returns:
    --------
    tuple: (X_train, y_train) as pandas DataFrames
    """
    print("Loading training data...")
    splits_dir = PROJECT_ROOT / SPLITS_DATA_DIR
    X_train = pd.read_csv(splits_dir / "X_train.csv")
    y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze()
    
    print(f"Training set: {len(X_train)} samples, {len(X_train.columns)} features")
    print(f"Class distribution: {y_train.value_counts().to_dict()}")
    
    return X_train, y_train


def train_pipeline():
    """
    Train complete ML pipeline and save it
    
    Returns:
    --------
    Pipeline: Trained pipeline
    """
    # Load data
    X_train, y_train = load_training_data()
    
    # Create pipeline
    print("\nCreating ML pipeline...")
    pipeline = create_pipeline()
    print("Pipeline steps:")
    for step_name, step in pipeline.steps:
        print(f"  - {step_name}: {type(step).__name__}")
    
    # Train pipeline
    print("\nTraining pipeline...")
    pipeline.fit(X_train, y_train)
    print("✓ Pipeline trained successfully")
    
    # Evaluate on training set
    train_score = pipeline.score(X_train, y_train)
    print(f"Training accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
    
    # Save pipeline
    pipeline_path = PIPELINE_DIR / "full_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"✓ Pipeline saved to {pipeline_path}")
    
    # Also save feature names for validation in production
    feature_names_path = PIPELINE_DIR / "feature_names.joblib"
    joblib.dump(X_train.columns.tolist(), feature_names_path)
    print(f"✓ Feature names saved to {feature_names_path}")
    
    return pipeline


def validate_pipeline(pipeline):
    """
    Validate pipeline on test set
    
    Parameters:
    -----------
    pipeline: Pipeline
        Trained pipeline
    """
    print("\nValidating pipeline on test set...")
    splits_dir = PROJECT_ROOT / SPLITS_DATA_DIR
    X_test = pd.read_csv(splits_dir / "X_test.csv")
    y_test = pd.read_csv(splits_dir / "y_test.csv").squeeze()
    
    test_score = pipeline.score(X_test, y_test)
    print(f"Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
    
    return test_score


if __name__ == "__main__":
    print("="*60)
    print("Training Complete ML Pipeline")
    print("="*60)
    
    # Train pipeline
    pipeline = train_pipeline()
    
    # Validate on test set
    validate_pipeline(pipeline)
    
    print("\n" + "="*60)
    print("Pipeline training complete!")
    print("="*60)
    print(f"\nPipeline saved to: {PIPELINE_DIR / 'full_pipeline.joblib'}")
    print("You can now use this pipeline in production via the API.")

