"""
Model training functions for Phase 4
Includes model definitions and GridSearchCV hyperparameter tuning
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    SPLITS_DATA_DIR, MODELS_DIR, RANDOM_SEED,
    CV_N_SPLITS, CV_N_REPEATS, CV_SCORING, HYPERPARAMETER_GRIDS
)

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_data():
    """
    Load preprocessed training and test data from data/splits/
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) as pandas DataFrames
    """
    print("Loading preprocessed data...")
    # Use absolute paths relative to project root
    splits_dir = PROJECT_ROOT / SPLITS_DATA_DIR
    X_train = pd.read_csv(splits_dir / "X_train.csv")
    X_test = pd.read_csv(splits_dir / "X_test.csv")
    # Read as DataFrame and squeeze to Series (for newer pandas versions)
    y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(splits_dir / "y_test.csv").squeeze()
    
    print(f"Training set: {len(X_train)} samples, {len(X_train.columns)} features")
    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def get_model_definitions():
    """
    Define all models to be trained
    
    Returns:
        dict: Dictionary mapping model names to sklearn model instances
    """
    models = {
        'logistic_regression': LogisticRegression(
            random_state=RANDOM_SEED, 
            max_iter=1000,
            solver='lbfgs'  # Default solver that supports multiclass
        ),
        'decision_tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
        'random_forest': RandomForestClassifier(random_state=RANDOM_SEED),
        'svm': SVC(random_state=RANDOM_SEED, probability=True),
        'naive_bayes': GaussianNB(),
        'knn': KNeighborsClassifier(),
        'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_SEED)
    }
    return models


def train_model_with_gridsearch(model_name, model, param_grid, X_train, y_train):
    """
    Train a single model using GridSearchCV with RepeatedStratifiedKFold
    
    Args:
        model_name: Name of the model
        model: sklearn model instance
        param_grid: Dictionary of hyperparameters to search
        X_train: Training features
        y_train: Training labels
    
    Returns:
        tuple: (best_model, best_params, cv_results, best_score)
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Create cross-validation strategy
    cv = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS,
        n_repeats=CV_N_REPEATS,
        random_state=RANDOM_SEED
    )
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV score ({CV_SCORING}): {best_score:.4f}")
    
    return best_model, best_params, grid_search.cv_results_, best_score


def train_all_models(X_train, y_train):
    """
    Train all models using GridSearchCV
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        dict: Dictionary mapping model names to (best_model, best_params, cv_results)
    """
    models = get_model_definitions()
    trained_models = {}
    
    # Ensure models directory exists (use absolute path)
    models_dir = PROJECT_ROOT / MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in models.items():
        param_grid = HYPERPARAMETER_GRIDS[model_name]
        
        best_model, best_params, cv_results, best_score = train_model_with_gridsearch(
            model_name, model, param_grid, X_train, y_train
        )
        
        # Save the trained model
        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(best_model, model_path)
        print(f"Saved model to {model_path}")
        
        trained_models[model_name] = {
            'model': best_model,
            'best_params': best_params,
            'cv_results': cv_results,
            'best_cv_score': best_score
        }
    
    return trained_models


def load_trained_model(model_name):
    """
    Load a trained model from disk
    
    Args:
        model_name: Name of the model
    
    Returns:
        Trained model instance
    """
    model_path = PROJECT_ROOT / MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train all models
    trained_models = train_all_models(X_train, y_train)
    
    print("\n" + "="*60)
    print("Training completed for all models!")
    print("="*60)

