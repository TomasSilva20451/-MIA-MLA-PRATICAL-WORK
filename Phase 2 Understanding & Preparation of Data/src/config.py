"""
Configuration constants for Phase 2 data preparation and Phase 4 model training
"""

# Data paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
SPLITS_DATA_DIR = "data/splits"
ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "artifacts/models"
RESULTS_DIR = "artifacts/results"
VISUALIZATIONS_DIR = "artifacts/visualizations"
PIPELINE_DIR = "artifacts/pipeline"

# Dataset configuration
DATASET_NAME = "polish_companies_bankruptcy"
DATASET_URL = "https://archive.ics.uci.edu/ml/datasets/Polish+Companies+Bankruptcy+Data"

# Data splitting
TRAIN_SIZE = 0.7
TEST_SIZE = 0.3
RANDOM_SEED = 42

# Feature selection
CORRELATION_THRESHOLD = 0.95  # Remove features with correlation > this value

# Preprocessing
OUTLIER_PERCENTILE_LOW = 1
OUTLIER_PERCENTILE_HIGH = 99

# Phase 4: Model Training Configuration
# Cross-validation parameters
CV_N_SPLITS = 5
CV_N_REPEATS = 3
CV_SCORING = 'accuracy'

# Hyperparameter grids for each model
HYPERPARAMETER_GRIDS = {
    'logistic_regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],  # l2 penalty for multiclass
        'solver': ['lbfgs', 'sag']  # Solvers that support multiclass
    },
    'decision_tree': {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear']
    },
    'naive_bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    'gradient_boosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
}

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Financial Risk Classification API"
API_DESCRIPTION = "API for predicting financial risk levels of small businesses"
API_VERSION = "1.0.0"

