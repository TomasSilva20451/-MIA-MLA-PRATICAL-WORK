"""
Configuration constants for Phase 2 data preparation
"""

# Data paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
SPLITS_DATA_DIR = "data/splits"
ARTIFACTS_DIR = "artifacts"

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

