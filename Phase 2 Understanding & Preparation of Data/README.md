# Phase 2 – Understanding & Preparation of Data

## Project Overview

**Title:** Intelligent Financial Risk Classification for Small Businesses Using Machine Learning

**Phase 2 Scope:** Data understanding, cleaning, feature preparation, and data splitting

## Quick Start

### Installation

```bash
# Create virtual environment (if not already created)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Phase 2 Pipeline


**Option 1: Jupyter Notebook (Interactive)**
```bash
# Open Jupyter notebook for interactive exploration
jupyter notebook Phase2_Data_Preparation.ipynb
```

This will:
1. Load the dataset
2. Create risk categories
3. Clean the data (missing values, duplicates, outliers)
4. Prepare features (remove redundancy, encode, scale)
5. Split into train/test sets
6. Save all outputs to appropriate directories

The Jupyter notebook includes visualizations and step-by-step exploration of the data.

## Project Structure

```
.
├── src/                    # Source code
│   ├── config.py          # Configuration constants
│   ├── data/              # Data processing modules
│   │   ├── load_data.py   # Data loading and risk categorization
│   │   ├── clean_data.py  # Missing values, duplicates, outliers
│   │   └── prepare_features.py  # Feature selection and scaling
│   └── pipeline/          # Pipeline modules
│       └── phase2_prepare_data.py  # Main entry point
├── data/
│   ├── raw/               # Raw dataset files
│   ├── processed/         # Cleaned dataset
│   └── splits/            # Train/test splits
├── artifacts/             # Preprocessing artifacts (scaler, feature list)
├── reports/               # Academic reports
└── requirements.txt       # Python dependencies
```

## Outputs

After running the pipeline, you'll find:

- **data/processed/cleaned_data.csv** - Complete cleaned dataset
- **data/splits/X_train.csv, X_test.csv** - Feature matrices
- **data/splits/y_train.csv, y_test.csv** - Target variables
- **artifacts/scaler.joblib** - Fitted StandardScaler
- **artifacts/feature_list.joblib** - List of selected features

## Dataset

The pipeline uses the **Polish Companies Bankruptcy Data** from UCI Machine Learning Repository. The dataset should be placed in `data/raw/` or in the project root directory as `polish+companies+bankruptcy+data/5year.arff`.

**Dataset URL:** https://archive.ics.uci.edu/ml/datasets/Polish+Companies+Bankruptcy+Data

## Configuration

Key settings can be modified in `src/config.py`:
- Train/test split ratio (default: 70/30)
- Random seed (default: 42)
- Correlation threshold for redundant features (default: 0.95)

## Academic Report

See `reports/academic_report.md` for the Phase 2 methodology summary.

---

**Author:** Tomás Silva
**Date:** 2025
