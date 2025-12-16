# Intelligent Financial Risk Classification for Small Businesses Using Machine Learning

## Project Overview

This project develops and evaluates a machine learning classification model to identify financial risk levels of small businesses using historical financial and economic indicators. The goal is to support early and informed decision-making for financial risk assessment.

**Problem Type:** Supervised Classification  
**Target:** 3-Class Financial Risk Classification (Low, Medium, High)  
**Framework:** MLA – MAAI Practical Work

## Objective

Develop and evaluate a machine learning classification model that identifies financial risk levels of small businesses using historical financial and economic indicators to support early and informed decision making.

## Dataset

**Source:** Polish Companies Bankruptcy Data from UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/ml/datasets/Polish+Companies+Bankruptcy+Data

- **Size:** 5,910 companies with 64 financial ratios
- **Time Period:** 2000-2004 (5 years)
- **Features:** Financial ratios covering liquidity, solvency, profitability, and efficiency metrics
- **Target:** Binary bankruptcy classification (adapted to 3-class risk levels)

## Project Status

- ✅ **Phase 1:** Project Definition and Planning
- ✅ **Phase 2:** Understanding & Preparation of Data (Current)
- ⏳ **Phase 3:** Model Selection and Training (Upcoming)
- ⏳ **Phase 4:** Model Evaluation and Validation (Upcoming)
- ⏳ **Phase 5:** Model Deployment and Monitoring (Upcoming)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/TomasSilva20451/-MIA-MLA-PRATICAL-WORK.git
cd "Phase 2 Understanding & Preparation of Data"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Phase 2 Data Preparation

**Jupyter Notebook (Interactive)**
```bash
jupyter notebook Phase2_Data_Preparation.ipynb
```

The notebook will guide you through:
1. Load the dataset
2. Create 3-class risk categories (Low, Medium, High)
3. Clean the data (handle missing values, remove duplicates, treat outliers)
4. Prepare features (remove redundancy, encode, scale)
5. Split into train/test sets (70/30, stratified)
6. Save all outputs to appropriate directories

## Project Structure

```
.
├── src/                           # Source code
│   ├── config.py                  # Configuration constants
│   └── data/                      # Data processing modules
│       ├── load_data.py           # Data loading and risk categorization
│       ├── clean_data.py          # Missing values, duplicates, outliers
│       └── prepare_features.py    # Feature selection and scaling
├── data/
│   ├── raw/                       # Raw dataset files
│   ├── processed/                 # Cleaned dataset
│   └── splits/                    # Train/test splits
├── artifacts/                     # Preprocessing artifacts (scaler, feature list)
├── reports/                       # Academic reports
├── Phase2_Data_Preparation.ipynb  # Interactive Jupyter notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Phase 2 Results

### Data Processing Summary

- **Original Dataset:** 5,910 companies, 64 features
- **After Cleaning:** 5,849 samples (60 duplicates removed)
- **Final Features:** 53 features (12 redundant features removed)
- **Risk Distribution:**
  - Low Risk: 55.8% (3,300 companies)
  - Medium Risk: 23.2% (1,375 companies)
  - High Risk: 21.0% (1,235 companies)

### Data Splits

- **Training Set:** 4,094 samples (70%)
- **Test Set:** 1,755 samples (30%)
- **Stratification:** Applied to maintain class proportions

### Outputs Generated

- `data/processed/cleaned_data.csv` - Complete cleaned dataset
- `data/splits/X_train.csv, X_test.csv` - Feature matrices
- `data/splits/y_train.csv, y_test.csv` - Target variables
- `artifacts/scaler.joblib` - Fitted StandardScaler
- `artifacts/feature_list.joblib` - Selected feature list

## Methodology

### Data Cleaning
- **Missing Values:** Median imputation (numerical), mode imputation (categorical)
- **Duplicates:** Removed 60 duplicate rows
- **Outliers:** Winsorization (clipped at 1st and 99th percentiles)

### Feature Engineering
- **Redundancy Removal:** Features with correlation > 0.95 removed
- **Encoding:** One-hot encoding for categorical variables
- **Scaling:** StandardScaler (Z-score normalization)

### Data Splitting
- **Ratio:** 70% training / 30% test
- **Method:** Stratified splitting (maintains class proportions)
- **Reproducibility:** Fixed random seed (42)

## Configuration

Key settings can be modified in `src/config.py`:
- Train/test split ratio (default: 70/30)
- Random seed (default: 42)
- Correlation threshold for redundant features (default: 0.95)

## Documentation

- **Academic Report:** `reports/academic_report.md` - Phase 2 methodology summary
- **Jupyter Notebook:** `Phase2_Data_Preparation.ipynb` - Interactive exploration with visualizations

## Requirements

See `requirements.txt` for complete list. Main dependencies:
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- scipy>=1.7.0
- matplotlib>=3.5.0
- seaborn>=0.12.0
- jupyter>=1.0.0

## Next Steps

Phase 2 is complete. The data is cleaned, prepared, and ready for:
- **Phase 3:** Model Selection and Training
- **Phase 4:** Model Evaluation and Validation
- **Phase 5:** Model Deployment and Monitoring

## Repository

**GitHub:** https://github.com/TomasSilva20451/-MIA-MLA-PRATICAL-WORK

## Author

**Tomás Silva**  
Academic Project - MLA/MAAI  
2025

## License

Academic project for educational purposes.
