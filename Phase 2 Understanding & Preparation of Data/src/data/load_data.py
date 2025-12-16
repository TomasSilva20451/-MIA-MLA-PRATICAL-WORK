"""
Data loading module
Loads dataset from local file or downloads from UCI if needed
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import RAW_DATA_DIR, DATASET_NAME


def load_arff_file(file_path):
    """Load ARFF file format"""
    try:
        from scipy.io import arff
        data, meta = arff.loadarff(str(file_path))
        df = pd.DataFrame(data)
        # Decode byte strings if present
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.decode('utf-8')
                except (AttributeError, UnicodeDecodeError):
                    pass
        return df
    except ImportError:
        raise ImportError("scipy is required to load ARFF files. Install with: pip install scipy")


def load_polish_bankruptcy_data(data_path=None, year=5):
    """
    Load Polish Companies Bankruptcy Data
    
    Parameters:
    -----------
    data_path : str or Path, optional
        Path to data file. If None, looks in data/raw/
    year : int, default=5
        Which year to load (1-5, where 5 is most recent)
        
    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    if data_path is None:
        # Try to find in data/raw/
        raw_dir = Path(RAW_DATA_DIR)
        possible_files = [
            raw_dir / f"{DATASET_NAME}.csv",
            raw_dir / f"{DATASET_NAME}_{year}year.csv",
            raw_dir / f"{year}year.arff",
            raw_dir / f"{DATASET_NAME}_{year}year.arff",
        ]
        
        # Also check in current directory (for existing data)
        current_dir = Path(".")
        possible_files.extend([
            current_dir / f"polish+companies+bankruptcy+data" / f"{year}year.arff",
            current_dir / f"polish+companies+bankruptcy+data" / f"{year}year.csv",
        ])
        
        data_path = None
        for path in possible_files:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                f"Data file not found. Please ensure the dataset is in {RAW_DATA_DIR}/ "
                f"or run the download script first."
            )
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load based on file extension
    if data_path.suffix == '.arff':
        df = load_arff_file(data_path)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        # Try ARFF first, then CSV
        try:
            df = load_arff_file(data_path)
        except:
            df = pd.read_csv(data_path)
    
    return df


def create_risk_categories(df, bankruptcy_col='class'):
    """
    Convert binary bankruptcy classification to 3-class risk levels
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with bankruptcy classification
    bankruptcy_col : str, default='class'
        Name of the bankruptcy target column
        
    Returns:
    --------
    pd.DataFrame : Dataset with new 'risk_level' column
    """
    df_risk = df.copy()
    
    # Find bankruptcy column
    if bankruptcy_col not in df_risk.columns:
        # Try to find binary column
        binary_cols = [col for col in df_risk.columns 
                      if df_risk[col].nunique() == 2]
        if binary_cols:
            bankruptcy_col = binary_cols[0]
        else:
            raise ValueError("Could not find bankruptcy target column")
    
    # Convert to numeric if needed
    if df_risk[bankruptcy_col].dtype == object:
        df_risk['_bankruptcy_numeric'] = df_risk[bankruptcy_col].astype(str).str.strip()
        df_risk['_bankruptcy_numeric'] = df_risk['_bankruptcy_numeric'].map({
            '0': 0, '1': 1, '0.0': 0, '1.0': 1, 'False': 0, 'True': 1
        }).fillna(0).astype(int)
    else:
        df_risk['_bankruptcy_numeric'] = df_risk[bankruptcy_col].astype(int)
    
    # Get numerical financial ratio columns
    financial_cols = [col for col in df_risk.select_dtypes(include=[np.number]).columns 
                     if col not in ['_bankruptcy_numeric'] and not col.startswith('_')]
    
    # Use key ratios for risk assessment (first few are typically most important)
    key_ratios = financial_cols[:min(7, len(financial_cols))]
    
    # Create composite risk score
    df_risk['_risk_score'] = 0.0
    
    for col in key_ratios:
        if col in df_risk.columns:
            col_data = df_risk[col].fillna(df_risk[col].median())
            
            # For profitability ratios (negative is bad)
            if col == financial_cols[0]:  # Usually ROA-like
                risk_contribution = -np.clip(col_data, None, 0)
                risk_contribution = risk_contribution / (abs(col_data.quantile(0.25)) + 1e-6)
                df_risk['_risk_score'] += risk_contribution
            # For debt ratios (high is bad)
            elif len(financial_cols) > 1 and col == financial_cols[1]:
                p75 = col_data.quantile(0.75)
                risk_contribution = np.clip(col_data - p75, 0, None)
                risk_contribution = risk_contribution / (p75 + 1e-6)
                df_risk['_risk_score'] += risk_contribution
            # For other ratios (low is bad)
            else:
                p25 = col_data.quantile(0.25)
                risk_contribution = -np.clip(col_data - p25, None, 0)
                risk_contribution = risk_contribution / (abs(p25) + 1e-6)
                df_risk['_risk_score'] += risk_contribution
    
    # Bankrupt companies get high risk score
    df_risk.loc[df_risk['_bankruptcy_numeric'] == 1, '_risk_score'] = df_risk['_risk_score'].max() + 1
    
    # Normalize risk score
    df_risk['_risk_score'] = (df_risk['_risk_score'] - df_risk['_risk_score'].min()) / (
        df_risk['_risk_score'].max() - df_risk['_risk_score'].min() + 1e-6
    )
    
    # Create risk categories using percentiles of non-bankrupt companies
    non_bankrupt_scores = df_risk[df_risk['_bankruptcy_numeric'] == 0]['_risk_score']
    
    if len(non_bankrupt_scores) > 0:
        # Use higher percentiles since most scores are near zero
        # Low: bottom 60% of non-bankrupt (scores < 60th percentile)
        # Medium: middle 25% (60th to 85th percentile)
        # High: top 15% of non-bankrupt (>= 85th percentile) OR bankrupt
        high_threshold = non_bankrupt_scores.quantile(0.85)  # Top 15% of non-bankrupt
        medium_threshold = non_bankrupt_scores.quantile(0.60)  # 60th percentile
        
        # Initialize all as Low
        df_risk['risk_level'] = 'Low'
        
        # Assign Medium: scores between 60th and 85th percentile
        medium_mask = (df_risk['_risk_score'] >= medium_threshold) & (df_risk['_risk_score'] < high_threshold)
        df_risk.loc[medium_mask, 'risk_level'] = 'Medium'
        
        # Assign High: top 15% of non-bankrupt OR bankrupt
        high_mask = (df_risk['_risk_score'] >= high_threshold) | (df_risk['_bankruptcy_numeric'] == 1)
        df_risk.loc[high_mask, 'risk_level'] = 'High'
    else:
        df_risk['risk_level'] = 'Low'
        # All bankrupt companies are High Risk
        df_risk.loc[df_risk['_bankruptcy_numeric'] == 1, 'risk_level'] = 'High'
    
    # Clean up
    df_risk = df_risk.drop(columns=['_bankruptcy_numeric'], errors='ignore')
    
    return df_risk

