"""
Data cleaning module
Handles missing values, duplicates, and outliers
"""

import pandas as pd
import numpy as np
from src.config import OUTLIER_PERCENTILE_LOW, OUTLIER_PERCENTILE_HIGH


def handle_missing_values(df):
    """
    Handle missing values using simple median (numeric) or mode (categorical) imputation
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame : Dataframe with missing values imputed
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                # Use median for numeric (more robust to outliers)
                impute_value = df_clean[col].median()
            else:
                # Use mode (most frequent) for categorical
                impute_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else None
            
            if impute_value is not None:
                df_clean[col] = df_clean[col].fillna(impute_value)
    
    return df_clean


def remove_duplicates(df):
    """
    Remove duplicate rows
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame : Dataframe with duplicates removed
    """
    return df.drop_duplicates()


def handle_outliers(df, clip_percentiles=True):
    """
    Handle outliers by clipping extreme percentiles (optional)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    clip_percentiles : bool, default=True
        Whether to clip values at percentiles
        
    Returns:
    --------
    pd.DataFrame : Dataframe with outliers handled
    """
    if not clip_percentiles:
        return df
    
    df_clean = df.copy()
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        lower = np.percentile(df_clean[col].dropna(), OUTLIER_PERCENTILE_LOW)
        upper = np.percentile(df_clean[col].dropna(), OUTLIER_PERCENTILE_HIGH)
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    
    return df_clean

