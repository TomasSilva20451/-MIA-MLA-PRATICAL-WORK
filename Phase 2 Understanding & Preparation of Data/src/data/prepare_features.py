"""
Feature preparation module
Handles feature selection, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import CORRELATION_THRESHOLD


def remove_redundant_features(df):
    """
    Remove highly correlated features (redundant)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame : Dataframe with redundant features removed
    list : List of features to keep
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        return df, numerical_cols
    
    corr_matrix = df[numerical_cols].corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_remove = [column for column in upper_triangle.columns 
                if any(upper_triangle[column] > CORRELATION_THRESHOLD)]
    
    features_to_keep = [col for col in df.columns if col not in to_remove]
    
    return df[features_to_keep], features_to_keep


def encode_categorical(df):
    """
    Encode categorical variables using one-hot encoding
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame : Dataframe with categorical variables encoded
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) == 0:
        return df
    
    # Exclude target variable if present
    if 'risk_level' in categorical_cols:
        categorical_cols = categorical_cols.drop('risk_level')
    
    if len(categorical_cols) == 0:
        return df
    
    return pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)


def scale_features(X_train, X_test, scaler=None):
    """
    Scale features using StandardScaler
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    X_test : pd.DataFrame or np.ndarray
        Test features
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None, fits a new one.
        
    Returns:
    --------
    np.ndarray : Scaled training features
    np.ndarray : Scaled test features
    StandardScaler : Fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

