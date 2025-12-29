"""
Model evaluation functions for Phase 4
Computes comprehensive metrics: accuracy, precision, recall, F1-score, confusion matrix
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import RESULTS_DIR
from pathlib import Path

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model on test data
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class and macro-averaged metrics
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get class labels
    classes = sorted(y_test.unique())
    # Convert to list if it's a numpy array
    if hasattr(classes, 'tolist'):
        classes = classes.tolist()
    else:
        classes = list(classes)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': dict(zip(classes, precision_per_class)),
        'recall_per_class': dict(zip(classes, recall_per_class)),
        'f1_per_class': dict(zip(classes, f1_per_class)),
        'confusion_matrix': cm.tolist(),
        'classes': classes,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return results


def evaluate_all_models(trained_models, X_test, y_test):
    """
    Evaluate all trained models on test data
    
    Args:
        trained_models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Dictionary mapping model names to evaluation results
    """
    print("\n" + "="*60)
    print("Evaluating all models on test set...")
    print("="*60)
    
    all_results = {}
    
    for model_name, model_info in trained_models.items():
        model = model_info['model']
        print(f"\nEvaluating {model_name}...")
        
        results = evaluate_model(model, X_test, y_test, model_name)
        all_results[model_name] = results
        
        # Print summary
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision (macro): {results['precision_macro']:.4f}")
        print(f"  Recall (macro): {results['recall_macro']:.4f}")
        print(f"  F1-score (macro): {results['f1_macro']:.4f}")
    
    return all_results


def save_evaluation_results(all_results, filename='model_evaluation_results.json'):
    """
    Save evaluation results to JSON file
    
    Args:
        all_results: Dictionary of evaluation results
        filename: Output filename
    """
    import json
    
    # Ensure results directory exists (use absolute path)
    results_dir = PROJECT_ROOT / RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    results_to_save = {}
    for model_name, results in all_results.items():
        results_to_save[model_name] = {
            k: v for k, v in results.items()
            if k != 'classification_report'  # Skip detailed report for JSON
        }
        # Add summary from classification report
        results_to_save[model_name]['classification_report_summary'] = {
            'macro_avg': results['classification_report'].get('macro avg', {}),
            'weighted_avg': results['classification_report'].get('weighted avg', {})
        }
    
    filepath = results_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nSaved evaluation results to {filepath}")


def create_results_dataframe(all_results):
    """
    Create a pandas DataFrame summarizing all model results
    
    Args:
        all_results: Dictionary of evaluation results
    
    Returns:
        pd.DataFrame: Summary DataFrame
    """
    summary_data = []
    
    for model_name, results in all_results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision (Macro)': results['precision_macro'],
            'Recall (Macro)': results['recall_macro'],
            'F1-Score (Macro)': results['f1_macro']
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Accuracy', ascending=False)
    
    return df


def save_results_dataframe(df, filename='model_comparison.csv'):
    """
    Save results DataFrame to CSV
    
    Args:
        df: Results DataFrame
        filename: Output filename
    """
    results_dir = PROJECT_ROOT / RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Saved comparison table to {filepath}")
    return filepath

