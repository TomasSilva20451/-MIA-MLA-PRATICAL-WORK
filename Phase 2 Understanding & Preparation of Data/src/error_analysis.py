"""
Error analysis for Phase 5
Detailed analysis of misclassifications and confusion patterns
"""

import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report
import sys
import os
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import RESULTS_DIR

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def analyze_confusion_matrix(y_true, y_pred, classes):
    """
    Analyze confusion matrix to extract error patterns
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class labels
    
    Returns:
        dict: Dictionary with error analysis results
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Calculate per-class metrics
    n_classes = len(classes)
    error_analysis = {
        'confusion_matrix': cm.tolist(),
        'classes': classes,
        'total_samples': len(y_true),
        'total_errors': int(np.sum(cm) - np.trace(cm)),
        'overall_accuracy': float(np.trace(cm) / np.sum(cm)),
        'per_class_metrics': {}
    }
    
    # Per-class analysis
    for i, class_name in enumerate(classes):
        # True positives, false positives, false negatives
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        # Actual and predicted counts
        actual_count = np.sum(cm[i, :])
        predicted_count = np.sum(cm[:, i])
        
        # Error rate
        error_rate = (fp + fn) / actual_count if actual_count > 0 else 0
        
        # Most confused with
        misclassifications = cm[i, :].copy()
        misclassifications[i] = 0  # Remove correct predictions
        most_confused_idx = np.argmax(misclassifications)
        most_confused_class = classes[most_confused_idx] if misclassifications[most_confused_idx] > 0 else None
        most_confused_count = int(misclassifications[most_confused_idx]) if most_confused_class else 0
        
        error_analysis['per_class_metrics'][class_name] = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'actual_count': int(actual_count),
            'predicted_count': int(predicted_count),
            'error_rate': float(error_rate),
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'most_confused_with': most_confused_class,
            'most_confused_count': most_confused_count
        }
    
    # Find most common confusion pairs
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'actual': classes[i],
                    'predicted': classes[j],
                    'count': int(cm[i, j]),
                    'percentage': float(cm[i, j] / np.sum(cm[i, :])) if np.sum(cm[i, :]) > 0 else 0.0
                })
    
    # Sort by count
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
    error_analysis['confusion_pairs'] = confusion_pairs[:10]  # Top 10
    
    return error_analysis


def analyze_model_errors(model, X_test, y_test, model_name):
    """
    Complete error analysis for a model
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    
    Returns:
        dict: Complete error analysis results
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get classes
    classes = sorted(y_test.unique())
    
    # Analyze confusion matrix
    error_analysis = analyze_confusion_matrix(y_test, y_pred, classes)
    error_analysis['model_name'] = model_name
    
    # Add classification report
    error_analysis['classification_report'] = classification_report(
        y_test, y_pred, labels=classes, output_dict=True
    )
    
    return error_analysis


def save_error_analysis(error_analysis, filename='error_analysis.json'):
    """
    Save error analysis results to JSON file
    
    Args:
        error_analysis: Dictionary with error analysis results
        filename: Output filename
    """
    results_dir = PROJECT_ROOT / RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = results_dir / filename
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_analysis = convert_to_serializable(error_analysis)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"Saved error analysis to {filepath}")
    return filepath


def print_error_summary(error_analysis):
    """
    Print a human-readable summary of error analysis
    
    Args:
        error_analysis: Dictionary with error analysis results
    """
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    print(f"Model: {error_analysis['model_name']}")
    print(f"Total Samples: {error_analysis['total_samples']}")
    print(f"Total Errors: {error_analysis['total_errors']}")
    print(f"Overall Accuracy: {error_analysis['overall_accuracy']:.4f} ({error_analysis['overall_accuracy']*100:.2f}%)")
    
    print("\n" + "-"*60)
    print("PER-CLASS ERROR RATES:")
    print("-"*60)
    for class_name, metrics in error_analysis['per_class_metrics'].items():
        print(f"\n{class_name}:")
        print(f"  Error Rate: {metrics['error_rate']:.4f} ({metrics['error_rate']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        if metrics['most_confused_with']:
            print(f"  Most confused with: {metrics['most_confused_with']} ({metrics['most_confused_count']} cases)")
    
    print("\n" + "-"*60)
    print("TOP CONFUSION PAIRS:")
    print("-"*60)
    for i, pair in enumerate(error_analysis['confusion_pairs'][:5], 1):
        print(f"{i}. {pair['actual']} → {pair['predicted']}: {pair['count']} cases ({pair['percentage']*100:.2f}%)")


if __name__ == "__main__":
    # Example usage
    import joblib
    from src.models.train_models import load_trained_model
    from src.config import SPLITS_DATA_DIR
    
    # Load data
    X_test = pd.read_csv(PROJECT_ROOT / SPLITS_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROJECT_ROOT / SPLITS_DATA_DIR / "y_test.csv").squeeze()
    
    # Load model
    model = load_trained_model('random_forest')
    
    # Analyze errors
    error_analysis = analyze_model_errors(model, X_test, y_test, 'random_forest')
    
    # Print summary
    print_error_summary(error_analysis)
    
    # Save results
    save_error_analysis(error_analysis)

