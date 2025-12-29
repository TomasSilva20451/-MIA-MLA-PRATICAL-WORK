"""
Model comparison and visualization utilities for Phase 4
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import VISUALIZATIONS_DIR

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_accuracy_comparison(all_results, save_path=None):
    """
    Create a bar chart comparing accuracy of all models
    
    Args:
        all_results: Dictionary of evaluation results
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure
    """
    model_names = []
    accuracies = []
    
    for model_name, results in all_results.items():
        model_names.append(model_name.replace('_', ' ').title())
        accuracies.append(results['accuracy'])
    
    # Sort by accuracy
    sorted_data = sorted(zip(model_names, accuracies), key=lambda x: x[1], reverse=True)
    model_names, accuracies = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(model_names, accuracies, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        vis_dir = PROJECT_ROOT / VISUALIZATIONS_DIR
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_path_full = vis_dir / Path(save_path).name if not Path(save_path).is_absolute() else save_path
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy comparison to {save_path_full}")
    
    return fig


def plot_metrics_comparison(all_results, save_path=None):
    """
    Create a bar chart comparing precision, recall, and F1-score (macro-averaged)
    
    Args:
        all_results: Dictionary of evaluation results
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure
    """
    model_names = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for model_name, results in all_results.items():
        model_names.append(model_name.replace('_', ' ').title())
        precision_scores.append(results['precision_macro'])
        recall_scores.append(results['recall_macro'])
        f1_scores.append(results['f1_macro'])
    
    # Sort by F1-score
    sorted_data = sorted(zip(model_names, precision_scores, recall_scores, f1_scores),
                        key=lambda x: x[3], reverse=True)
    model_names, precision_scores, recall_scores, f1_scores = zip(*sorted_data)
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, precision_scores, width, label='Precision', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, recall_scores, width, label='Recall', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics Comparison (Macro-Averaged)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        vis_dir = PROJECT_ROOT / VISUALIZATIONS_DIR
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_path_full = vis_dir / Path(save_path).name if not Path(save_path).is_absolute() else save_path
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path_full}")
    
    return fig


def plot_confusion_matrices(all_results, save_path=None):
    """
    Create confusion matrix heatmaps for all models
    
    Args:
        all_results: Dictionary of evaluation results
        save_path: Base path to save figures (optional)
    
    Returns:
        list of matplotlib figures
    """
    n_models = len(all_results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        cm = np.array(results['confusion_matrix'])
        classes = results['classes']
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Count'})
        
        ax.set_title(f'{model_name.replace("_", " ").title()}\nAccuracy: {results["accuracy"]:.4f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        vis_dir = PROJECT_ROOT / VISUALIZATIONS_DIR
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_path_full = vis_dir / Path(save_path).name if not Path(save_path).is_absolute() else save_path
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrices to {save_path_full}")
    
    return fig


def create_comparison_table(all_results):
    """
    Create a formatted comparison table of all models
    
    Args:
        all_results: Dictionary of evaluation results
    
    Returns:
        pd.DataFrame: Formatted comparison table
    """
    data = []
    
    for model_name, results in all_results.items():
        data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision (Macro)': f"{results['precision_macro']:.4f}",
            'Recall (Macro)': f"{results['recall_macro']:.4f}",
            'F1-Score (Macro)': f"{results['f1_macro']:.4f}"
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Accuracy', ascending=False, key=lambda x: x.astype(float))
    
    return df


def generate_all_visualizations(all_results):
    """
    Generate all comparison visualizations
    
    Args:
        all_results: Dictionary of evaluation results
    """
    print("\nGenerating visualizations...")
    
    vis_dir = PROJECT_ROOT / VISUALIZATIONS_DIR
    
    # Accuracy comparison
    plot_accuracy_comparison(
        all_results,
        save_path=str(vis_dir / "accuracy_comparison.png")
    )
    
    # Metrics comparison
    plot_metrics_comparison(
        all_results,
        save_path=str(vis_dir / "metrics_comparison.png")
    )
    
    # Confusion matrices
    plot_confusion_matrices(
        all_results,
        save_path=str(vis_dir / "confusion_matrices.png")
    )
    
    print("\nAll visualizations generated!")

