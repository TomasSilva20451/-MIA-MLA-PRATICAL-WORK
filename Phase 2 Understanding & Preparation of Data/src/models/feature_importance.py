"""
Feature importance analysis for Phase 5
Extracts and visualizes feature importance from tree-based models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR, VISUALIZATIONS_DIR

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def get_feature_importance(model, feature_names, top_n=15):
    """
    Extract feature importance from a tree-based model
    
    Args:
        model: Trained sklearn model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        pd.DataFrame: DataFrame with feature names and importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Get top N
    top_features = importance_df.head(top_n).copy()
    
    return top_features, importance_df


def plot_feature_importance(importance_df, model_name, save_path=None, top_n=15):
    """
    Create a horizontal bar chart of feature importance
    
    Args:
        importance_df: DataFrame with feature names and importance scores
        model_name: Name of the model
        save_path: Path to save the figure (optional)
        top_n: Number of top features to display
    
    Returns:
        matplotlib figure
    """
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, 
                   color='steelblue', alpha=0.7)
    
    # Set y-axis labels
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=10)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'].values)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{importance:.4f}',
                ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance - {model_name.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Highest importance at top
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        vis_dir = PROJECT_ROOT / VISUALIZATIONS_DIR
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_path_full = vis_dir / Path(save_path).name if not Path(save_path).is_absolute() else save_path
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path_full}")
    
    return fig


def analyze_feature_importance(model_name='random_forest', feature_names=None, top_n=15):
    """
    Complete feature importance analysis for a model
    
    Args:
        model_name: Name of the model to analyze
        feature_names: List of feature names (if None, will try to load from data)
        top_n: Number of top features to analyze
    
    Returns:
        tuple: (top_features_df, full_importance_df)
    """
    # Load model
    model_path = PROJECT_ROOT / MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Loaded {model_name} model from {model_path}")
    
    # Get feature names if not provided
    if feature_names is None:
        # Try to load from splits data
        splits_dir = PROJECT_ROOT / "data" / "splits"
        try:
            X_train = pd.read_csv(splits_dir / "X_train.csv")
            feature_names = X_train.columns.tolist()
        except:
            raise ValueError("Could not load feature names. Please provide feature_names parameter.")
    
    # Get feature importance
    top_features, full_importance = get_feature_importance(model, feature_names, top_n)
    
    print(f"\nTop {top_n} Most Important Features:")
    print("="*60)
    for idx, row in top_features.iterrows():
        print(f"{row['feature']:30s} {row['importance']:.6f}")
    
    # Plot
    plot_feature_importance(full_importance, model_name, 
                          save_path=f"feature_importance_{model_name}.png", 
                          top_n=top_n)
    
    return top_features, full_importance


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from src.config import SPLITS_DATA_DIR
    
    # Load feature names
    X_train = pd.read_csv(PROJECT_ROOT / SPLITS_DATA_DIR / "X_train.csv")
    feature_names = X_train.columns.tolist()
    
    # Analyze Random Forest (selected model)
    top_features, full_importance = analyze_feature_importance(
        model_name='random_forest',
        feature_names=feature_names,
        top_n=15
    )

