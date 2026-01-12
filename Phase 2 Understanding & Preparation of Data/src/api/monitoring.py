"""
Monitoring module for API
Handles performance metrics, prediction logging, data drift detection, and alerts
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import sys
import os

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MONITORING_DIR = PROJECT_ROOT / "artifacts" / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

# File paths
PREDICTIONS_LOG = MONITORING_DIR / "predictions.jsonl"
METRICS_FILE = MONITORING_DIR / "metrics.json"
DRIFT_STATS_FILE = MONITORING_DIR / "drift_stats.json"

# In-memory storage for performance metrics
_performance_metrics = {
    'request_count': 0,
    'error_count': 0,
    'response_times': deque(maxlen=1000),  # Keep last 1000 response times
    'predictions_history': deque(maxlen=100),  # Keep last 100 predictions in memory
    'start_time': time.time(),
    'endpoint_counts': defaultdict(int),
    'class_distribution': defaultdict(int),
    'confidence_scores': deque(maxlen=1000)
}

# Training data statistics for drift detection (loaded on startup)
_training_stats = None


def load_training_statistics():
    """
    Load training data statistics for drift detection baseline
    """
    global _training_stats
    
    if _training_stats is None:
        try:
            # Load training data to compute baseline statistics
            splits_dir = PROJECT_ROOT / "data" / "splits"
            X_train = pd.read_csv(splits_dir / "X_train.csv")
            
            # Compute statistics for each feature
            _training_stats = {
                'mean': X_train.mean().to_dict(),
                'std': X_train.std().to_dict(),
                'min': X_train.min().to_dict(),
                'max': X_train.max().to_dict(),
                'feature_count': len(X_train.columns)
            }
            
            # Save to file for persistence
            with open(DRIFT_STATS_FILE, 'w') as f:
                json.dump(_training_stats, f, indent=2)
            
            print(f"✓ Training statistics loaded for drift detection ({len(_training_stats['mean'])} features)")
        except Exception as e:
            print(f"⚠ Warning: Could not load training statistics: {e}")
            _training_stats = {
                'mean': {},
                'std': {},
                'min': {},
                'max': {},
                'feature_count': 0
            }
    
    return _training_stats


def log_prediction(features: Dict[str, float], prediction: str, 
                   probabilities: Dict[str, float], confidence: float,
                   response_time: float):
    """
    Log a prediction to file and in-memory history
    
    Parameters:
    -----------
    features: dict
        Input features
    prediction: str
        Predicted risk level
    probabilities: dict
        Probabilities for each class
    confidence: float
        Confidence score
    response_time: float
        Time taken for prediction (seconds)
    """
    timestamp = datetime.now().isoformat()
    
    # Create prediction record
    record = {
        'timestamp': timestamp,
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': probabilities,
        'response_time': response_time,
        'features_summary': {
            # Store summary statistics of features (not all values for privacy/space)
            'num_features': len(features),
            'feature_names': list(features.keys())[:5]  # First 5 feature names
        }
    }
    
    # Append to JSONL file
    try:
        with open(PREDICTIONS_LOG, 'a') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        print(f"⚠ Warning: Could not log prediction: {e}")
    
    # Add to in-memory history
    _performance_metrics['predictions_history'].append(record)
    
    # Update class distribution
    _performance_metrics['class_distribution'][prediction] += 1
    _performance_metrics['confidence_scores'].append(confidence)


def record_request(endpoint: str, response_time: float, success: bool = True):
    """
    Record API request metrics
    
    Parameters:
    -----------
    endpoint: str
        API endpoint called
    response_time: float
        Response time in seconds
    success: bool
        Whether request was successful
    """
    _performance_metrics['request_count'] += 1
    _performance_metrics['response_times'].append(response_time)
    _performance_metrics['endpoint_counts'][endpoint] += 1
    
    if not success:
        _performance_metrics['error_count'] += 1


def get_performance_metrics() -> Dict[str, Any]:
    """
    Get current performance metrics
    
    Returns:
    --------
    dict: Performance metrics dictionary
    """
    response_times = list(_performance_metrics['response_times'])
    confidence_scores = list(_performance_metrics['confidence_scores'])
    uptime = time.time() - _performance_metrics['start_time']
    
    metrics = {
        'uptime_seconds': uptime,
        'total_requests': _performance_metrics['request_count'],
        'total_errors': _performance_metrics['error_count'],
        'error_rate': _performance_metrics['error_count'] / max(_performance_metrics['request_count'], 1),
        'response_time': {
            'mean': np.mean(response_times) if response_times else 0,
            'min': np.min(response_times) if response_times else 0,
            'max': np.max(response_times) if response_times else 0,
            'p95': np.percentile(response_times, 95) if response_times else 0
        },
        'throughput': {
            'requests_per_second': _performance_metrics['request_count'] / max(uptime, 1),
            'requests_per_minute': (_performance_metrics['request_count'] / max(uptime, 1)) * 60
        },
        'endpoint_counts': dict(_performance_metrics['endpoint_counts']),
        'class_distribution': dict(_performance_metrics['class_distribution']),
        'confidence': {
            'mean': np.mean(confidence_scores) if confidence_scores else 0,
            'min': np.min(confidence_scores) if confidence_scores else 0,
            'max': np.max(confidence_scores) if confidence_scores else 0
        }
    }
    
    # Save to file
    try:
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"⚠ Warning: Could not save metrics: {e}")
    
    return metrics


def get_prediction_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get prediction history
    
    Parameters:
    -----------
    limit: int
        Maximum number of predictions to return
        
    Returns:
    --------
    list: List of prediction records
    """
    history = list(_performance_metrics['predictions_history'])
    
    # Also try to load from file if needed
    if len(history) < limit:
        try:
            if PREDICTIONS_LOG.exists():
                with open(PREDICTIONS_LOG, 'r') as f:
                    lines = f.readlines()
                    # Parse last N lines
                    for line in lines[-limit:]:
                        try:
                            record = json.loads(line.strip())
                            if record not in history:
                                history.append(record)
                        except:
                            continue
        except Exception as e:
            print(f"⚠ Warning: Could not load prediction history: {e}")
    
    # Return most recent first, limited
    return list(reversed(history[-limit:]))


def detect_data_drift(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Detect data drift by comparing feature values with training statistics
    
    Parameters:
    -----------
    features: dict
        Current feature values
        
    Returns:
    --------
    dict: Drift detection results
    """
    training_stats = load_training_statistics()
    
    if not training_stats['mean']:
        return {
            'drift_detected': False,
            'message': 'Training statistics not available'
        }
    
    drift_features = []
    drift_scores = {}
    
    for feature_name, value in features.items():
        if feature_name in training_stats['mean']:
            mean = training_stats['mean'][feature_name]
            std = training_stats['std'][feature_name]
            
            if std > 0:  # Avoid division by zero
                # Calculate z-score (how many standard deviations from mean)
                z_score = abs((value - mean) / std)
                
                # Flag as drift if more than 3 standard deviations (outlier)
                if z_score > 3:
                    drift_features.append(feature_name)
                    drift_scores[feature_name] = {
                        'z_score': z_score,
                        'value': value,
                        'training_mean': mean,
                        'training_std': std
                    }
    
    drift_detected = len(drift_features) > 0
    
    return {
        'drift_detected': drift_detected,
        'drift_features': drift_features,
        'drift_scores': drift_scores,
        'num_features_checked': len([f for f in features.keys() if f in training_stats['mean']]),
        'num_features_with_drift': len(drift_features)
    }


def check_model_degradation() -> Dict[str, Any]:
    """
    Check for model degradation based on confidence and class distribution
    
    Returns:
    --------
    dict: Degradation alerts
    """
    alerts = []
    
    # Check average confidence
    confidence_scores = list(_performance_metrics['confidence_scores'])
    if len(confidence_scores) >= 10:  # Need at least 10 predictions
        avg_confidence = np.mean(confidence_scores)
        if avg_confidence < 0.85:  # Threshold: 85% average confidence
            alerts.append({
                'type': 'low_confidence',
                'severity': 'warning',
                'message': f'Average confidence is low: {avg_confidence:.2%}',
                'threshold': '85%',
                'current_value': f'{avg_confidence:.2%}'
            })
    
    # Check class distribution (should be roughly similar to training: Low ~56%, Medium ~23%, High ~21%)
    class_dist = _performance_metrics['class_distribution']
    total = sum(class_dist.values())
    
    if total >= 20:  # Need at least 20 predictions
        expected_dist = {'Low': 0.558, 'Medium': 0.232, 'High': 0.210}
        
        for class_name, expected_pct in expected_dist.items():
            actual_pct = class_dist.get(class_name, 0) / total
            diff = abs(actual_pct - expected_pct)
            
            # Alert if distribution differs by more than 20 percentage points
            if diff > 0.20:
                alerts.append({
                    'type': 'distribution_shift',
                    'severity': 'warning',
                    'message': f'{class_name} class distribution shifted significantly',
                    'expected': f'{expected_pct:.1%}',
                    'actual': f'{actual_pct:.1%}',
                    'difference': f'{diff:.1%}'
                })
    
    # Check error rate
    error_rate = _performance_metrics['error_count'] / max(_performance_metrics['request_count'], 1)
    if error_rate > 0.05:  # More than 5% error rate
        alerts.append({
            'type': 'high_error_rate',
            'severity': 'critical',
            'message': f'Error rate is high: {error_rate:.2%}',
            'threshold': '5%',
            'current_value': f'{error_rate:.2%}'
        })
    
    return {
        'alerts': alerts,
        'num_alerts': len(alerts),
        'status': 'healthy' if len(alerts) == 0 else 'degraded'
    }


def get_drift_status() -> Dict[str, Any]:
    """
    Get overall data drift status based on recent predictions
    
    Returns:
    --------
    dict: Overall drift status
    """
    history = get_prediction_history(limit=50)  # Check last 50 predictions
    
    if not history:
        return {
            'status': 'insufficient_data',
            'message': 'Not enough predictions to detect drift',
            'predictions_analyzed': 0
        }
    
    # Aggregate drift detection across recent predictions
    drift_summary = {
        'total_checked': 0,
        'features_with_drift': set(),
        'drift_frequency': defaultdict(int)
    }
    
    for record in history:
        # We don't have full features in history, so we'll use a simplified approach
        # In a real system, we'd store feature statistics
        drift_summary['total_checked'] += 1
    
    # For now, return a summary based on available data
    return {
        'status': 'monitoring',
        'predictions_analyzed': len(history),
        'message': 'Drift detection active. Check individual predictions for drift details.',
        'note': 'Full drift analysis requires feature values from each prediction'
    }

