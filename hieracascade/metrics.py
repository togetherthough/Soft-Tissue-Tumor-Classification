"""Evaluation metrics for HieraCascade-STS"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from typing import Dict, List, Optional, Tuple
import torch


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'macro',
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute comprehensive classification metrics
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Predicted probabilities (N, C) - optional for AUC
        average: 'macro', 'micro', or 'weighted'
        class_names: Optional class names for reporting
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Balanced accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    metrics[f'{average}_precision'] = precision
    metrics[f'{average}_recall'] = recall
    metrics[f'{average}_f1'] = f1
    
    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f1_val in enumerate(f1_per_class):
        class_name = class_names[i] if class_names else f'class_{i}'
        metrics[f'f1_{class_name}'] = f1_val
    
    # AUC (if probabilities provided)
    if y_prob is not None:
        n_classes = y_prob.shape[1]
        if n_classes == 2:
            # Binary
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Multi-class OVR
            try:
                metrics['auc_ovr'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average=average
                )
            except ValueError:
                # Not all classes present
                metrics['auc_ovr'] = np.nan
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None
) -> np.ndarray:
    """Compute confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: None, 'true', 'pred', or 'all'
        
    Returns:
        Confusion matrix (C, C)
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    return cm


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
):
    """Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )
    print(report)


def compute_per_site_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sites: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """Compute metrics separately for each site
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sites: Site identifiers
        y_prob: Optional predicted probabilities
        
    Returns:
        Dict mapping site to metrics dict
    """
    site_metrics = {}
    
    unique_sites = np.unique(sites)
    
    for site in unique_sites:
        mask = sites == site
        
        site_y_true = y_true[mask]
        site_y_pred = y_pred[mask]
        site_y_prob = y_prob[mask] if y_prob is not None else None
        
        site_metrics[str(site)] = compute_metrics(
            site_y_true, site_y_pred, site_y_prob
        )
    
    return site_metrics


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_fn: Function that takes (y_true, y_pred) and returns a scalar
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0-1)
        random_state: Random seed
        
    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
    mean = np.mean(bootstrap_scores)
    
    return mean, lower, upper


def aggregate_predictions(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
    y_prob_list: Optional[List[np.ndarray]] = None
) -> Dict[str, float]:
    """Aggregate predictions from multiple folds
    
    Args:
        y_true_list: List of true labels per fold
        y_pred_list: List of predictions per fold
        y_prob_list: Optional list of probabilities per fold
        
    Returns:
        Aggregated metrics with mean and std
    """
    fold_metrics = []
    
    for i in range(len(y_true_list)):
        y_prob = y_prob_list[i] if y_prob_list else None
        metrics = compute_metrics(y_true_list[i], y_pred_list[i], y_prob)
        fold_metrics.append(metrics)
    
    # Aggregate
    aggregated = {}
    
    # Get all metric keys
    all_keys = set()
    for m in fold_metrics:
        all_keys.update(m.keys())
    
    for key in all_keys:
        values = [m.get(key, np.nan) for m in fold_metrics]
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 0:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
    
    return aggregated


class MetricsTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked values"""
        self.y_true = []
        self.y_pred = []
        self.y_prob = []
        self.losses = []
    
    def update(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None,
        loss: Optional[float] = None
    ):
        """Update with batch results
        
        Args:
            y_true: True labels (B,)
            y_pred: Predicted labels (B,)
            y_prob: Predicted probabilities (B, C)
            loss: Batch loss
        """
        self.y_true.append(y_true.cpu().numpy())
        self.y_pred.append(y_pred.cpu().numpy())
        
        if y_prob is not None:
            self.y_prob.append(y_prob.cpu().numpy())
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute metrics from all tracked values"""
        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)
        
        y_prob = None
        if len(self.y_prob) > 0:
            y_prob = np.concatenate(self.y_prob)
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        if len(self.losses) > 0:
            metrics['loss'] = np.mean(self.losses)
        
        return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = '') -> str:
    """Format metrics dictionary as string
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for keys
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    
    return '\n'.join(lines)
