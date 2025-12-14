"""
Evaluation Metrics for Sign Classification

Includes:
- Accuracy computation
- F1 score (macro, micro, weighted)
- Confusion matrix
- Per-class metrics
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted class indices (N,) or logits (N, C)
        targets: Ground truth labels (N,)
        
    Returns:
        Accuracy as float
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    
    return correct / total


def compute_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'macro',
    num_classes: Optional[int] = None
) -> float:
    """
    Compute F1 score.
    
    Args:
        predictions: Predicted class indices (N,) or logits (N, C)
        targets: Ground truth labels (N,)
        average: 'macro', 'micro', 'weighted', or None for per-class
        num_classes: Number of classes (optional)
        
    Returns:
        F1 score as float (or array if average=None)
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    labels = None
    if num_classes is not None:
        labels = list(range(num_classes))
    
    return f1_score(
        targets_np, preds_np, 
        average=average, 
        labels=labels,
        zero_division=0
    )


def compute_precision_recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'macro',
    num_classes: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute precision and recall.
    
    Args:
        predictions: Predicted class indices (N,) or logits (N, C)
        targets: Ground truth labels (N,)
        average: 'macro', 'micro', 'weighted', or None for per-class
        num_classes: Number of classes (optional)
        
    Returns:
        Tuple of (precision, recall)
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    labels = None
    if num_classes is not None:
        labels = list(range(num_classes))
    
    precision = precision_score(
        targets_np, preds_np, 
        average=average, 
        labels=labels,
        zero_division=0
    )
    
    recall = recall_score(
        targets_np, preds_np, 
        average=average, 
        labels=labels,
        zero_division=0
    )
    
    return precision, recall


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices (N,) or logits (N, C)
        targets: Ground truth labels (N,)
        num_classes: Number of classes (optional)
        
    Returns:
        Confusion matrix as numpy array (num_classes, num_classes)
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    labels = None
    if num_classes is not None:
        labels = list(range(num_classes))
    
    return confusion_matrix(targets_np, preds_np, labels=labels)


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None
) -> Dict:
    """
    Compute all metrics.
    
    Args:
        predictions: Predicted class indices (N,) or logits (N, C)
        targets: Ground truth labels (N,)
        class_names: Optional list of class names
        num_classes: Number of classes
        
    Returns:
        Dictionary with all metrics
    """
    if predictions.dim() > 1:
        num_classes = predictions.shape[-1] if num_classes is None else num_classes
        predictions = predictions.argmax(dim=-1)
    
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    metrics = {
        'accuracy': compute_accuracy(predictions, targets),
        'f1_macro': compute_f1(predictions, targets, 'macro', num_classes),
        'f1_micro': compute_f1(predictions, targets, 'micro', num_classes),
        'f1_weighted': compute_f1(predictions, targets, 'weighted', num_classes),
    }
    
    precision, recall = compute_precision_recall(
        predictions, targets, 'macro', num_classes
    )
    metrics['precision_macro'] = precision
    metrics['recall_macro'] = recall
    
    # Confusion matrix
    metrics['confusion_matrix'] = compute_confusion_matrix(
        predictions, targets, num_classes
    )
    
    # Per-class metrics
    if class_names is not None:
        labels = list(range(len(class_names)))
        report = classification_report(
            targets_np, preds_np,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        metrics['per_class'] = {
            name: {
                'precision': report[name]['precision'],
                'recall': report[name]['recall'],
                'f1': report[name]['f1-score'],
                'support': report[name]['support']
            }
            for name in class_names if name in report
        }
    
    return metrics


class MetricTracker:
    """
    Track metrics over training/evaluation epochs.
    """
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize metric tracker.
        
        Args:
            metrics: List of metric names to track
        """
        if metrics is None:
            metrics = ['loss', 'accuracy', 'f1_macro']
        
        self.metrics = metrics
        self.history = {m: [] for m in metrics}
        self.current_epoch = {}
    
    def reset_epoch(self):
        """Reset current epoch accumulator."""
        self.current_epoch = {m: [] for m in self.metrics}
    
    def update(self, **kwargs):
        """
        Update with batch metrics.
        
        Args:
            **kwargs: Metric name-value pairs
        """
        for name, value in kwargs.items():
            if name in self.current_epoch:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.current_epoch[name].append(value)
    
    def end_epoch(self) -> Dict[str, float]:
        """
        Compute epoch averages and store in history.
        
        Returns:
            Dictionary of epoch metrics
        """
        epoch_metrics = {}
        
        for name in self.metrics:
            if self.current_epoch[name]:
                avg = np.mean(self.current_epoch[name])
                self.history[name].append(avg)
                epoch_metrics[name] = avg
            else:
                epoch_metrics[name] = 0.0
        
        self.reset_epoch()
        return epoch_metrics
    
    def get_history(self, metric: str) -> List[float]:
        """Get history for a specific metric."""
        return self.history.get(metric, [])
    
    def get_best(self, metric: str, mode: str = 'max') -> Tuple[int, float]:
        """
        Get best epoch and value for a metric.
        
        Args:
            metric: Metric name
            mode: 'max' or 'min'
            
        Returns:
            Tuple of (best_epoch, best_value)
        """
        history = self.history.get(metric, [])
        if not history:
            return -1, 0.0
        
        if mode == 'max':
            best_idx = np.argmax(history)
        else:
            best_idx = np.argmin(history)
        
        return best_idx, history[best_idx]


if __name__ == "__main__":
    # Test metrics
    num_classes = 10
    N = 100
    
    # Create dummy predictions and targets
    predictions = torch.randint(0, num_classes, (N,))
    targets = torch.randint(0, num_classes, (N,))
    
    # Make some correct for realistic accuracy
    predictions[:50] = targets[:50]
    
    class_names = [f'CLASS_{i}' for i in range(num_classes)]
    
    # Compute all metrics
    metrics = compute_all_metrics(predictions, targets, class_names, num_classes)
    
    print("Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall: {metrics['recall_macro']:.4f}")
    
    print("\nConfusion Matrix shape:", metrics['confusion_matrix'].shape)
    
    print("\nPer-class metrics:")
    for name, class_metrics in list(metrics['per_class'].items())[:3]:
        print(f"  {name}: P={class_metrics['precision']:.2f}, "
              f"R={class_metrics['recall']:.2f}, F1={class_metrics['f1']:.2f}")
    
    # Test metric tracker
    print("\nMetric Tracker test:")
    tracker = MetricTracker(['loss', 'accuracy'])
    
    for epoch in range(3):
        tracker.reset_epoch()
        
        for batch in range(10):
            tracker.update(
                loss=np.random.uniform(0.5, 2.0),
                accuracy=np.random.uniform(0.5, 0.9)
            )
        
        epoch_metrics = tracker.end_epoch()
        print(f"  Epoch {epoch}: loss={epoch_metrics['loss']:.4f}, "
              f"acc={epoch_metrics['accuracy']:.4f}")
    
    best_epoch, best_acc = tracker.get_best('accuracy', 'max')
    print(f"  Best accuracy: {best_acc:.4f} at epoch {best_epoch}")
