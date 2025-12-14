"""
Visualization Utilities

Features:
- Skeleton drawing on video frames
- Attention heatmap overlay
- Confusion matrix plotting
- Training curves visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from pathlib import Path


# Color scheme for visualization
COLORS = {
    'pose': (0, 255, 0),      # Green
    'left_hand': (255, 128, 0),  # Orange
    'right_hand': (0, 128, 255),  # Blue
    'connection': (200, 200, 200),  # Gray
}


# Skeleton connections
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Arms
    (11, 13), (13, 15), (12, 14), (14, 16),
    # Legs (optional)
    (23, 25), (25, 27), (24, 26), (26, 28),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    num_pose: int = 33,
    num_hand: int = 21,
    threshold: float = 0.5,
    line_thickness: int = 2,
    point_radius: int = 4
) -> np.ndarray:
    """
    Draw skeleton on video frame.
    
    Args:
        frame: Input BGR image
        keypoints: Keypoint array (V, 3) with x, y, visibility
        num_pose: Number of pose landmarks
        num_hand: Number of hand landmarks per hand
        threshold: Visibility threshold
        line_thickness: Thickness of connection lines
        point_radius: Radius of keypoint circles
        
    Returns:
        Frame with skeleton overlay
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    def get_point(idx):
        """Get pixel coordinates for keypoint."""
        if idx >= len(keypoints) or keypoints[idx, 2] < threshold:
            return None
        return (int(keypoints[idx, 0] * w), int(keypoints[idx, 1] * h))
    
    # Draw pose connections
    for start, end in POSE_CONNECTIONS:
        pt1 = get_point(start)
        pt2 = get_point(end)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, COLORS['pose'], line_thickness)
    
    # Draw left hand connections
    left_hand_offset = num_pose
    for start, end in HAND_CONNECTIONS:
        pt1 = get_point(left_hand_offset + start)
        pt2 = get_point(left_hand_offset + end)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, COLORS['left_hand'], line_thickness)
    
    # Draw right hand connections
    right_hand_offset = num_pose + num_hand
    for start, end in HAND_CONNECTIONS:
        pt1 = get_point(right_hand_offset + start)
        pt2 = get_point(right_hand_offset + end)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, COLORS['right_hand'], line_thickness)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if kp[2] >= threshold:
            x, y = int(kp[0] * w), int(kp[1] * h)
            
            if i < num_pose:
                color = COLORS['pose']
            elif i < num_pose + num_hand:
                color = COLORS['left_hand']
            else:
                color = COLORS['right_hand']
            
            cv2.circle(frame, (x, y), point_radius, color, -1)
            cv2.circle(frame, (x, y), point_radius, (255, 255, 255), 1)
    
    return frame


def draw_attention_heatmap(
    frame: np.ndarray,
    attention_weights: np.ndarray,
    alpha: float = 0.3,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay attention heatmap on frame.
    
    For per-frame attention, draws a temporal bar.
    For spatial attention, creates a heatmap overlay.
    
    Args:
        frame: Input BGR image
        attention_weights: Attention weights (T,) for temporal or (V,) for spatial
        alpha: Transparency of overlay
        colormap: OpenCV colormap
        
    Returns:
        Frame with attention overlay
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Normalize attention
    attention_norm = attention_weights / (attention_weights.max() + 1e-10)
    
    # Draw attention bar at bottom
    bar_height = 30
    bar_y = h - bar_height - 10
    
    # Background
    cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + bar_height), 
                 (30, 30, 30), -1)
    
    # Draw attention segments
    segment_width = (w - 20) / len(attention_norm)
    
    for i, weight in enumerate(attention_norm):
        x1 = int(10 + i * segment_width)
        x2 = int(10 + (i + 1) * segment_width) - 1
        
        # Color based on attention (red = high, blue = low)
        color_val = np.array([[int(weight * 255)]], dtype=np.uint8)
        color_map = cv2.applyColorMap(color_val, colormap)
        color = tuple(map(int, color_map[0, 0]))
        
        cv2.rectangle(frame, (x1, bar_y + 2), (x2, bar_y + bar_height - 2), 
                     color, -1)
    
    # Label
    cv2.putText(frame, "Temporal Attention", (10, bar_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate confusion matrix plot.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the matrix
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_training_curves(
    history: List[Dict],
    metrics: List[str] = ['loss', 'accuracy'],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: List of epoch metrics dictionaries
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    epochs = [h['epoch'] for h in history]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        train_values = [h.get(train_key, h.get(metric)) for h in history]
        val_values = [h.get(val_key) for h in history]
        
        ax.plot(epochs, train_values, 'b-', label='Train', linewidth=2)
        if val_values[0] is not None:
            ax.plot(epochs, val_values, 'r-', label='Validation', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Over Training', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark best epoch
        if 'acc' in metric or 'f1' in metric:
            best_idx = np.argmax(val_values) if val_values[0] is not None else np.argmax(train_values)
        else:
            best_idx = np.argmin(val_values) if val_values[0] is not None else np.argmin(train_values)
        
        best_val = val_values[best_idx] if val_values[0] is not None else train_values[best_idx]
        ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5)
        ax.annotate(f'Best: {best_val:.4f}', 
                   xy=(epochs[best_idx], best_val),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot class distribution bar chart.
    
    Args:
        class_counts: Dictionary of class names to counts
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = ax.bar(classes, counts, color='steelblue', edgecolor='white')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=10)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")
    
    return fig


def create_demo_video(
    video_path: str,
    keypoints: np.ndarray,
    predictions: List[Tuple[int, str, float]],
    attention_weights: np.ndarray,
    class_names: List[str],
    output_path: str,
    fps: int = 30
) -> str:
    """
    Create demo video with overlays.
    
    Args:
        video_path: Input video path
        keypoints: Keypoints array (T, V, 3)
        predictions: List of (frame_idx, class_name, confidence) tuples
        attention_weights: Attention weights (T,)
        class_names: List of class names
        output_path: Output video path
        fps: Output FPS
        
    Returns:
        Path to output video
    """
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    current_pred = None
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(keypoints):
            break
        
        # Draw skeleton
        frame = draw_skeleton(frame, keypoints[frame_idx])
        
        # Draw attention
        if attention_weights is not None:
            frame = draw_attention_heatmap(frame, attention_weights)
        
        # Find current prediction
        for pred_frame, pred_class, pred_conf in predictions:
            if pred_frame <= frame_idx:
                current_pred = (pred_class, pred_conf)
        
        # Draw prediction
        if current_pred:
            pred_class, pred_conf = current_pred
            text = f"{pred_class}: {pred_conf:.1%}"
            
            # Background
            cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
            
            # Text
            color = (0, 255, 0) if pred_conf > 0.8 else (0, 255, 255)
            cv2.putText(frame, text, (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"Demo video saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Test visualizations
    import numpy as np
    
    # Test confusion matrix
    y_true = np.random.randint(0, 10, 100)
    y_pred = np.random.randint(0, 10, 100)
    class_names = [f"CLASS_{i}" for i in range(10)]
    
    fig = plot_confusion_matrix(y_true, y_pred, class_names, 
                                save_path="test_confusion.png")
    plt.close()
    
    # Test training curves
    history = [
        {'epoch': i+1, 'train_loss': 2.0/(i+1), 'val_loss': 2.5/(i+1),
         'train_accuracy': 0.5 + 0.05*i, 'val_accuracy': 0.45 + 0.05*i}
        for i in range(20)
    ]
    
    fig = plot_training_curves(history, save_path="test_curves.png")
    plt.close()
    
    print("Visualization tests complete!")
