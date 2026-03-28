"""
Evaluation script for BISINDO ST-GCN model.

Generates:
- Per-class precision, recall, F1-score (Table 1 for karya tulis)
- Confusion matrix heatmap (Gambar 1) -- PNG, 300dpi
- Training curve plot (Gambar 3) -- PNG, 300dpi

Usage:
    python -m ml.evaluate
"""

import os
import sys
import csv

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.graph import build_adjacency_matrix
from ml.model import STGCN
from ml.dataset import create_dataloaders


# Constants
CLASSES = ["TOLONG", "BAHAYA", "KEBAKARAN"]
CHECKPOINT_PATH = os.path.join("ml", "checkpoints", "best_model.pt")
OUTPUT_DIR = os.path.join("ml", "results")
LOG_FILE = "training_log.csv"


def load_best_model(device):
    """Load the best model checkpoint."""
    A = build_adjacency_matrix().to(device)
    model = STGCN(num_classes=3, A=A).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded best model from epoch {checkpoint['epoch']} "
          f"(val_acc: {checkpoint['val_accuracy']:.2%})")

    return model


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Get all predictions and true labels from a dataloader."""
    all_preds = []
    all_labels = []
    all_confidences = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Softmax for confidence
        probs = F.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probs, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confidences.extend(confidences.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_confidences)


def generate_classification_report(y_true, y_pred):
    """Print and return classification metrics (Table 1)."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Table 1 -- Karya Tulis)")
    print("=" * 60)

    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print(report)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None
    )
    accuracy = accuracy_score(y_true, y_pred)

    # Save as CSV for easy copy to karya tulis
    csv_path = os.path.join(OUTPUT_DIR, "classification_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
        for i, cls in enumerate(CLASSES):
            writer.writerow([cls, f"{precision[i]:.4f}", f"{recall[i]:.4f}",
                             f"{f1[i]:.4f}", int(support[i])])

        # Macro average
        macro_p = precision.mean()
        macro_r = recall.mean()
        macro_f1 = f1.mean()
        writer.writerow(["Macro Avg", f"{macro_p:.4f}", f"{macro_r:.4f}",
                         f"{macro_f1:.4f}", int(support.sum())])
        writer.writerow(["Accuracy", f"{accuracy:.4f}", "", "", int(support.sum())])

    print(f"  Metrics saved to {csv_path}")

    return accuracy


def plot_confusion_matrix(y_true, y_pred):
    """Generate confusion matrix heatmap (Gambar 1)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES,
        square=True, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 16, "weight": "bold"}
    )
    ax.set_xlabel("Prediksi", fontsize=14)
    ax.set_ylabel("Label Sebenarnya", fontsize=14)
    ax.set_title("Confusion Matrix -- Kondisi Normal", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to {path}")


def plot_training_curve():
    """Generate training curve: train_loss vs val_loss (Gambar 3)."""
    if not os.path.exists(LOG_FILE):
        print(f"  Training log not found: {LOG_FILE}")
        return

    epochs, train_losses, val_losses, val_accs = [], [], [], []

    with open(LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            val_accs.append(float(row['val_accuracy']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, label='Train Loss', color='#2196F3', linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', color='#FF5722', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, val_accs, label='Validation Accuracy', color='#4CAF50', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curve.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Training curve saved to {path}")


def main():
    """Run full evaluation pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_best_model(device)

    # Load test set (original only, no augmentation)
    loaders = create_dataloaders(include_augmented=False)
    test_loader = loaders['test']
    print(f"Test set: {len(test_loader.dataset)} samples")

    # Get predictions
    y_pred, y_true, confidences = get_predictions(model, test_loader, device)

    # Generate outputs
    accuracy = generate_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    plot_training_curve()

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"  Test accuracy: {accuracy:.2%}")
    print(f"  Mean confidence: {confidences.mean():.4f}")
    print(f"  Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
