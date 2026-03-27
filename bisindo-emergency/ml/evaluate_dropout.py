"""
Keypoint dropout robustness evaluation for BISINDO ST-GCN.

Tests model robustness by randomly zeroing out keypoints at various rates.
Generates Table 2 and Gambar 2 for the karya tulis.

Usage:
    python -m ml.evaluate_dropout
"""

import os
import sys
import csv

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.graph import build_adjacency_matrix
from ml.model import STGCN
from ml.dataset import create_dataloaders


# Constants
CLASSES = ["TOLONG", "BAHAYA", "KEBAKARAN"]
CHECKPOINT_PATH = os.path.join("ml", "checkpoints", "best_model.pt")
OUTPUT_DIR = os.path.join("ml", "results")

DROPOUT_RATES = [0.0, 0.10, 0.25, 0.50]
NUM_SEEDS = 10
NUM_KEYPOINTS = 75


def load_best_model(device):
    """Load the best model checkpoint."""
    A = build_adjacency_matrix().to(device)
    model = STGCN(num_classes=3, A=A).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def apply_keypoint_dropout(inputs: torch.Tensor, dropout_rate: float,
                           seed: int) -> torch.Tensor:
    """Zero out a fraction of keypoints in the input tensor.

    Args:
        inputs: (batch, 3, 60, 75) tensor.
        dropout_rate: Fraction of keypoints to zero out.
        seed: Random seed for reproducibility.

    Returns:
        Modified tensor with selected keypoints zeroed out.
    """
    if dropout_rate == 0.0:
        return inputs

    rng = np.random.RandomState(seed)
    n_drop = int(np.floor(dropout_rate * NUM_KEYPOINTS))

    # Select keypoints to drop (same for all samples in batch)
    drop_indices = rng.choice(NUM_KEYPOINTS, size=n_drop, replace=False)

    # Zero out: set all 3 channels to 0 for dropped keypoints across all frames
    dropped = inputs.clone()
    dropped[:, :, :, drop_indices] = 0.0

    return dropped


@torch.no_grad()
def evaluate_with_dropout(model, dataloader, dropout_rate, seed, device):
    """Evaluate model accuracy with keypoint dropout applied.

    Returns:
        accuracy (float)
    """
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Apply keypoint dropout
        inputs = apply_keypoint_dropout(inputs, dropout_rate, seed)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / max(total, 1)


def run_dropout_evaluation():
    """Run full dropout evaluation across all rates and seeds."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥  Device: {device}")

    # Load model
    model = load_best_model(device)
    print("✓ Model loaded")

    # Load test set (original only)
    loaders = create_dataloaders(include_augmented=False)
    test_loader = loaders['test']
    print(f"📊 Test set: {len(test_loader.dataset)} samples")

    # Evaluate at each dropout rate
    results = {}
    baseline_accuracy = None

    print(f"\n{'='*60}")
    print("KEYPOINT DROPOUT ROBUSTNESS EVALUATION")
    print(f"{'='*60}")

    for rate in DROPOUT_RATES:
        n_drop = int(np.floor(rate * NUM_KEYPOINTS))
        print(f"\n🔄 Dropout rate: {rate:.0%} ({n_drop}/{NUM_KEYPOINTS} keypoints)")

        accuracies = []
        for seed in range(NUM_SEEDS):
            acc = evaluate_with_dropout(model, test_loader, rate, seed, device)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        if rate == 0.0:
            baseline_accuracy = mean_acc
            dar = 0.0
        else:
            dar = ((baseline_accuracy - mean_acc) / baseline_accuracy) * 100.0

        results[rate] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "dar": dar,
            "accuracies": accuracies,
        }

        print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  DAR:      {dar:.2f}%")

    # Save Table 2 as CSV
    save_table(results)

    # Plot Gambar 2
    plot_degradation(results)

    print(f"\n{'='*60}")
    print("✓ Dropout evaluation complete!")
    print(f"  Results saved to: {OUTPUT_DIR}/")


def save_table(results):
    """Save dropout results as CSV (Table 2)."""
    csv_path = os.path.join(OUTPUT_DIR, "dropout_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dropout Rate", "Keypoints Dropped",
                         "Accuracy Mean", "Accuracy Std", "DAR (%)"])
        for rate, data in results.items():
            n_drop = int(np.floor(rate * NUM_KEYPOINTS))
            writer.writerow([
                f"{rate:.0%}",
                f"{n_drop}/{NUM_KEYPOINTS}",
                f"{data['mean_accuracy']:.4f}",
                f"{data['std_accuracy']:.4f}",
                f"{data['dar']:.2f}"
            ])
    print(f"  → Table saved to {csv_path}")


def plot_degradation(results):
    """Plot accuracy degradation vs dropout rate with error bars (Gambar 2)."""
    rates = list(results.keys())
    means = [results[r]["mean_accuracy"] for r in rates]
    stds = [results[r]["std_accuracy"] for r in rates]
    rate_labels = [f"{r:.0%}" for r in rates]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        rate_labels, means, yerr=stds,
        fmt='o-', color='#E53935', linewidth=2.5,
        markersize=10, capsize=6, capthick=2,
        ecolor='#FFAB91', markerfacecolor='#E53935',
        markeredgecolor='white', markeredgewidth=2
    )

    ax.set_xlabel("Tingkat Dropout Keypoint", fontsize=13)
    ax.set_ylabel("Akurasi", fontsize=13)
    ax.set_title("Degradasi Akurasi vs Tingkat Dropout Keypoint",
                 fontsize=15, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (rate, mean, std) in enumerate(zip(rates, means, stds)):
        ax.annotate(f"{mean:.2%}",
                    (rate_labels[i], mean),
                    textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "dropout_degradation.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Plot saved to {path}")


if __name__ == "__main__":
    run_dropout_evaluation()
