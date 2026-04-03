"""
Training script for BISINDO ST-GCN model.

Config: Adam(lr=0.001, wd=1e-4), ReduceLROnPlateau, CrossEntropyLoss,
max 200 epochs, early stopping patience=20, checkpoint best val_accuracy.

Usage:
    python -m ml.train
"""

import os
import sys
import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.graph import build_adjacency_matrix
from ml.model import STGCN
from ml.dataset import create_dataloaders


# Configuration
CONFIG = {
    "num_classes": 3,
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "max_epochs": 200,
    "early_stopping_patience": 20,
    "scheduler_patience": 10,
    "scheduler_factor": 0.5,
    "checkpoint_dir": os.path.join("ml", "checkpoints"),
    "log_file": "training_log.csv",
}


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


def train():
    """Main training loop."""
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    loaders = create_dataloaders(batch_size=CONFIG["batch_size"])
    print(f"  Train: {len(loaders['train'].dataset)} samples")
    print(f"  Val:   {len(loaders['val'].dataset)} samples")
    print(f"  Test:  {len(loaders['test'].dataset)} samples")

    # Model
    print("\nBuilding model...")
    A = build_adjacency_matrix().to(device)
    model = STGCN(num_classes=CONFIG["num_classes"], A=A).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG["lr"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"]
    )

    # Checkpoint directory
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")

    # CSV logger
    log_file = open(CONFIG["log_file"], 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "lr"])

    # Training state
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training (max {CONFIG['max_epochs']} epochs)...\n")

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, loaders['train'], criterion, optimizer, device)

        # Validate
        val_loss, val_accuracy = evaluate(model, loaders['val'], criterion, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log to CSV
        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{val_accuracy:.4f}", f"{current_lr:.8f}"])
        log_file.flush()

        elapsed = time.time() - start_time

        # Print progress
        print(f"Epoch {epoch:3d}/{CONFIG['max_epochs']} | "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_acc: {val_accuracy:.2%} | "
              f"lr: {current_lr:.6f} | "
              f"time: {elapsed:.1f}s")

        # Checkpoint best model
        # We save if accuracy improves OR if accuracy is stable at 100% but the loss decreases
        if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_loss < best_val_loss):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  [BEST] New best model saved (val_acc: {val_accuracy:.2%}, val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {CONFIG['early_stopping_patience']} epochs)")
            break

        # Warning signs
        if epoch == 30 and val_accuracy < 0.40:
            print("\nWARNING: Val accuracy < 40% after 30 epochs.")
            print("  Check normalization consistency first!")
            print("  Are the .npy files correctly shaped (60, 75, 3)?")

    log_file.close()

    # Final report
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val accuracy: {best_val_accuracy:.2%}")
    print(f"  Checkpoint saved: {checkpoint_path}")
    print(f"  Training log:     {CONFIG['log_file']}")

    # Evaluate on test set with best model
    print(f"\nEvaluating best model on test set...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_accuracy = evaluate(model, loaders['test'], criterion, device)
    print(f"  Test accuracy: {test_accuracy:.2%}")
    print(f"  Test loss:     {test_loss:.4f}")


if __name__ == "__main__":
    train()
