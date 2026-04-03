"""
PyTorch Dataset and DataLoader for BISINDO keypoint sequences.

Handles stratified train/val/test splitting and data loading.

Usage:
    python -m ml.dataset          # Generate splits
    python -m ml.dataset --verify # Verify splits
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Constants
CLASSES = ["TOLONG", "BAHAYA", "KEBAKARAN"]
LABEL_MAP = {"TOLONG": 0, "BAHAYA": 1, "KEBAKARAN": 2}
PROCESSED_DIR = os.path.join("dataset", "processed")
SPLIT_FILE = "split_indices.json"
RANDOM_SEED = 42
BATCH_SIZE = 32


class BISINDODataset(Dataset):
    """PyTorch Dataset for BISINDO keypoint sequences.

    Returns tensors of shape (3, 60, 75) — (channel, time, node).
    """

    def __init__(self, file_paths: list, labels: list):
        """
        Args:
            file_paths: List of .npy file paths.
            labels: List of integer labels corresponding to file_paths.
        """
        assert len(file_paths) == len(labels), \
            f"Mismatch: {len(file_paths)} files, {len(labels)} labels"
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load (60, 75, 3)
        arr = np.load(self.file_paths[idx]).astype(np.float32)

        # Transpose to (3, 60, 75) — (channel, time, node)
        arr = np.transpose(arr, (2, 0, 1))

        tensor = torch.from_numpy(arr)
        label = self.labels[idx]

        return tensor, label


def collect_original_files() -> tuple:
    """Collect all original (non-augmented) .npy files and their labels.

    Returns:
        (file_paths, labels) — lists of file paths and integer labels.
    """
    file_paths = []
    labels = []

    for cls in CLASSES:
        class_dir = os.path.join(PROCESSED_DIR, cls)
        npy_files = sorted(glob.glob(os.path.join(class_dir, f"{cls}_*.npy")))

        # Filter out augmented files
        original_files = [f for f in npy_files if "_aug" not in os.path.basename(f)]

        for f in original_files:
            file_paths.append(f)
            labels.append(LABEL_MAP[cls])

    return file_paths, labels


def generate_splits():
    """Generate stratified train/val/test splits and save to JSON."""
    file_paths, labels = collect_original_files()

    if len(file_paths) == 0:
        print("No processed .npy files found.")
        print(f"  Looking in: {os.path.abspath(PROCESSED_DIR)}")
        print("  Run `python -m ml.extract_keypoints` first.")
        sys.exit(1)

    print(f"Found {len(file_paths)} original samples across {len(CLASSES)} classes")

    # First split: train (70%) vs temp (30%)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    # Second split: val (50% of temp = 15%) vs test (50% of temp = 15%)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_labels
    )

    # Find augmented files for train set
    train_aug_files = []
    train_aug_labels = []
    for filepath, label in zip(train_files, train_labels):
        base, ext = os.path.splitext(filepath)
        for i in range(1, 11):  # _aug1 to _aug10
            aug_path = f"{base}_aug{i}{ext}"
            if os.path.exists(aug_path):
                train_aug_files.append(aug_path)
                train_aug_labels.append(label)

    # Save split info
    split_info = {
        "random_seed": RANDOM_SEED,
        "label_map": LABEL_MAP,
        "train_files": train_files,
        "train_labels": train_labels,
        "train_aug_files": train_aug_files,
        "train_aug_labels": train_aug_labels,
        "val_files": val_files,
        "val_labels": val_labels,
        "test_files": test_files,
        "test_labels": test_labels,
    }

    with open(SPLIT_FILE, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplits saved to {SPLIT_FILE}")
    print(f"  Train: {len(train_files)} original + {len(train_aug_files)} augmented "
          f"= {len(train_files) + len(train_aug_files)}")
    print(f"  Val:   {len(val_files)} (original only)")
    print(f"  Test:  {len(test_files)} (original only)")

    # Print per-class breakdown
    for cls, label in LABEL_MAP.items():
        n_train = sum(1 for l in train_labels if l == label)
        n_val = sum(1 for l in val_labels if l == label)
        n_test = sum(1 for l in test_labels if l == label)
        print(f"  {cls}: train={n_train}, val={n_val}, test={n_test}")


def load_splits(include_augmented: bool = True) -> dict:
    """Load split indices from JSON file.

    Args:
        include_augmented: If True, include augmented files in train set.

    Returns:
        Dict with 'train', 'val', 'test' keys, each containing (files, labels).
    """
    if not os.path.exists(SPLIT_FILE):
        print(f"Split file not found: {SPLIT_FILE}")
        print("  Run `python -m ml.dataset` first.")
        sys.exit(1)

    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)

    train_files = splits["train_files"]
    train_labels = splits["train_labels"]

    if include_augmented:
        train_files = train_files + splits.get("train_aug_files", [])
        train_labels = train_labels + splits.get("train_aug_labels", [])

    return {
        "train": (train_files, train_labels),
        "val": (splits["val_files"], splits["val_labels"]),
        "test": (splits["test_files"], splits["test_labels"]),
    }


def create_dataloaders(batch_size: int = BATCH_SIZE, include_augmented: bool = True) -> dict:
    """Create PyTorch DataLoaders for train/val/test.

    Returns:
        Dict with 'train', 'val', 'test' DataLoader instances.
    """
    splits = load_splits(include_augmented)

    loaders = {}
    for split_name, (files, labels) in splits.items():
        dataset = BISINDODataset(files, labels)
        shuffle = (split_name == "train")
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False
        )

    return loaders


def verify_splits():
    """Verify the integrity of saved splits."""
    splits = load_splits(include_augmented=False)

    print("Verifying splits...\n")
    for split_name, (files, labels) in splits.items():
        missing = [f for f in files if not os.path.exists(f)]
        print(f"  {split_name}: {len(files)} files, {len(missing)} missing")

        if missing:
            for m in missing[:5]:
                print(f"    MISSING: {m}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")

    # Check no overlap
    train_set = set(splits["train"][0])
    val_set = set(splits["val"][0])
    test_set = set(splits["test"][0])

    assert train_set.isdisjoint(val_set), "Train/val overlap!"
    assert train_set.isdisjoint(test_set), "Train/test overlap!"
    assert val_set.isdisjoint(test_set), "Val/test overlap!"
    print("\nNo overlap between splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BISINDO Dataset Manager")
    parser.add_argument("--verify", action="store_true", help="Verify existing splits")
    args = parser.parse_args()

    if args.verify:
        verify_splits()
    else:
        generate_splits()
