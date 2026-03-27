"""
Data augmentation for BISINDO keypoint sequences.

Augmentations are applied to numpy arrays (NOT video) and only to the train split.
Each original sample generates 4 augmented copies → 5x total data.

Usage:
    python -m ml.augment
"""

import os
import sys
import json
import glob

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


# Constants
TARGET_FRAMES = 60
NUM_LANDMARKS = 75

CLASSES = ["TOLONG", "BAHAYA", "KEBAKARAN"]
PROCESSED_DIR = os.path.join("dataset", "processed")


def gaussian_noise(sequence: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to simulate MediaPipe detection imprecision.

    Args:
        sequence: Array of shape (60, 75, 3).
        std: Standard deviation of noise.

    Returns:
        Augmented array of same shape.
    """
    noise = np.random.normal(0, std, sequence.shape).astype(np.float32)
    return sequence + noise


def time_warp(sequence: np.ndarray, factor_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """Warp temporal axis to simulate speed variation.

    Args:
        sequence: Array of shape (60, 75, 3).
        factor_range: Min and max warp factor.

    Returns:
        Augmented array of shape (60, 75, 3).
    """
    T = sequence.shape[0]
    factor = np.random.uniform(*factor_range)
    warped_length = int(T * factor)

    if warped_length < 2:
        warped_length = 2

    # Original time axis
    original_time = np.arange(T)
    # Warped time axis
    warped_time = np.linspace(0, T - 1, warped_length)

    # Flatten spatial dims for interpolation: (T, 75*3)
    flat = sequence.reshape(T, -1)

    # Interpolate along time axis
    interpolator = interp1d(original_time, flat, axis=0, kind='linear',
                            fill_value='extrapolate')
    warped_flat = interpolator(warped_time)

    # Reshape back: (warped_length, 75, 3)
    warped = warped_flat.reshape(warped_length, NUM_LANDMARKS, 3)

    # Resize back to TARGET_FRAMES using interpolation
    if warped_length != TARGET_FRAMES:
        final_time = np.linspace(0, warped_length - 1, TARGET_FRAMES)
        warped_flat2 = warped.reshape(warped_length, -1)
        interpolator2 = interp1d(np.arange(warped_length), warped_flat2,
                                 axis=0, kind='linear', fill_value='extrapolate')
        result = interpolator2(final_time).reshape(TARGET_FRAMES, NUM_LANDMARKS, 3)
    else:
        result = warped

    return result.astype(np.float32)


def spatial_scale(sequence: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """Scale all coordinates uniformly to simulate distance variation.

    The SAME scale factor is applied to all frames in the sequence.

    Args:
        sequence: Array of shape (60, 75, 3).
        scale_range: Min and max scale factor.

    Returns:
        Augmented array of same shape.
    """
    factor = np.random.uniform(*scale_range)
    return (sequence * factor).astype(np.float32)


def temporal_jitter(sequence: np.ndarray, std: float = 0.005) -> np.ndarray:
    """Add temporally consistent jitter to simulate natural hand tremor.

    Generates noise at start and end of sequence, linearly interpolates
    between them for smooth, consistent noise.

    Args:
        sequence: Array of shape (60, 75, 3).
        std: Standard deviation for start/end noise.

    Returns:
        Augmented array of same shape.
    """
    T = sequence.shape[0]
    shape = (NUM_LANDMARKS, 3)

    # Random noise at start and end
    n_start = np.random.normal(0, std, shape).astype(np.float32)
    n_end = np.random.normal(0, std, shape).astype(np.float32)

    # Linear interpolation from start to end over T frames
    weights = np.linspace(0, 1, T).reshape(T, 1, 1)
    noise = n_start * (1 - weights) + n_end * weights

    return (sequence + noise).astype(np.float32)


def augment_sample(sequence: np.ndarray) -> list:
    """Generate 4 augmented versions of a single sample.

    Returns:
        List of 4 augmented arrays, each (60, 75, 3).
    """
    return [
        gaussian_noise(sequence),
        time_warp(sequence),
        spatial_scale(sequence),
        temporal_jitter(sequence),
    ]


def augment_train_split(split_file: str = "split_indices.json"):
    """Augment all samples in the train split.

    Reads split_indices.json to identify train files, generates 4 augmented
    copies per file, saves with _aug1..4 suffix.
    """
    # Load split indices
    if not os.path.exists(split_file):
        print(f"✗ Split file not found: {split_file}")
        print("  Run `python -m ml.dataset` first to generate splits.")
        sys.exit(1)

    with open(split_file, 'r') as f:
        splits = json.load(f)

    train_files = splits.get("train_files", [])
    if not train_files:
        print("✗ No train files found in split indices.")
        sys.exit(1)

    print(f"🔄 Augmenting {len(train_files)} train samples (4 augmentations each)...")

    total_created = 0
    for filepath in tqdm(train_files, desc="Augmenting"):
        if not os.path.exists(filepath):
            print(f"  ⚠ File not found: {filepath}")
            continue

        sequence = np.load(filepath)
        augmented = augment_sample(sequence)

        # Save augmented versions
        base, ext = os.path.splitext(filepath)
        for i, aug_seq in enumerate(augmented, start=1):
            aug_path = f"{base}_aug{i}{ext}"
            np.save(aug_path, aug_seq)
            total_created += 1

    print(f"\n✓ Created {total_created} augmented files.")
    print(f"  Total dataset size: {len(train_files)} original + {total_created} augmented "
          f"= {len(train_files) + total_created} train samples")


if __name__ == "__main__":
    augment_train_split()
