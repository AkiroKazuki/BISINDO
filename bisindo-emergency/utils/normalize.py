"""
Shoulder-center normalization for BISINDO keypoints.

CRITICAL: This module is the SINGLE SOURCE OF TRUTH for normalization.
Used by both ml/extract_keypoints.py (training) and backend/inference.py (real-time).
Do NOT copy-paste this logic — always import from here.
"""

import numpy as np


def normalize_shoulder_center(keypoints: np.ndarray) -> np.ndarray:
    """Normalize keypoints relative to shoulder center.

    Centers all keypoints around the midpoint of left and right shoulders,
    then scales by the inter-shoulder distance.

    Args:
        keypoints: Array of shape (75, 3) — 75 landmarks with (x, y, z).
                   Indices 11 and 12 are left and right shoulder (pose landmarks).

    Returns:
        Normalized keypoints of shape (75, 3). If shoulder distance is 0
        (detection failure), returns the original keypoints unchanged.
    """
    keypoints = keypoints.copy()

    left_shoulder = keypoints[11]   # Pose landmark index 11
    right_shoulder = keypoints[12]  # Pose landmark index 12

    # Compute shoulder center
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    # Compute scale factor (inter-shoulder distance)
    scale = np.linalg.norm(left_shoulder - right_shoulder)

    # If scale is 0, skip normalization (detection failure)
    if scale < 1e-6:
        return keypoints

    # Center all keypoints
    keypoints -= shoulder_center

    # Scale by inter-shoulder distance
    keypoints /= scale

    return keypoints


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Normalize an entire sequence of keypoints frame-by-frame.

    Args:
        sequence: Array of shape (T, 75, 3) — T frames of 75 landmarks.

    Returns:
        Normalized sequence of same shape (T, 75, 3).
    """
    normalized = np.zeros_like(sequence)
    for t in range(sequence.shape[0]):
        normalized[t] = normalize_shoulder_center(sequence[t])
    return normalized
