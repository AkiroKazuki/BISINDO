"""
Graph definition for the 75-node MediaPipe Holistic skeleton.

Defines edges for pose (33), left hand (21), right hand (21),
plus cross-body connections between pose wrists and hand wrists.

Usage:
    from ml.graph import build_adjacency_matrix
    A_hat = build_adjacency_matrix()  # Returns (75, 75) torch.Tensor
"""

import numpy as np
import torch


NUM_NODES = 75


def get_pose_edges() -> list:
    """MediaPipe Pose connections (33 nodes, index 0-32).

    Based on the standard MediaPipe Pose topology.
    """
    return [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),    # Right eye
        (0, 4), (4, 5), (5, 6), (6, 8),    # Left eye
        (9, 10),                             # Mouth
        # Torso
        (11, 12),                            # Shoulders
        (11, 23), (12, 24), (23, 24),       # Shoulders to hips, hips
        # Right arm
        (11, 13), (13, 15),                  # Right shoulder → elbow → wrist
        (15, 17), (15, 19), (15, 21),       # Right wrist → fingers
        (17, 19),
        # Left arm
        (12, 14), (14, 16),                  # Left shoulder → elbow → wrist
        (16, 18), (16, 20), (16, 22),       # Left wrist → fingers
        (18, 20),
        # Right leg
        (23, 25), (25, 27),                  # Right hip → knee → ankle
        (27, 29), (27, 31), (29, 31),       # Right ankle → foot
        # Left leg
        (24, 26), (26, 28),                  # Left hip → knee → ankle
        (28, 30), (28, 32), (30, 32),       # Left ankle → foot
    ]


def get_hand_edges(offset: int = 0) -> list:
    """MediaPipe Hand connections (21 nodes per hand).

    Wrist → each finger base, then finger base → tip sequentially.

    Args:
        offset: Index offset (33 for left hand, 54 for right hand).
    """
    edges = [
        # Wrist to finger bases
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # Thumb
        (1, 2), (2, 3), (3, 4),
        # Index
        (5, 6), (6, 7), (7, 8),
        # Middle
        (9, 10), (10, 11), (11, 12),
        # Ring
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (17, 18), (18, 19), (19, 20),
    ]
    return [(i + offset, j + offset) for i, j in edges]


def get_cross_body_edges() -> list:
    """Cross-body connections linking pose wrists to hand wrists.

    CRITICAL: Without these edges, information cannot flow between
    the pose skeleton and hand detail graphs.
    """
    return [
        (15, 33),   # Pose left wrist → Left hand wrist
        (16, 54),   # Pose right wrist → Right hand wrist
    ]


def get_all_edges() -> list:
    """Get complete edge list for the 75-node graph."""
    edges = []
    edges.extend(get_pose_edges())             # Pose: 0-32
    edges.extend(get_hand_edges(offset=33))    # Left hand: 33-53
    edges.extend(get_hand_edges(offset=54))    # Right hand: 54-74
    edges.extend(get_cross_body_edges())       # Cross-body links
    return edges


def build_adjacency_matrix() -> torch.Tensor:
    """Build symmetrically normalized adjacency matrix for the skeleton graph.

    Returns:
        A_hat: Tensor of shape (75, 75) — normalized adjacency matrix.
               Should be registered as a buffer (not parameter) in the model.
    """
    # 1. Initialize
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

    # 2. Fill edges (undirected)
    for i, j in get_all_edges():
        A[i][j] = 1.0
        A[j][i] = 1.0

    # 3. Self-loops
    A += np.eye(NUM_NODES, dtype=np.float32)

    # 4. Degree matrix
    D = np.diag(np.sum(A, axis=1))

    # 5. D^(-1/2)
    D_inv_sqrt = np.zeros_like(D)
    for i in range(NUM_NODES):
        if D[i][i] > 0:
            D_inv_sqrt[i][i] = D[i][i] ** (-0.5)

    # 6. Symmetric normalization: A_hat = D^(-1/2) @ A @ D^(-1/2)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.from_numpy(A_hat)


if __name__ == "__main__":
    A = build_adjacency_matrix()
    print(f"Adjacency matrix shape: {A.shape}")
    print(f"Symmetric: {torch.allclose(A, A.T)}")
    print(f"Non-zero entries: {(A != 0).sum().item()}")
    print(f"Total edges: {len(get_all_edges())}")
    print(f"Self-loops: {NUM_NODES}")
