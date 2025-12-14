"""
Keypoint Preprocessing Module

Features:
1. Normalization: center to hip, scale by torso length
2. Velocity & acceleration computation
3. Skeleton-graph adjacency matrix
4. Sequence padding/truncation
"""

import numpy as np
from typing import Tuple, Optional
import yaml


class KeypointPreprocessor:
    """
    Preprocessing pipeline for keypoint sequences.
    
    Normalizes keypoints, computes motion features,
    and prepares data for model input.
    """
    
    # MediaPipe Holistic landmark indices
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    
    # Skeleton connections for pose
    POSE_CONNECTIONS = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Left arm
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        # Right arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        # Left leg
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    ]
    
    # Hand connections (same for left and right)
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17),
    ]
    
    def __init__(self, config_path: str = "config/default.yaml"):
        """
        Initialize preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        kp_config = self.config.get('keypoints', {})
        self.num_pose = 33 if kp_config.get('use_pose', True) else 0
        self.num_hand = 21 if kp_config.get('use_hands', True) else 0
        self.total_landmarks = self.num_pose + (2 * self.num_hand)
        
        training_config = self.config.get('training', {})
        self.sequence_length = training_config.get('sequence_length', 30)
        
        # Build adjacency matrix
        self.adjacency_matrix = self.build_adjacency_matrix()
    
    def preprocess(
        self, 
        keypoints: np.ndarray,
        normalize: bool = True,
        compute_velocity: bool = True,
        pad_sequence: bool = True
    ) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            keypoints: Input keypoints (T, V, 3)
            normalize: Whether to normalize coordinates
            compute_velocity: Whether to add velocity features
            pad_sequence: Whether to pad/truncate to target length
            
        Returns:
            Preprocessed keypoints
        """
        result = keypoints.copy()
        
        if normalize:
            result = self.normalize(result)
        
        if compute_velocity:
            velocity = self.compute_velocity(result)
            # Append velocity as additional channels
            result = np.concatenate([result, velocity], axis=-1)
        
        if pad_sequence:
            result = self.pad_or_truncate(result, self.sequence_length)
        
        return result
    
    def normalize(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints to standard pose.
        
        Centering: Center at hip midpoint
        Scaling: Scale by torso length (hip to shoulder)
        
        Args:
            keypoints: Input keypoints (T, V, 3)
            
        Returns:
            Normalized keypoints
        """
        result = keypoints.copy()
        T = result.shape[0]
        
        for t in range(T):
            frame = result[t]
            
            # Get hip center
            left_hip = frame[self.LEFT_HIP, :2]
            right_hip = frame[self.RIGHT_HIP, :2]
            hip_center = (left_hip + right_hip) / 2
            
            # Get shoulder center
            left_shoulder = frame[self.LEFT_SHOULDER, :2]
            right_shoulder = frame[self.RIGHT_SHOULDER, :2]
            shoulder_center = (left_shoulder + right_shoulder) / 2
            
            # Compute torso length for scaling
            torso_length = np.linalg.norm(shoulder_center - hip_center)
            
            if torso_length < 0.01:  # Avoid division by zero
                torso_length = 0.3  # Default scale
            
            # Center at hip
            result[t, :, 0] -= hip_center[0]
            result[t, :, 1] -= hip_center[1]
            
            # Scale by torso length
            result[t, :, 0] /= torso_length
            result[t, :, 1] /= torso_length
            
            # Shift to center at 0.5, 0.5 for visualization
            result[t, :, 0] += 0.5
            result[t, :, 1] += 0.5
        
        return result
    
    def compute_velocity(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Compute temporal velocity features.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            
        Returns:
            Velocity features (T, V, 2) for dx, dy
        """
        T, V, C = keypoints.shape
        
        # Compute velocity (first-order difference)
        velocity = np.zeros((T, V, 2))
        velocity[1:] = keypoints[1:, :, :2] - keypoints[:-1, :, :2]
        
        # First frame has zero velocity
        velocity[0] = 0
        
        return velocity
    
    def compute_acceleration(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Compute temporal acceleration features.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            
        Returns:
            Acceleration features (T, V, 2) for ddx, ddy
        """
        velocity = self.compute_velocity(keypoints)
        
        T, V, _ = velocity.shape
        acceleration = np.zeros((T, V, 2))
        acceleration[1:] = velocity[1:] - velocity[:-1]
        acceleration[0] = 0
        
        return acceleration
    
    def build_adjacency_matrix(self) -> np.ndarray:
        """
        Build skeleton graph adjacency matrix.
        
        Returns:
            Adjacency matrix (V, V) as numpy array
        """
        V = self.total_landmarks
        adj = np.zeros((V, V))
        
        # Add self-loops
        np.fill_diagonal(adj, 1)
        
        # Add pose connections
        for start, end in self.POSE_CONNECTIONS:
            if start < self.num_pose and end < self.num_pose:
                adj[start, end] = 1
                adj[end, start] = 1
        
        # Add left hand connections
        if self.num_hand > 0:
            offset = self.num_pose
            for start, end in self.HAND_CONNECTIONS:
                adj[offset + start, offset + end] = 1
                adj[offset + end, offset + start] = 1
            
            # Add right hand connections
            offset = self.num_pose + self.num_hand
            for start, end in self.HAND_CONNECTIONS:
                adj[offset + start, offset + end] = 1
                adj[offset + end, offset + start] = 1
            
            # Connect hands to wrists
            # Left wrist (15) to left hand root
            adj[15, self.num_pose] = 1
            adj[self.num_pose, 15] = 1
            # Right wrist (16) to right hand root
            adj[16, self.num_pose + self.num_hand] = 1
            adj[self.num_pose + self.num_hand, 16] = 1
        
        # Normalize adjacency matrix (symmetric normalization)
        degree = adj.sum(axis=1)
        degree = np.where(degree > 0, degree, 1)  # Avoid division by zero
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return adj_normalized
    
    def pad_or_truncate(
        self, 
        keypoints: np.ndarray, 
        target_length: int
    ) -> np.ndarray:
        """
        Pad or truncate sequence to target length.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            target_length: Target sequence length
            
        Returns:
            Keypoints with shape (target_length, V, C)
        """
        T, V, C = keypoints.shape
        
        if T == target_length:
            return keypoints
        elif T > target_length:
            # Truncate (take center portion)
            start = (T - target_length) // 2
            return keypoints[start:start + target_length]
        else:
            # Pad with zeros
            result = np.zeros((target_length, V, C))
            start = (target_length - T) // 2
            result[start:start + T] = keypoints
            return result
    
    def resample_sequence(
        self, 
        keypoints: np.ndarray, 
        target_length: int
    ) -> np.ndarray:
        """
        Resample sequence to target length using interpolation.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            target_length: Target sequence length
            
        Returns:
            Resampled keypoints (target_length, V, C)
        """
        T, V, C = keypoints.shape
        
        if T == target_length:
            return keypoints
        
        # Interpolate each landmark and channel
        old_indices = np.arange(T)
        new_indices = np.linspace(0, T - 1, target_length)
        
        result = np.zeros((target_length, V, C))
        
        for v in range(V):
            for c in range(C):
                result[:, v, c] = np.interp(new_indices, old_indices, keypoints[:, v, c])
        
        return result
    
    def get_feature_dim(self, include_velocity: bool = True) -> int:
        """
        Get total feature dimension.
        
        Args:
            include_velocity: Whether velocity features are included
            
        Returns:
            Total feature dimension per landmark
        """
        dim = 3  # x, y, visibility
        if include_velocity:
            dim += 2  # dx, dy
        return dim
    
    def flatten_for_input(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Flatten keypoints for model input.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            
        Returns:
            Flattened features (T, V*C)
        """
        T, V, C = keypoints.shape
        return keypoints.reshape(T, V * C)


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = KeypointPreprocessor()
    
    # Create dummy keypoints
    T, V = 45, preprocessor.total_landmarks
    keypoints = np.random.rand(T, V, 3) * 0.5 + 0.25  # Center around 0.5
    keypoints[:, :, 2] = 1.0  # All visible
    
    print(f"Input shape: {keypoints.shape}")
    
    # Test preprocessing
    processed = preprocessor.preprocess(keypoints)
    print(f"Output shape: {processed.shape}")
    
    # Test adjacency matrix
    adj = preprocessor.adjacency_matrix
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Adjacency matrix sum: {adj.sum():.2f}")
    print(f"Non-zero entries: {(adj > 0).sum()}")
    
    # Test velocity computation
    velocity = preprocessor.compute_velocity(keypoints)
    print(f"Velocity shape: {velocity.shape}")
    
    # Test flattening
    flattened = preprocessor.flatten_for_input(processed)
    print(f"Flattened shape: {flattened.shape}")
