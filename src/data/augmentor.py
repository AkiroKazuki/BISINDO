"""
Data Augmentation Module for Robustness

Augmentation Types:
1. Spatial: horizontal flip, random rotation, scale
2. Temporal: time warping, speed variation
3. Noise: Gaussian noise on keypoints
4. Dropout: random landmark dropout (simulating occlusion)
"""

import numpy as np
from typing import Tuple, List, Optional
import yaml


class KeypointAugmentor:
    """
    Augmentation pipeline for keypoint sequences.
    
    Applies various augmentations to improve model robustness
    to occlusion, noise, and timing variations.
    """
    
    # Mapping for horizontal flip (left-right swap)
    # Based on MediaPipe pose landmark indices
    POSE_FLIP_PAIRS = [
        (1, 4), (2, 5), (3, 6),  # Eyes
        (7, 8),  # Ears
        (9, 10),  # Mouth
        (11, 12),  # Shoulders
        (13, 14),  # Elbows
        (15, 16),  # Wrists
        (17, 18), (19, 20), (21, 22),  # Fingers
        (23, 24),  # Hips
        (25, 26),  # Knees
        (27, 28),  # Ankles
        (29, 30), (31, 32),  # Feet
    ]
    
    def __init__(self, config_path: str = "config/default.yaml"):
        """
        Initialize augmentor with config.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aug_config = self.config.get('augmentation', {})
        self.enabled = self.aug_config.get('enabled', True)
        
        # Augmentation parameters
        self.brightness_range = self.aug_config.get('brightness_range', [0.6, 1.4])
        self.blur_probability = self.aug_config.get('blur_probability', 0.2)
        self.horizontal_flip = self.aug_config.get('horizontal_flip', True)
        self.time_warp = self.aug_config.get('time_warp', True)
        self.time_warp_range = self.aug_config.get('time_warp_range', [0.8, 1.2])
        self.noise_std = self.aug_config.get('noise_std', 0.02)
        self.dropout_prob = self.aug_config.get('dropout_prob', 0.1)
        
        # Keypoint configuration
        kp_config = self.config.get('keypoints', {})
        self.num_pose = 33 if kp_config.get('use_pose', True) else 0
        self.num_hand = 21 if kp_config.get('use_hands', True) else 0
    
    def augment(
        self, 
        keypoints: np.ndarray,
        augmentations: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Apply random augmentations to keypoint sequence.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            augmentations: List of augmentations to apply, or None for random
            
        Returns:
            Augmented keypoints
        """
        if not self.enabled:
            return keypoints.copy()
        
        result = keypoints.copy()
        
        if augmentations is None:
            # Randomly select augmentations
            augmentations = []
            
            if self.horizontal_flip and np.random.random() < 0.5:
                augmentations.append('flip')
            
            if self.time_warp and np.random.random() < 0.5:
                augmentations.append('time_warp')
            
            if np.random.random() < 0.7:
                augmentations.append('noise')
            
            if np.random.random() < 0.3:
                augmentations.append('dropout')
            
            if np.random.random() < 0.3:
                augmentations.append('scale')
            
            if np.random.random() < 0.2:
                augmentations.append('rotate')
        
        # Apply augmentations
        for aug in augmentations:
            if aug == 'flip':
                result = self.flip_horizontal(result)
            elif aug == 'time_warp':
                result = self.warp_time(result)
            elif aug == 'noise':
                result = self.add_noise(result)
            elif aug == 'dropout':
                result = self.dropout_landmarks(result)
            elif aug == 'scale':
                result = self.random_scale(result)
            elif aug == 'rotate':
                result = self.random_rotate(result)
        
        return result
    
    def flip_horizontal(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Mirror keypoints horizontally.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            
        Returns:
            Horizontally flipped keypoints
        """
        result = keypoints.copy()
        
        # Flip x coordinates (assume normalized 0-1)
        result[:, :, 0] = 1.0 - result[:, :, 0]
        
        # Swap left-right landmark pairs for pose
        for left, right in self.POSE_FLIP_PAIRS:
            if left < self.num_pose and right < self.num_pose:
                result[:, [left, right]] = result[:, [right, left]]
        
        # Swap left and right hands entirely
        if self.num_hand > 0:
            left_hand_start = self.num_pose
            right_hand_start = self.num_pose + self.num_hand
            
            left_hand = result[:, left_hand_start:left_hand_start + self.num_hand].copy()
            right_hand = result[:, right_hand_start:right_hand_start + self.num_hand].copy()
            
            result[:, left_hand_start:left_hand_start + self.num_hand] = right_hand
            result[:, right_hand_start:right_hand_start + self.num_hand] = left_hand
        
        return result
    
    def warp_time(
        self, 
        keypoints: np.ndarray,
        factor_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Apply temporal warping (speed change).
        
        Args:
            keypoints: Input keypoints (T, V, C)
            factor_range: Range for speed factor (min, max)
            
        Returns:
            Time-warped keypoints
        """
        if factor_range is None:
            factor_range = self.time_warp_range
        
        T, V, C = keypoints.shape
        
        # Random speed factor
        factor = np.random.uniform(factor_range[0], factor_range[1])
        
        # New length
        new_T = int(T * factor)
        new_T = max(new_T, 10)  # Minimum length
        
        # Interpolate
        old_indices = np.linspace(0, T - 1, new_T)
        result = np.zeros((new_T, V, C))
        
        for v in range(V):
            for c in range(C):
                result[:, v, c] = np.interp(
                    old_indices,
                    np.arange(T),
                    keypoints[:, v, c]
                )
        
        return result
    
    def add_noise(
        self, 
        keypoints: np.ndarray, 
        std: Optional[float] = None
    ) -> np.ndarray:
        """
        Add Gaussian noise to keypoints.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            std: Standard deviation of noise
            
        Returns:
            Noisy keypoints
        """
        if std is None:
            std = self.noise_std
        
        result = keypoints.copy()
        
        # Add noise only to x, y coordinates (not visibility)
        noise = np.random.normal(0, std, keypoints[:, :, :2].shape)
        result[:, :, :2] += noise
        
        # Clip to valid range
        result[:, :, :2] = np.clip(result[:, :, :2], 0, 1)
        
        return result
    
    def dropout_landmarks(
        self, 
        keypoints: np.ndarray,
        prob: Optional[float] = None
    ) -> np.ndarray:
        """
        Randomly dropout landmarks to simulate occlusion.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            prob: Dropout probability per landmark
            
        Returns:
            Keypoints with some landmarks dropped
        """
        if prob is None:
            prob = self.dropout_prob
        
        result = keypoints.copy()
        T, V, C = result.shape
        
        # Random dropout mask
        dropout_mask = np.random.random((T, V)) < prob
        
        # Set dropped landmarks to zero with low visibility
        result[dropout_mask, :2] = 0
        result[dropout_mask, 2] = 0.1  # Low visibility
        
        return result
    
    def random_scale(
        self, 
        keypoints: np.ndarray,
        scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Apply random spatial scaling.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            scale_range: Range for scale factor
            
        Returns:
            Scaled keypoints
        """
        result = keypoints.copy()
        
        # Random scale factor
        scale = np.random.uniform(scale_range[0], scale_range[1])
        
        # Find center (mean of all keypoints)
        visible = result[:, :, 2] > 0.5
        if visible.sum() > 0:
            center_x = result[:, :, 0][visible].mean()
            center_y = result[:, :, 1][visible].mean()
        else:
            center_x, center_y = 0.5, 0.5
        
        # Scale around center
        result[:, :, 0] = (result[:, :, 0] - center_x) * scale + center_x
        result[:, :, 1] = (result[:, :, 1] - center_y) * scale + center_y
        
        # Clip to valid range
        result[:, :, :2] = np.clip(result[:, :, :2], 0, 1)
        
        return result
    
    def random_rotate(
        self, 
        keypoints: np.ndarray,
        angle_range: Tuple[float, float] = (-15, 15)
    ) -> np.ndarray:
        """
        Apply random 2D rotation.
        
        Args:
            keypoints: Input keypoints (T, V, C)
            angle_range: Range for rotation angle in degrees
            
        Returns:
            Rotated keypoints
        """
        result = keypoints.copy()
        
        # Random rotation angle
        angle = np.random.uniform(angle_range[0], angle_range[1])
        angle_rad = np.deg2rad(angle)
        
        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Find center
        visible = result[:, :, 2] > 0.5
        if visible.sum() > 0:
            center_x = result[:, :, 0][visible].mean()
            center_y = result[:, :, 1][visible].mean()
        else:
            center_x, center_y = 0.5, 0.5
        
        # Translate to origin
        x = result[:, :, 0] - center_x
        y = result[:, :, 1] - center_y
        
        # Rotate
        result[:, :, 0] = x * cos_a - y * sin_a + center_x
        result[:, :, 1] = x * sin_a + y * cos_a + center_y
        
        # Clip to valid range
        result[:, :, :2] = np.clip(result[:, :, :2], 0, 1)
        
        return result
    
    def augment_batch(
        self, 
        keypoints_list: List[np.ndarray],
        num_augmented: int = 5
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Generate multiple augmented versions of each sample.
        
        Args:
            keypoints_list: List of keypoint arrays
            num_augmented: Number of augmented versions per sample
            
        Returns:
            List of (keypoints, augmentation_description) tuples
        """
        results = []
        
        for kps in keypoints_list:
            # Original
            results.append((kps.copy(), 'original'))
            
            # Augmented versions
            for i in range(num_augmented):
                aug_kps = self.augment(kps)
                results.append((aug_kps, f'augmented_{i}'))
        
        return results


if __name__ == "__main__":
    # Test augmentations
    import matplotlib.pyplot as plt
    
    # Create dummy keypoints
    T, V, C = 30, 75, 3
    keypoints = np.random.rand(T, V, C)
    keypoints[:, :, 2] = 1.0  # All visible
    
    augmentor = KeypointAugmentor()
    
    # Test each augmentation
    augmentations = ['flip', 'time_warp', 'noise', 'dropout', 'scale', 'rotate']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(keypoints[:, :, 0].T, aspect='auto')
    axes[0].set_title('Original')
    
    for i, aug in enumerate(augmentations):
        aug_kps = augmentor.augment(keypoints, [aug])
        axes[i + 1].imshow(aug_kps[:, :, 0].T, aspect='auto')
        axes[i + 1].set_title(f'{aug}')
    
    # Random combination
    aug_kps = augmentor.augment(keypoints)
    axes[7].imshow(aug_kps[:, :, 0].T, aspect='auto')
    axes[7].set_title('Random Combo')
    
    plt.tight_layout()
    plt.savefig('augmentation_test.png')
    print("Augmentation test saved to augmentation_test.png")
