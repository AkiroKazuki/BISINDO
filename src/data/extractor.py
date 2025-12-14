"""
MediaPipe Keypoint Extraction Module

Features:
- Holistic model for pose + hands extraction
- Batch processing with progress tracking
- Handling missing landmarks (occlusion-robust)
- Output: numpy arrays with shape (T, num_landmarks, 3)
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
import yaml
import json


class KeypointExtractor:
    """
    Extract keypoints from video using MediaPipe Holistic.
    
    Extracts pose (33 landmarks) and hand (21 landmarks each) keypoints,
    outputting normalized coordinates with visibility scores.
    """
    
    # Landmark indices for reference
    POSE_LANDMARKS = 33
    HAND_LANDMARKS = 21
    
    def __init__(self, config_path: str = "config/default.yaml"):
        """
        Initialize MediaPipe Holistic model.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.keypoint_config = self.config['keypoints']
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        
        # Calculate total landmarks
        self.num_pose = self.POSE_LANDMARKS if self.keypoint_config['use_pose'] else 0
        self.num_hand = self.HAND_LANDMARKS if self.keypoint_config['use_hands'] else 0
        self.total_landmarks = self.num_pose + (2 * self.num_hand)  # 2 hands
        
        print(f"KeypointExtractor initialized:")
        print(f"  Pose landmarks: {self.num_pose}")
        print(f"  Hand landmarks: {self.num_hand} x 2 = {self.num_hand * 2}")
        print(f"  Total landmarks: {self.total_landmarks}")
    
    def extract_from_video(
        self, 
        video_path: str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ) -> np.ndarray:
        """
        Extract keypoints from video file.
        
        Args:
            video_path: Path to video file
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
            
        Returns:
            Keypoints array with shape (T, num_landmarks, 3)
            where 3 = (x, y, visibility/confidence)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keypoints_list = []
        
        with self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        ) as holistic:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = holistic.process(frame_rgb)
                
                # Extract keypoints
                frame_keypoints = self._extract_frame_keypoints(results)
                keypoints_list.append(frame_keypoints)
        
        cap.release()
        
        keypoints = np.array(keypoints_list)
        
        # Handle missing landmarks
        keypoints = self.handle_missing_landmarks(keypoints)
        
        return keypoints
    
    def _extract_frame_keypoints(self, results) -> np.ndarray:
        """
        Extract keypoints from single frame results.
        
        Args:
            results: MediaPipe Holistic results
            
        Returns:
            Keypoints array (num_landmarks, 3)
        """
        keypoints = np.zeros((self.total_landmarks, 3))
        idx = 0
        
        # Extract pose landmarks
        if self.keypoint_config['use_pose']:
            if results.pose_landmarks:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    keypoints[idx + i] = [landmark.x, landmark.y, landmark.visibility]
            else:
                # Mark as missing (visibility = 0)
                keypoints[idx:idx + self.num_pose, 2] = 0
            idx += self.num_pose
        
        # Extract left hand landmarks
        if self.keypoint_config['use_hands']:
            if results.left_hand_landmarks:
                for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                    keypoints[idx + i] = [landmark.x, landmark.y, 1.0]
            else:
                keypoints[idx:idx + self.num_hand, 2] = 0
            idx += self.num_hand
            
            # Extract right hand landmarks
            if results.right_hand_landmarks:
                for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                    keypoints[idx + i] = [landmark.x, landmark.y, 1.0]
            else:
                keypoints[idx:idx + self.num_hand, 2] = 0
        
        return keypoints
    
    def handle_missing_landmarks(
        self, 
        keypoints: np.ndarray,
        method: str = "interpolate"
    ) -> np.ndarray:
        """
        Handle missing landmarks using interpolation or fill.
        
        Args:
            keypoints: Input keypoints (T, V, 3)
            method: "interpolate" or "zero"
            
        Returns:
            Processed keypoints with missing values handled
        """
        T, V, C = keypoints.shape
        
        if method == "zero":
            # Already zeros, just return
            return keypoints
        
        # Interpolate missing values
        for v in range(V):
            visibility = keypoints[:, v, 2]
            
            # Find frames with valid data
            valid_frames = np.where(visibility > 0.5)[0]
            
            if len(valid_frames) == 0:
                # No valid frames, keep as zeros
                continue
            elif len(valid_frames) == T:
                # All frames valid, no interpolation needed
                continue
            
            # Interpolate x and y coordinates
            for c in range(2):  # x, y only
                valid_values = keypoints[valid_frames, v, c]
                
                # Linear interpolation
                all_frames = np.arange(T)
                interpolated = np.interp(all_frames, valid_frames, valid_values)
                keypoints[:, v, c] = interpolated
                
                # Update visibility for interpolated frames
                # (keep lower confidence for interpolated values)
                keypoints[:, v, 2] = np.maximum(keypoints[:, v, 2], 0.3)
        
        return keypoints
    
    def extract_batch(
        self,
        video_paths: List[str],
        output_dir: str,
        progress: bool = True
    ) -> Dict[str, str]:
        """
        Batch extract keypoints from multiple videos.
        
        Args:
            video_paths: List of video file paths
            output_dir: Directory to save extracted keypoints
            progress: Show progress bar
            
        Returns:
            Dictionary mapping video paths to output paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        iterator = tqdm(video_paths, desc="Extracting keypoints") if progress else video_paths
        
        for video_path in iterator:
            try:
                # Extract keypoints
                keypoints = self.extract_from_video(video_path)
                
                # Generate output filename
                video_name = Path(video_path).stem
                output_path = output_dir / f"{video_name}.npy"
                
                # Save keypoints
                np.save(output_path, keypoints)
                
                # Also save metadata
                metadata = {
                    'source_video': video_path,
                    'num_frames': keypoints.shape[0],
                    'num_landmarks': keypoints.shape[1],
                    'landmark_dim': keypoints.shape[2]
                }
                metadata_path = output_dir / f"{video_name}_meta.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                results[video_path] = str(output_path)
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results[video_path] = None
        
        # Summary
        successful = sum(1 for v in results.values() if v is not None)
        print(f"\nExtraction complete: {successful}/{len(video_paths)} successful")
        
        return results
    
    def visualize_keypoints(
        self, 
        video_path: str, 
        keypoints: np.ndarray,
        output_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualize extracted keypoints on video.
        
        Args:
            video_path: Original video path
            keypoints: Extracted keypoints (T, V, 3)
            output_path: Optional path to save visualization
            show: Whether to display visualization
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer if saving
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(keypoints):
                break
            
            # Draw keypoints
            frame_kps = keypoints[frame_idx]
            frame = self._draw_keypoints(frame, frame_kps, width, height)
            
            if out:
                out.write(frame)
            
            if show:
                cv2.imshow('Keypoints Visualization', frame)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                    break
            
            frame_idx += 1
        
        cap.release()
        if out:
            out.release()
        if show:
            cv2.destroyAllWindows()
    
    def _draw_keypoints(
        self, 
        frame: np.ndarray, 
        keypoints: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        """Draw keypoints on frame."""
        # Pose connections (simplified)
        pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Face
            (0, 4), (4, 5), (5, 6), (6, 8),  # Face
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso
            (23, 24),  # Hips
        ]
        
        # Draw pose landmarks and connections
        for start, end in pose_connections:
            if start < len(keypoints) and end < len(keypoints):
                if keypoints[start, 2] > 0.3 and keypoints[end, 2] > 0.3:
                    pt1 = (int(keypoints[start, 0] * width), 
                           int(keypoints[start, 1] * height))
                    pt2 = (int(keypoints[end, 0] * width), 
                           int(keypoints[end, 1] * height))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw all keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.3:  # Visible
                x = int(kp[0] * width)
                y = int(kp[1] * height)
                
                # Color based on body part
                if i < self.num_pose:
                    color = (0, 255, 0)  # Green for pose
                elif i < self.num_pose + self.num_hand:
                    color = (255, 0, 0)  # Blue for left hand
                else:
                    color = (0, 0, 255)  # Red for right hand
                
                cv2.circle(frame, (x, y), 4, color, -1)
        
        return frame


if __name__ == "__main__":
    # Test extraction
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        extractor = KeypointExtractor()
        keypoints = extractor.extract_from_video(video_path)
        print(f"Extracted keypoints shape: {keypoints.shape}")
        
        # Visualize
        extractor.visualize_keypoints(video_path, keypoints)
    else:
        print("Usage: python extractor.py <video_path>")
