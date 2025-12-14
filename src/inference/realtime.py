"""
Real-time Inference Pipeline

Features:
- Webcam input processing
- Sliding window inference
- Smoothing predictions over time
- Confidence thresholding
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from collections import deque
import time


class RealtimeInference:
    """
    Real-time inference pipeline for sign language recognition.
    
    Processes webcam frames, extracts keypoints, and makes
    predictions using a trained model.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config/default.yaml",
        device: Optional[str] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device is None:
            device = self.config.get('training', {}).get('device', 'cpu')
        
        if device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"Inference device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Class names
        self.classes = self.config['dataset']['classes']
        self.num_classes = len(self.classes)
        
        # Keypoint configuration
        kp_config = self.config.get('keypoints', {})
        self.num_pose = 33 if kp_config.get('use_pose', True) else 0
        self.num_hand = 21 if kp_config.get('use_hands', True) else 0
        self.total_landmarks = self.num_pose + 2 * self.num_hand
        
        # Sequence configuration
        training_config = self.config.get('training', {})
        self.sequence_length = training_config.get('sequence_length', 30)
        
        # Inference configuration
        inference_config = self.config.get('inference', {})
        self.confidence_threshold = inference_config.get('confidence_threshold', 0.7)
        self.smoothing_window = inference_config.get('smoothing_window', 5)
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Buffers for temporal processing
        self.keypoint_buffer = deque(maxlen=self.sequence_length)
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        
        # State
        self.current_prediction = None
        self.current_confidence = 0.0
        self.last_attention_weights = None
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        from ..models.classifier import SignClassifier
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get config from checkpoint or use current config
        model_config = checkpoint.get('config', self.config)
        
        # Create model
        model = SignClassifier(model_config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Loaded model from {model_path}")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best val accuracy: {checkpoint.get('best_val_acc', 'unknown'):.4f}")
        
        return model
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame and update prediction.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Extract keypoints
        keypoints = self._extract_keypoints(frame)
        
        # Add to buffer
        self.keypoint_buffer.append(keypoints)
        
        # Make prediction if buffer is full
        result = {
            'class': None,
            'confidence': 0.0,
            'keypoints': keypoints,
            'attention_weights': None,
            'buffer_size': len(self.keypoint_buffer)
        }
        
        if len(self.keypoint_buffer) >= self.sequence_length // 2:
            # Pad if needed
            sequence = self._prepare_sequence()
            
            # Run inference
            pred_class, confidence, attention = self._infer(sequence)
            
            # Smooth predictions
            self.prediction_buffer.append((pred_class, confidence))
            smoothed_class, smoothed_conf = self._smooth_predictions()
            
            # Update result
            result['class'] = self.classes[smoothed_class] if smoothed_conf >= self.confidence_threshold else None
            result['confidence'] = smoothed_conf
            result['attention_weights'] = attention
            
            self.current_prediction = result['class']
            self.current_confidence = smoothed_conf
            self.last_attention_weights = attention
        
        # Track FPS
        elapsed = time.time() - start_time
        self.fps_counter.append(1.0 / (elapsed + 1e-6))
        result['fps'] = np.mean(self.fps_counter)
        
        return result
    
    def _extract_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Extract keypoints from frame using MediaPipe."""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.holistic.process(frame_rgb)
        
        # Initialize keypoints array
        keypoints = np.zeros((self.total_landmarks, 3))
        idx = 0
        
        # Extract pose landmarks
        if self.num_pose > 0:
            if results.pose_landmarks:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    keypoints[idx + i] = [landmark.x, landmark.y, landmark.visibility]
            idx += self.num_pose
        
        # Extract hand landmarks
        if self.num_hand > 0:
            # Left hand
            if results.left_hand_landmarks:
                for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                    keypoints[idx + i] = [landmark.x, landmark.y, 1.0]
            idx += self.num_hand
            
            # Right hand
            if results.right_hand_landmarks:
                for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                    keypoints[idx + i] = [landmark.x, landmark.y, 1.0]
        
        return keypoints
    
    def _prepare_sequence(self) -> torch.Tensor:
        """Prepare keypoint sequence for model input."""
        from ..data.preprocessing import KeypointPreprocessor
        
        # Convert buffer to numpy array
        keypoints = np.array(list(self.keypoint_buffer))  # (T, V, 3)
        
        # Pad or truncate
        T = keypoints.shape[0]
        if T < self.sequence_length:
            # Pad with zeros
            padding = np.zeros((self.sequence_length - T, self.total_landmarks, 3))
            keypoints = np.concatenate([padding, keypoints], axis=0)
        elif T > self.sequence_length:
            # Take latest frames
            keypoints = keypoints[-self.sequence_length:]
        
        # Compute velocity
        velocity = np.zeros((self.sequence_length, self.total_landmarks, 2))
        velocity[1:] = keypoints[1:, :, :2] - keypoints[:-1, :, :2]
        
        # Combine position and velocity
        features = np.concatenate([keypoints, velocity], axis=-1)  # (T, V, 5)
        
        # Convert to tensor
        tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, T, V, 5)
        
        return tensor.to(self.device)
    
    def _infer(self, sequence: torch.Tensor) -> Tuple[int, float, Optional[np.ndarray]]:
        """Run model inference."""
        with torch.no_grad():
            logits, attention = self.model(sequence)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_class = probs.max(dim=-1)
        
        attention_np = attention.cpu().numpy()[0] if attention is not None else None
        
        return pred_class.item(), confidence.item(), attention_np
    
    def _smooth_predictions(self) -> Tuple[int, float]:
        """Smooth predictions over time using voting."""
        if len(self.prediction_buffer) == 0:
            return 0, 0.0
        
        # Weighted voting based on confidence
        votes = {}
        total_weight = 0
        
        for pred_class, confidence in self.prediction_buffer:
            if pred_class not in votes:
                votes[pred_class] = 0
            votes[pred_class] += confidence
            total_weight += confidence
        
        if total_weight == 0:
            return 0, 0.0
        
        # Get class with highest weighted vote
        best_class = max(votes, key=votes.get)
        smoothed_confidence = votes[best_class] / total_weight
        
        return best_class, smoothed_confidence
    
    def draw_overlay(
        self,
        frame: np.ndarray,
        result: Dict,
        show_skeleton: bool = True,
        show_attention: bool = True
    ) -> np.ndarray:
        """
        Draw visualization overlay on frame.
        
        Args:
            frame: Input frame
            result: Prediction result dictionary
            show_skeleton: Whether to draw skeleton
            show_attention: Whether to show attention heatmap
            
        Returns:
            Frame with overlay
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw skeleton
        if show_skeleton and result['keypoints'] is not None:
            frame = self._draw_skeleton(frame, result['keypoints'])
        
        # Draw prediction box
        prediction = result.get('class')
        confidence = result.get('confidence', 0.0)
        
        # Background for prediction
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        
        if prediction:
            # Show prediction
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
            text = f"{prediction}: {confidence:.1%}"
            cv2.putText(frame, text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        else:
            cv2.putText(frame, "Detecting...", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 3)
        
        # Confidence bar
        bar_width = int((w - 20) * confidence)
        cv2.rectangle(frame, (10, 65), (10 + bar_width, 75), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 65), (w - 10, 75), (255, 255, 255), 1)
        
        # Buffer indicator
        buffer_pct = result.get('buffer_size', 0) / self.sequence_length
        cv2.putText(frame, f"Buffer: {buffer_pct:.0%}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # FPS
        fps = result.get('fps', 0)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Attention heatmap (timeline at bottom)
        if show_attention and result.get('attention_weights') is not None:
            frame = self._draw_attention_bar(frame, result['attention_weights'])
        
        return frame
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw skeleton on frame."""
        h, w = frame.shape[:2]
        
        # Connections
        pose_connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
        ]
        
        # Draw pose connections
        for start, end in pose_connections:
            if start < len(keypoints) and end < len(keypoints):
                if keypoints[start, 2] > 0.5 and keypoints[end, 2] > 0.5:
                    pt1 = (int(keypoints[start, 0] * w), int(keypoints[start, 1] * h))
                    pt2 = (int(keypoints[end, 0] * w), int(keypoints[end, 1] * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw landmarks
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.5:
                x = int(kp[0] * w)
                y = int(kp[1] * h)
                
                # Color by body part
                if i < self.num_pose:
                    color = (0, 255, 0)  # Green
                elif i < self.num_pose + self.num_hand:
                    color = (255, 0, 0)  # Blue (left)
                else:
                    color = (0, 0, 255)  # Red (right)
                
                cv2.circle(frame, (x, y), 4, color, -1)
        
        return frame
    
    def _draw_attention_bar(
        self, 
        frame: np.ndarray, 
        attention: np.ndarray
    ) -> np.ndarray:
        """Draw attention timeline bar at bottom of frame."""
        h, w = frame.shape[:2]
        bar_height = 30
        
        # Background
        cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        
        # Normalize attention to bar width
        attention_norm = attention / attention.max()
        bar_width = w - 20
        segment_width = bar_width / len(attention)
        
        for i, weight in enumerate(attention_norm):
            x1 = int(10 + i * segment_width)
            x2 = int(10 + (i + 1) * segment_width)
            
            # Color based on attention weight
            intensity = int(255 * weight)
            color = (0, intensity, 255 - intensity)
            
            cv2.rectangle(frame, (x1, h - bar_height + 5), 
                         (x2, h - 5), color, -1)
        
        cv2.putText(frame, "Attention", (w - 80, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def run_webcam(
        self, 
        camera_id: int = 0,
        mirror: bool = True
    ) -> None:
        """
        Run real-time inference from webcam.
        
        Args:
            camera_id: Camera device ID
            mirror: Whether to mirror the video
        """
        print("\nStarting webcam inference...")
        print("Press 'q' to quit, 'r' to reset buffer")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if mirror:
                    frame = cv2.flip(frame, 1)
                
                # Process frame
                result = self.process_frame(frame)
                
                # Draw overlay
                display = self.draw_overlay(frame, result)
                
                # Show frame
                cv2.imshow('BISINDO Sign Detection', display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.keypoint_buffer.clear()
                    self.prediction_buffer.clear()
                    print("Buffer reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
    
    def reset(self) -> None:
        """Reset all buffers and states."""
        self.keypoint_buffer.clear()
        self.prediction_buffer.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.last_attention_weights = None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-mirror', action='store_true')
    args = parser.parse_args()
    
    inference = RealtimeInference(
        model_path=args.model,
        config_path=args.config
    )
    
    inference.run_webcam(
        camera_id=args.camera,
        mirror=not args.no_mirror
    )
