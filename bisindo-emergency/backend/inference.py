"""
Real-time inference pipeline for BISINDO ST-GCN.

Normalizes incoming frames using the SAME shared normalization as training,
then runs ST-GCN forward pass with softmax for confidence scores.
"""

import os
import sys
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.graph import build_adjacency_matrix
from ml.model import STGCN
from utils.normalize import normalize_shoulder_center
from backend.buffer import SlidingWindowBuffer, ConfirmationTracker, MotionDetector


logger = logging.getLogger(__name__)

# Label mapping (same as training)
LABEL_MAP = {0: "TOLONG", 1: "BAHAYA", 2: "KEBAKARAN"}
CHECKPOINT_PATH = os.path.join("ml", "checkpoints", "best_model.pt")


class InferencePipeline:
    """Real-time gesture inference pipeline.

    Combines sliding window buffer, ST-GCN inference, and confirmation logic.
    """

    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH,
                 device: Optional[str] = None):
        """
        Args:
            checkpoint_path: Path to best_model.pt checkpoint.
            device: Torch device string ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        # Device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model = None
        self.model_loaded = False
        self._load_model(checkpoint_path)

        # Buffer, motion detection, and confirmation
        self.buffer = SlidingWindowBuffer(window_size=60, stride=15)
        self.motion_detector = MotionDetector(motion_threshold=0.15)
        self.confirmation = ConfirmationTracker(
            buffer_size=5,
            min_agreement=4,
            confidence_threshold=0.85,
            cooldown_seconds=10.0
        )

    def _load_model(self, checkpoint_path: str):
        """Load ST-GCN model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Model checkpoint not found: {checkpoint_path}")
            logger.warning("  Inference will return dummy results until model is trained.")
            return

        try:
            A = build_adjacency_matrix().to(self.device)
            self.model = STGCN(num_classes=3, A=A).to(self.device)

            checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                    weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_loaded = True

            logger.info(f"Model loaded from {checkpoint_path} "
                        f"(val_acc: {checkpoint.get('val_accuracy', 'N/A')})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False

    def process_frame(self, keypoints: np.ndarray) -> dict:
        """Process a single frame of keypoints.

        Args:
            keypoints: Array of shape (75, 3) — raw keypoint coordinates.

        Returns:
            Dict with keys:
                - 'class': predicted class name or None
                - 'confidence': confidence score or 0.0
                - 'is_confirmed': whether gesture is confirmed
                - 'in_cooldown': whether in cooldown period
                - 'buffer_full': whether buffer has 60 frames
        """
        # Normalize using shared normalization
        normalized = normalize_shoulder_center(keypoints)

        # Add to buffer
        should_infer = self.buffer.add_frame(normalized)

        result = {
            'class': None,
            'confidence': 0.0,
            'is_confirmed': False,
            'in_cooldown': self.confirmation.is_in_cooldown(),
            'buffer_full': self.buffer.is_full(),
        }

        if not should_infer:
            return result

        if not self.model_loaded:
            # Return dummy result if model not loaded
            result['class'] = 'NO_MODEL'
            return result

        # Check for sufficient hand/arm movement before inference
        window = self.buffer.get_window()
        if not self.motion_detector.has_sufficient_motion(window):
            # User is idle — do NOT force a class prediction
            result['class'] = None
            result['confidence'] = 0.0
            return result

        # Run inference
        class_name, confidence = self._infer()

        # Add to confirmation tracker
        is_confirmed, confirmed_class = self.confirmation.add_prediction(
            class_name, confidence
        )

        result['class'] = class_name
        result['confidence'] = confidence
        result['is_confirmed'] = is_confirmed

        if is_confirmed:
            result['class'] = confirmed_class
            logger.info(f"CONFIRMED: {confirmed_class} (confidence: {confidence:.2%})")

        return result

    @torch.no_grad()
    def _infer(self) -> Tuple[str, float]:
        """Run ST-GCN inference on current buffer.

        Returns:
            (class_name, confidence) tuple.
        """
        # Stack buffer -> (60, 75, 3)
        window = self.buffer.get_window()
        sequence = np.stack(window, axis=0).astype(np.float32)

        # [DIAGNOSTIC] DUMP THE SEQUENCE
        np.save("live_buffer_dump.npy", sequence)

        # Transpose -> (3, 60, 75)
        sequence = np.transpose(sequence, (2, 0, 1))

        # Unsqueeze -> (1, 3, 60, 75)
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(tensor)

        # Softmax for confidence
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

        class_idx = predicted.item()
        class_name = LABEL_MAP[class_idx]
        conf = confidence.item()

        return class_name, conf

    def reset(self):
        """Reset all state (buffer + confirmation)."""
        self.buffer.reset()
        self.confirmation.reset()
