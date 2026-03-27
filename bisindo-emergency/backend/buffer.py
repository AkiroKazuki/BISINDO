"""
Sliding window buffer and confirmation logic for real-time inference.

SlidingWindowBuffer: Collects frames in a FIFO deque, triggers inference at stride intervals.
ConfirmationTracker: Requires 4/5 consistent predictions + high confidence before confirming.
"""

import time
from collections import deque, Counter
from typing import Optional, Tuple


class SlidingWindowBuffer:
    """FIFO sliding window buffer for frame collection.

    Collects frames and triggers inference every `stride` frames
    once the buffer is full (60 frames).
    """

    def __init__(self, window_size: int = 60, stride: int = 15):
        """
        Args:
            window_size: Number of frames to collect (60 = 2s at 30fps).
            stride: Trigger inference every N frames (15 = ~2x/sec at 30fps).
        """
        self.window_size = window_size
        self.stride = stride
        self.buffer = deque(maxlen=window_size)
        self.frame_count = 0

    def add_frame(self, keypoints) -> bool:
        """Add a frame and check if inference should be triggered.

        Args:
            keypoints: Array of shape (75, 3) for a single frame.

        Returns:
            True if inference should be triggered, False otherwise.
        """
        self.buffer.append(keypoints)
        self.frame_count += 1

        should_infer = (
            len(self.buffer) == self.window_size
            and self.frame_count % self.stride == 0
        )

        return should_infer

    def get_window(self) -> list:
        """Get the current window of frames.

        Returns:
            List of keypoint arrays (each (75, 3)), length = window_size.
        """
        return list(self.buffer)

    def is_full(self) -> bool:
        """Check if buffer has collected enough frames."""
        return len(self.buffer) == self.window_size

    def reset(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.frame_count = 0


class ConfirmationTracker:
    """Confirmation logic for gesture detection.

    Requires 4 out of 5 recent predictions to agree AND confidence > threshold
    before confirming a gesture. Implements cooldown to prevent repeated triggers.
    """

    def __init__(self, buffer_size: int = 5, min_agreement: int = 4,
                 confidence_threshold: float = 0.85, cooldown_seconds: float = 10.0):
        """
        Args:
            buffer_size: Number of recent predictions to track.
            min_agreement: Minimum predictions that must agree.
            confidence_threshold: Minimum confidence for confirmation.
            cooldown_seconds: Cooldown period after confirmation.
        """
        self.buffer_size = buffer_size
        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds

        self.predictions = deque(maxlen=buffer_size)
        self.last_confirmed_time = 0.0

    def add_prediction(self, class_name: str, confidence: float) -> Tuple[bool, Optional[str]]:
        """Add a prediction and check for confirmation.

        Args:
            class_name: Predicted class name (e.g., "TOLONG").
            confidence: Prediction confidence (0-1).

        Returns:
            Tuple of (is_confirmed, confirmed_class_name).
            If not confirmed, confirmed_class_name is None.
        """
        self.predictions.append((class_name, confidence))

        # Check cooldown
        now = time.time()
        if now - self.last_confirmed_time < self.cooldown_seconds:
            return False, None

        # Check if we have enough predictions
        if len(self.predictions) < self.min_agreement:
            return False, None

        # Check mode (most common prediction)
        class_counts = Counter(cls for cls, _ in self.predictions)
        most_common_class, count = class_counts.most_common(1)[0]

        # Check agreement threshold and confidence
        if count >= self.min_agreement and confidence > self.confidence_threshold:
            # CONFIRMED
            self.last_confirmed_time = now
            self.predictions.clear()
            return True, most_common_class

        return False, None

    def is_in_cooldown(self) -> bool:
        """Check if currently in cooldown period."""
        return time.time() - self.last_confirmed_time < self.cooldown_seconds

    def reset(self):
        """Reset all state."""
        self.predictions.clear()
        self.last_confirmed_time = 0.0
