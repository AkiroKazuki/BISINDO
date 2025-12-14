"""
Video Recording Module for BISINDO Dataset Collection

Features:
- Guided recording with countdown timer
- Real-time skeleton overlay preview
- Automatic file naming convention
- Quality validation checks
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import yaml
import time


class VideoCollector:
    """
    Video collector for recording sign language samples.
    
    Provides guided recording sessions with real-time skeleton overlay,
    countdown timers, and quality validation.
    """
    
    def __init__(self, config_path: str = "config/recording.yaml"):
        """
        Initialize video collector with configuration.
        
        Args:
            config_path: Path to recording configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Video settings
        self.video_config = self.config['recording']['video']
        self.timing_config = self.config['recording']['timing']
        self.display_config = self.config['recording']['display']
        self.quality_config = self.config['recording']['quality']
        
        # Initialize MediaPipe for skeleton preview
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Session state
        self.current_session = None
        self.output_dir = None
        
    def start_session(
        self, 
        subject_id: str, 
        class_name: str,
        lighting: str = "medium",
        occlusion: str = "none",
        output_base: str = "data/raw"
    ) -> None:
        """
        Start a recording session for specific conditions.
        
        Args:
            subject_id: Subject identifier (e.g., "S01")
            class_name: Sign class name (e.g., "TOLONG")
            lighting: Lighting condition ("bright", "medium", "dark")
            occlusion: Occlusion condition ("none", "partial")
            output_base: Base directory for output
        """
        self.current_session = {
            'subject_id': subject_id,
            'class_name': class_name,
            'lighting': lighting,
            'occlusion': occlusion,
            'start_time': datetime.now(),
            'recordings': []
        }
        
        # Create output directory
        self.output_dir = Path(output_base) / subject_id / class_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Recording Session Started")
        print(f"{'='*50}")
        print(f"Subject: {subject_id}")
        print(f"Class: {class_name}")
        print(f"Lighting: {lighting}")
        print(f"Occlusion: {occlusion}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*50}\n")
        
    def record_sample(self, rep_number: int) -> Optional[str]:
        """
        Record a single sample.
        
        Args:
            rep_number: Repetition number
            
        Returns:
            Path to recorded video file, or None if failed
        """
        if self.current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        session = self.current_session
        
        # Generate filename
        filename = (
            f"{session['subject_id']}_"
            f"{session['class_name']}_"
            f"{session['lighting']}_"
            f"{session['occlusion']}_"
            f"{rep_number:02d}.mp4"
        )
        output_path = self.output_dir / filename
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_config['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_config['height'])
        cap.set(cv2.CAP_PROP_FPS, self.video_config['fps'])
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.video_config['codec'])
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.video_config['fps'],
            (self.video_config['width'], self.video_config['height'])
        )
        
        # Initialize MediaPipe
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            frames_recorded = 0
            recording = False
            countdown_start = None
            recording_start = None
            
            # Get instruction for this class
            instructions = self.config.get('protocol', {}).get('instructions', {})
            instruction = instructions.get(session['class_name'], f"Perform {session['class_name']} sign")
            
            print(f"\nRecording {rep_number}: {instruction}")
            print("Press SPACE to start countdown, ESC to cancel")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror frame if enabled
                if self.display_config['mirror_mode']:
                    frame = cv2.flip(frame, 1)
                
                # Process with MediaPipe for skeleton overlay
                if self.display_config['show_skeleton']:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(frame_rgb)
                    frame = self._draw_skeleton(frame, results)
                
                # Handle countdown phase
                if countdown_start is not None and not recording:
                    elapsed = time.time() - countdown_start
                    remaining = self.timing_config['countdown_seconds'] - elapsed
                    
                    if remaining > 0:
                        # Show countdown
                        self._draw_countdown(frame, int(remaining) + 1)
                    else:
                        # Start recording
                        recording = True
                        recording_start = time.time()
                        print("  Recording...")
                
                # Handle recording phase
                if recording:
                    elapsed = time.time() - recording_start
                    remaining = self.timing_config['recording_seconds'] - elapsed
                    
                    if remaining > 0:
                        # Record frame
                        if self.display_config['mirror_mode']:
                            # Flip back for saving
                            out.write(cv2.flip(frame, 1))
                        else:
                            out.write(frame)
                        frames_recorded += 1
                        
                        # Show recording indicator
                        self._draw_recording_indicator(frame, remaining)
                    else:
                        # Recording complete
                        break
                else:
                    # Show instruction
                    self._draw_instruction(frame, instruction)
                
                # Display frame
                cv2.imshow('BISINDO Recording', frame)
                
                # Handle key input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("  Recording cancelled")
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    if output_path.exists():
                        output_path.unlink()
                    return None
                elif key == 32 and countdown_start is None:  # SPACE
                    countdown_start = time.time()
                    print("  Countdown started...")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Validate recording
        if self.validate_recording(str(output_path)):
            print(f"  Saved: {output_path}")
            self.current_session['recordings'].append(str(output_path))
            return str(output_path)
        else:
            print(f"  Recording failed quality check, please retry")
            if output_path.exists():
                output_path.unlink()
            return None
    
    def validate_recording(self, video_path: str) -> bool:
        """
        Validate recording quality.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if recording passes quality checks
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Check minimum frames
        if frame_count < self.quality_config['min_frames']:
            print(f"  Warning: Only {frame_count} frames (min: {self.quality_config['min_frames']})")
            return False
        
        return True
    
    def _draw_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw MediaPipe skeleton on frame."""
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return frame
    
    def _draw_countdown(self, frame: np.ndarray, seconds: int) -> None:
        """Draw countdown overlay on frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Countdown number
        cv2.putText(
            frame, str(seconds),
            (w // 2 - 50, h // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10
        )
        
        cv2.putText(
            frame, "Get Ready!",
            (w // 2 - 100, h // 2 - 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
    
    def _draw_recording_indicator(self, frame: np.ndarray, remaining: float) -> None:
        """Draw recording indicator on frame."""
        h, w = frame.shape[:2]
        
        # Red recording dot
        cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
        
        # Recording text
        cv2.putText(
            frame, f"REC {remaining:.1f}s",
            (55, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
    
    def _draw_instruction(self, frame: np.ndarray, instruction: str) -> None:
        """Draw instruction overlay on frame."""
        h, w = frame.shape[:2]
        
        # Background bar
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        
        # Instruction text
        cv2.putText(
            frame, instruction,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        
        # Press space hint
        cv2.putText(
            frame, "Press SPACE to start",
            (w - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
        )
    
    def run_full_session(
        self,
        subject_id: str,
        class_name: str,
        num_reps: int = 15,
        lighting: str = "medium",
        occlusion: str = "none"
    ) -> list:
        """
        Run a full recording session with multiple repetitions.
        
        Args:
            subject_id: Subject identifier
            class_name: Sign class name
            num_reps: Number of repetitions to record
            lighting: Lighting condition
            occlusion: Occlusion condition
            
        Returns:
            List of successfully recorded video paths
        """
        self.start_session(subject_id, class_name, lighting, occlusion)
        
        successful = []
        rep = 1
        
        while len(successful) < num_reps:
            print(f"\n--- Repetition {len(successful) + 1}/{num_reps} ---")
            
            result = self.record_sample(rep)
            if result:
                successful.append(result)
            
            rep += 1
            
            # Rest between recordings
            if len(successful) < num_reps:
                print(f"Rest for {self.timing_config['rest_between_reps']} seconds...")
                time.sleep(self.timing_config['rest_between_reps'])
        
        print(f"\n{'='*50}")
        print(f"Session Complete!")
        print(f"Successfully recorded: {len(successful)}/{num_reps}")
        print(f"{'='*50}")
        
        return successful


if __name__ == "__main__":
    # Test recording
    collector = VideoCollector()
    collector.run_full_session(
        subject_id="S01",
        class_name="TOLONG",
        num_reps=3,
        lighting="medium",
        occlusion="none"
    )
