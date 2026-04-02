"""
Keypoint extraction from video files using MediaPipe HolisticLandmarker (Tasks API).

Processes raw video files -> normalized .npy keypoint arrays (60, 75, 3).

Supports nested directory structures (e.g., dataset/raw/TOLONG/anin/*.mp4).
Deduplicates files that share the same basename but different extensions.

Usage:
    python -m ml.extract_keypoints
"""

import os
import sys
import glob

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python import BaseOptions
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.normalize import normalize_shoulder_center

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + 2 * NUM_HAND_LANDMARKS  # 75
TARGET_FRAMES = 60

RAW_DIR = os.path.join("dataset", "raw")
PROCESSED_DIR = os.path.join("dataset", "processed")
CLASSES = ["TOLONG", "BAHAYA", "KEBAKARAN"]
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")

# Model file -- downloaded separately
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "holistic_landmarker.task"
)


def extract_landmarks_from_result(result):
    """Extract 75 landmarks (x, y, z) from a HolisticLandmarkerResult.

    In the Tasks API, pose_landmarks is a flat list of 33 NormalizedLandmark,
    left/right_hand_landmarks are flat lists of 21 NormalizedLandmark each.

    Returns:
        Array of shape (75, 3). Missing landmarks are filled with zeros.
    """
    keypoints = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float32)

    # Pose landmarks (index 0-32) -- flat list of 33 landmarks
    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks):
            if i < NUM_POSE_LANDMARKS:
                keypoints[i] = [lm.x, lm.y, lm.z]

    # Left hand landmarks (index 33-53) -- flat list of 21 landmarks
    if result.left_hand_landmarks:
        for i, lm in enumerate(result.left_hand_landmarks):
            if i < NUM_HAND_LANDMARKS:
                keypoints[NUM_POSE_LANDMARKS + i] = [lm.x, lm.y, lm.z]

    # Right hand landmarks (index 54-74) -- flat list of 21 landmarks
    if result.right_hand_landmarks:
        for i, lm in enumerate(result.right_hand_landmarks):
            if i < NUM_HAND_LANDMARKS:
                keypoints[NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS + i] = [lm.x, lm.y, lm.z]

    return keypoints


def pad_or_trim(sequence, target_length=TARGET_FRAMES):
    """Pad (repeat last frame) or center-crop to exactly target_length frames."""
    T = sequence.shape[0]
    if T > target_length:
        start = (T - target_length) // 2
        return sequence[start: start + target_length]
    elif T < target_length:
        pad_count = target_length - T
        last_frame = sequence[-1:]
        padding = np.repeat(last_frame, pad_count, axis=0)
        return np.concatenate([sequence, padding], axis=0)
    return sequence


def collect_video_files(class_dir):
    """Recursively collect video files, deduplicating by basename.

    If both video.mov and video.mp4 exist, only the first one found is kept.
    """
    seen_basenames = set()
    unique_files = []

    for root, _, files in os.walk(class_dir):
        for filename in sorted(files):
            if not filename.lower().endswith(VIDEO_EXTENSIONS):
                continue
            stem = os.path.splitext(filename)[0].lower()
            if stem in seen_basenames:
                continue
            seen_basenames.add(stem)
            unique_files.append(os.path.join(root, filename))

    return sorted(unique_files)


def _create_landmarker():
    """Create a fresh HolisticLandmarker instance.

    A new instance is needed per video because VIDEO mode tracks internal
    state and requires monotonically increasing timestamps.
    """
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return HolisticLandmarker.create_from_options(options)


def process_video(video_path):
    """Process a single video into a normalized (60, 75, 3) keypoint array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    # Create a fresh landmarker for this video (timestamps start at 0)
    landmarker = _create_landmarker()

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame_idx * 1000.0 / fps)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        keypoints = extract_landmarks_from_result(result)
        frames.append(keypoints)
        frame_idx += 1

    cap.release()
    landmarker.close()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from: {video_path}")

    sequence = np.stack(frames, axis=0).astype(np.float32)

    # Normalize each frame
    for t in range(sequence.shape[0]):
        sequence[t] = normalize_shoulder_center(sequence[t])

    # Pad or trim to exactly 60 frames
    sequence = pad_or_trim(sequence, TARGET_FRAMES)
    return sequence


def process_all_videos():
    """Process all raw videos for all classes."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Download it with:")
        print('  curl -L -o holistic_landmarker.task "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"')
        sys.exit(1)

    for cls in CLASSES:
        os.makedirs(os.path.join(PROCESSED_DIR, cls), exist_ok=True)

    total_processed = 0
    total_skipped = 0
    total_duplicates = 0

    for cls in CLASSES:
        raw_class_dir = os.path.join(RAW_DIR, cls)
        processed_class_dir = os.path.join(PROCESSED_DIR, cls)

        video_files = collect_video_files(raw_class_dir)

        # Count duplicates
        all_count = sum(
            1 for _, _, files in os.walk(raw_class_dir)
            for f in files if f.lower().endswith(VIDEO_EXTENSIONS)
        )
        dupes = all_count - len(video_files)
        total_duplicates += dupes

        if not video_files:
            print(f"WARNING: No video files found in {raw_class_dir}")
            continue

        print(f"\nProcessing class: {cls} ({len(video_files)} unique videos"
              f"{f', {dupes} duplicates skipped' if dupes else ''})")

        for idx, video_path in enumerate(tqdm(video_files, desc=cls), start=1):
            output_filename = f"{cls}_{idx:03d}.npy"
            output_path = os.path.join(processed_class_dir, output_filename)
            try:
                sequence = process_video(video_path)
                np.save(output_path, sequence)
                total_processed += 1
            except Exception as e:
                print(f"  ERROR processing {video_path}: {e}")
                total_skipped += 1

    print(f"\n{'='*50}")
    print(f"Total processed:  {total_processed}")
    print(f"Total skipped:    {total_skipped}")
    print(f"Total duplicates: {total_duplicates}")
    verify_output()


def verify_output():
    """Verify all processed .npy files have correct shape."""
    print(f"\n{'='*50}")
    print("Verifying output files...\n")
    all_ok = True
    for cls in CLASSES:
        processed_dir = os.path.join(PROCESSED_DIR, cls)
        npy_files = sorted(glob.glob(os.path.join(processed_dir, "*.npy")))
        print(f"  {cls}: {len(npy_files)} files")
        for f in npy_files:
            arr = np.load(f)
            if arr.shape != (TARGET_FRAMES, TOTAL_LANDMARKS, 3):
                print(f"    ERROR: {os.path.basename(f)}: wrong shape {arr.shape}")
                all_ok = False
                continue
            if np.all(arr == 0):
                print(f"    ERROR: {os.path.basename(f)}: all zeros!")
                all_ok = False
                continue
            mean_val = np.abs(arr.mean())
            if mean_val > 5.0:
                print(f"    WARNING: {os.path.basename(f)}: mean={mean_val:.4f}")
    if all_ok:
        print("\nAll files verified successfully!")
    else:
        print("\nSome files have issues -- check output above.")


if __name__ == "__main__":
    process_all_videos()
