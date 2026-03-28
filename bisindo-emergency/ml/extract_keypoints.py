"""
Keypoint extraction from video files using MediaPipe Holistic.

Processes raw video files -> normalized .npy keypoint arrays (60, 75, 3).

Usage:
    python -m ml.extract_keypoints
"""

import os
import sys
import glob

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.normalize import normalize_shoulder_center

mp_holistic = mp.solutions.holistic

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + 2 * NUM_HAND_LANDMARKS
TARGET_FRAMES = 60

RAW_DIR = os.path.join("dataset", "raw")
PROCESSED_DIR = os.path.join("dataset", "processed")
CLASSES = ["TOLONG", "BAHAYA", "KEBAKARAN"]


def extract_landmarks_from_frame(results):
    keypoints = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = [lm.x, lm.y, lm.z]
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            keypoints[NUM_POSE_LANDMARKS + i] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            keypoints[NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS + i] = [lm.x, lm.y, lm.z]
    return keypoints


def pad_or_trim(sequence, target_length=TARGET_FRAMES):
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


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames = []
    with mp_holistic.Holistic(
        static_image_mode=False, model_complexity=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            keypoints = extract_landmarks_from_frame(results)
            frames.append(keypoints)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from: {video_path}")
    sequence = np.stack(frames, axis=0).astype(np.float32)
    for t in range(sequence.shape[0]):
        sequence[t] = normalize_shoulder_center(sequence[t])
    sequence = pad_or_trim(sequence, TARGET_FRAMES)
    return sequence


def process_all_videos():
    for cls in CLASSES:
        os.makedirs(os.path.join(PROCESSED_DIR, cls), exist_ok=True)
    total_processed = 0
    total_skipped = 0
    for cls in CLASSES:
        raw_class_dir = os.path.join(RAW_DIR, cls)
        processed_class_dir = os.path.join(PROCESSED_DIR, cls)
        video_files = sorted(
            glob.glob(os.path.join(raw_class_dir, "*.mp4"))
            + glob.glob(os.path.join(raw_class_dir, "*.avi"))
            + glob.glob(os.path.join(raw_class_dir, "*.mov"))
            + glob.glob(os.path.join(raw_class_dir, "*.MOV"))
        )
        if not video_files:
            print(f"WARNING: No video files found in {raw_class_dir}")
            continue
        print(f"\nProcessing class: {cls} ({len(video_files)} videos)")
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
    print(f"Total processed: {total_processed}")
    print(f"Total skipped:   {total_skipped}")
    verify_output()


def verify_output():
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
                print(f"    WARNING: {os.path.basename(f)}: mean={mean_val:.4f} (may not be normalized)")
    if all_ok:
        print("\nAll files verified successfully!")
    else:
        print("\nSome files have issues -- check output above.")


if __name__ == "__main__":
    process_all_videos()
