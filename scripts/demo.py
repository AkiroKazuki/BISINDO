#!/usr/bin/env python3
"""
Demo Application

Real-time sign language detection demo with webcam.

Usage:
    python scripts/demo.py --model checkpoints/best_model.pt
    python scripts/demo.py --model checkpoints/best_model.pt --no-skeleton
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="BISINDO Sign Language Detection Demo"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID'
    )
    
    parser.add_argument(
        '--no-mirror',
        action='store_true',
        help='Disable mirror mode'
    )
    
    parser.add_argument(
        '--no-skeleton',
        action='store_true',
        help='Disable skeleton visualization'
    )
    
    parser.add_argument(
        '--no-attention',
        action='store_true',
        help='Disable attention visualization'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.7,
        help='Minimum confidence threshold for predictions'
    )
    
    parser.add_argument(
        '--fullscreen',
        action='store_true',
        help='Run in fullscreen mode'
    )
    
    parser.add_argument(
        '--record',
        type=str,
        help='Record demo to video file'
    )
    
    args = parser.parse_args()
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("BISINDO Sign Language Detection Demo")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Camera: {args.camera}")
    print(f"Mirror mode: {not args.no_mirror}")
    print(f"Skeleton: {not args.no_skeleton}")
    print(f"Attention: {not args.no_attention}")
    print(f"Confidence threshold: {args.confidence}")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reset buffer")
    print("  's' - Toggle skeleton")
    print("  'a' - Toggle attention")
    print("  'f' - Toggle fullscreen")
    print("  SPACE - Screenshot")
    print("="*60 + "\n")
    
    # Import inference module
    from src.inference.realtime import RealtimeInference
    
    # Initialize inference
    print("Loading model...")
    inference = RealtimeInference(
        model_path=str(model_path),
        config_path=args.config
    )
    
    # Override confidence threshold
    inference.confidence_threshold = args.confidence
    
    print("Starting webcam...\n")
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    # Video writer for recording
    out = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.record, fourcc, 30, (640, 480))
        print(f"Recording to: {args.record}")
    
    # Window setup
    window_name = 'BISINDO Sign Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                             cv2.WINDOW_FULLSCREEN)
    
    # State
    show_skeleton = not args.no_skeleton
    show_attention = not args.no_attention
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            
            # Process frame
            result = inference.process_frame(frame)
            
            # Draw overlay
            display = inference.draw_overlay(
                frame, result,
                show_skeleton=show_skeleton,
                show_attention=show_attention
            )
            
            # Record if enabled
            if out:
                out.write(display)
            
            # Show frame
            cv2.imshow(window_name, display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                inference.reset()
                print("Buffer reset")
            elif key == ord('s'):
                show_skeleton = not show_skeleton
                print(f"Skeleton: {'ON' if show_skeleton else 'OFF'}")
            elif key == ord('a'):
                show_attention = not show_attention
                print(f"Attention: {'ON' if show_attention else 'OFF'}")
            elif key == ord('f'):
                # Toggle fullscreen
                prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                         cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                         cv2.WINDOW_FULLSCREEN)
            elif key == ord(' '):
                # Screenshot
                screenshot_path = f"screenshot_{screenshot_count:03d}.png"
                cv2.imwrite(screenshot_path, display)
                print(f"Screenshot saved: {screenshot_path}")
                screenshot_count += 1
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("\nDemo ended.")


if __name__ == "__main__":
    main()
