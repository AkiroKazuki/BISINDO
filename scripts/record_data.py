#!/usr/bin/env python3
"""
Data Recording Script

Guided recording of sign language samples for dataset creation.

Usage:
    python scripts/record_data.py --subject S01 --class TOLONG
    python scripts/record_data.py --subject S01 --all-classes
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.collector import VideoCollector


def main():
    parser = argparse.ArgumentParser(
        description="Record sign language samples for BISINDO dataset"
    )
    
    parser.add_argument(
        '--subject', '-s',
        type=str,
        required=True,
        help='Subject ID (e.g., S01, S02)'
    )
    
    parser.add_argument(
        '--class', '-c',
        type=str,
        dest='sign_class',
        help='Sign class to record (e.g., TOLONG, BAHAYA)'
    )
    
    parser.add_argument(
        '--all-classes',
        action='store_true',
        help='Record all 10 classes in sequence'
    )
    
    parser.add_argument(
        '--reps', '-r',
        type=int,
        default=15,
        help='Number of repetitions per class (default: 15)'
    )
    
    parser.add_argument(
        '--lighting', '-l',
        type=str,
        default='medium',
        choices=['bright', 'medium', 'dark'],
        help='Lighting condition (default: medium)'
    )
    
    parser.add_argument(
        '--occlusion', '-o',
        type=str,
        default='none',
        choices=['none', 'partial'],
        help='Occlusion condition (default: none)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/recording.yaml',
        help='Path to recording config file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory for recordings'
    )
    
    args = parser.parse_args()
    
    # Define all classes
    all_classes = [
        'TOLONG', 'BAHAYA', 'KEBAKARAN', 'SAKIT', 'GEMPA',
        'BANJIR', 'PENCURI', 'PINGSAN', 'KECELAKAAN', 'DARURAT'
    ]
    
    # Determine which classes to record
    if args.all_classes:
        classes_to_record = all_classes
    elif args.sign_class:
        if args.sign_class not in all_classes:
            print(f"Error: Unknown class '{args.sign_class}'")
            print(f"Available classes: {', '.join(all_classes)}")
            sys.exit(1)
        classes_to_record = [args.sign_class]
    else:
        print("Error: Specify --class or --all-classes")
        sys.exit(1)
    
    # Initialize collector
    try:
        collector = VideoCollector(config_path=args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default settings...")
        # Create minimal config
        import yaml
        default_config = {
            'recording': {
                'video': {'width': 640, 'height': 480, 'fps': 30, 'codec': 'mp4v'},
                'timing': {'countdown_seconds': 3, 'recording_seconds': 3, 'rest_between_reps': 2},
                'display': {'show_skeleton': True, 'show_countdown': True, 'show_instructions': True, 'mirror_mode': True},
                'quality': {'min_frames': 60, 'min_keypoint_confidence': 0.5, 'require_hands_visible': True}
            }
        }
        
        config_path = Path(args.config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        
        collector = VideoCollector(config_path=args.config)
    
    # Record each class
    print(f"\n{'='*60}")
    print(f"BISINDO Recording Session")
    print(f"{'='*60}")
    print(f"Subject: {args.subject}")
    print(f"Classes: {', '.join(classes_to_record)}")
    print(f"Repetitions: {args.reps}")
    print(f"Lighting: {args.lighting}")
    print(f"Occlusion: {args.occlusion}")
    print(f"{'='*60}\n")
    
    total_recordings = 0
    
    for sign_class in classes_to_record:
        print(f"\n>>> Recording class: {sign_class}")
        
        recordings = collector.run_full_session(
            subject_id=args.subject,
            class_name=sign_class,
            num_reps=args.reps,
            lighting=args.lighting,
            occlusion=args.occlusion
        )
        
        total_recordings += len(recordings)
        
        if len(classes_to_record) > 1:
            input("\nPress Enter to continue to next class...")
    
    print(f"\n{'='*60}")
    print(f"Recording Complete!")
    print(f"Total recordings: {total_recordings}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
