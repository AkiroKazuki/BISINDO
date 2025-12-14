#!/usr/bin/env python3
"""
Keypoint Extraction Script

Extract keypoints from recorded videos using MediaPipe.

Usage:
    python scripts/extract_keypoints.py --input data/raw --output data/processed
    python scripts/extract_keypoints.py --video path/to/video.mp4
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.extractor import KeypointExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract keypoints from videos"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input directory containing videos or single video file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/processed',
        help='Output directory for keypoints'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Visualize extracted keypoints'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Single video to process (alternative to --input)'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = KeypointExtractor(config_path=args.config)
    
    if args.video:
        # Process single video
        video_path = Path(args.video)
        
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            sys.exit(1)
        
        print(f"Processing: {video_path}")
        keypoints = extractor.extract_from_video(str(video_path))
        
        # Save
        output_path = Path(args.output) / f"{video_path.stem}.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import numpy as np
        np.save(output_path, keypoints)
        print(f"Saved: {output_path}")
        print(f"Shape: {keypoints.shape}")
        
        # Visualize if requested
        if args.visualize:
            extractor.visualize_keypoints(str(video_path), keypoints)
    
    elif args.input:
        # Process directory
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)
        
        # Find all videos
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        videos = []
        
        for ext in video_extensions:
            videos.extend(input_dir.rglob(f'*{ext}'))
        
        if not videos:
            print(f"No videos found in {input_dir}")
            sys.exit(1)
        
        print(f"Found {len(videos)} videos")
        
        # Process each video, maintaining directory structure
        for video_path in videos:
            # Compute relative path
            rel_path = video_path.relative_to(input_dir)
            output_subdir = output_dir / rel_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Extract keypoints
            results = extractor.extract_batch(
                [str(video_path)],
                str(output_subdir)
            )
        
        print(f"\nExtraction complete!")
        print(f"Output saved to: {output_dir}")
    
    else:
        print("Error: Specify --input or --video")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
