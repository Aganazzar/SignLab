#!/usr/bin/env python3
"""
WLASL Dataset Processing Script

WHAT IS WLASL:
- Word-Level American Sign Language dataset
- 2000+ signs with multiple video samples per sign
- Much larger than your current dataset (15 signs)
- Videos need to be converted to landmark features

OVERVIEW OF PROCESSING PIPELINE:
1. Download WLASL from Kaggle (manual step)
2. Extract videos and metadata
3. For each video:
   - Load video frames
   - Extract MediaPipe landmarks (same as current system)
   - Save as .npy feature sequences
4. Organize into train/val/test splits
5. Update sign_vocab.json with new signs

HOW TO USE:
1. Download dataset from Kaggle:
   - Go to: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
   - Download and extract to: ./wlasl_raw/
   
2. Install kaggle API (optional):
   pip install kaggle
   
3. Process videos to features:
   python process_wlasl.py --input wlasl_raw --output wlasl_processed --max_signs 100
   
4. Train with new data:
   python train_wlasl.py

ARGUMENTS:
--input: Path to raw WLASL videos
--output: Where to save processed .npy features
--max_signs: Limit number of signs (start small, e.g., 50-100)
--min_samples: Minimum samples per sign (default: 5)
--sequence_length: Frames per sequence (default: 45, same as current)

PROCESSING TIME:
- ~100 signs × 20 videos = 2000 videos
- ~2-5 seconds per video
- Total: ~2-3 hours for 100 signs
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from realtime.feature_extractor import StreamingFeatureExtractor


def load_wlasl_metadata(wlasl_dir):
    """
    Load WLASL metadata file (usually JSON or CSV)
    
    Returns:
        dict: {sign_name: [video_paths]}
    """
    metadata_path = Path(wlasl_dir) / "WLASL_v0.3.json"
    
    if not metadata_path.exists():
        print(f"❌ Metadata not found at {metadata_path}")
        print("Looking for video files directly...")
        return discover_videos_from_folders(wlasl_dir)
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    # Parse WLASL structure
    sign_videos = defaultdict(list)
    for entry in data:
        gloss = entry.get('gloss', '').strip().lower()
        instances = entry.get('instances', [])
        
        for inst in instances:
            video_id = inst.get('video_id', '')
            if video_id:
                # Construct video path
                video_path = Path(wlasl_dir) / "videos" / f"{video_id}.mp4"
                if video_path.exists():
                    sign_videos[gloss].append(str(video_path))
    
    return sign_videos


def discover_videos_from_folders(wlasl_dir):
    """
    Fallback: Discover videos from folder structure
    Assumes structure: wlasl_raw/sign_name/*.mp4
    """
    sign_videos = defaultdict(list)
    wlasl_path = Path(wlasl_dir)
    
    for sign_folder in wlasl_path.iterdir():
        if sign_folder.is_dir():
            sign_name = sign_folder.name.lower()
            for video_file in sign_folder.glob("*.mp4"):
                sign_videos[sign_name].append(str(video_file))
    
    return sign_videos


def extract_features_from_video(video_path, extractor, target_length=45):
    """
    Extract landmark features from video file
    
    Args:
        video_path: Path to video file
        extractor: StreamingFeatureExtractor instance
        target_length: Number of frames to extract
    
    Returns:
        np.ndarray: Feature sequence (target_length, feature_dim)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    features_list = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        cap.release()
        return None
    
    # Sample frames uniformly - use sequential reading instead of seeking
    if frame_count > target_length:
        # Calculate which frames to keep
        frame_indices = set(np.linspace(0, frame_count - 1, target_length, dtype=int))
    else:
        frame_indices = set(range(frame_count))
    
    current_frame = 0
    feature_dim = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Only process frames we want to keep
        if current_frame in frame_indices:
            try:
                # Convert to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract features
                features = extractor.extract(rgb)
                
                if features is not None:
                    # Store feature dimension for padding
                    if feature_dim is None:
                        feature_dim = len(features)
                    
                    # Only add if feature dimension matches
                    if len(features) == feature_dim:
                        features_list.append(features)
            except Exception:
                # Skip problematic frames silently
                pass
        
        current_frame += 1
        
        # Early exit if we have enough frames
        if len(features_list) >= target_length:
            break
    
    cap.release()
    
    # Handle sequences with no valid features
    if len(features_list) == 0 or feature_dim is None:
        return None
    
    # Convert to numpy array
    try:
        features_array = np.array(features_list, dtype=np.float32)
    except Exception:
        return None
    
    # Pad or truncate to target length
    if len(features_array) < target_length:
        # Pad with zeros
        padding = np.zeros((target_length - len(features_array), feature_dim), dtype=np.float32)
        features_array = np.vstack([features_array, padding])
    elif len(features_array) > target_length:
        # Truncate
        features_array = features_array[:target_length]
    
    return features_array


def process_wlasl_dataset(input_dir, output_dir, max_signs=100, min_samples=5, 
                          sequence_length=45):
    """
    Main processing function
    
    Args:
        input_dir: Path to raw WLASL videos
        output_dir: Where to save processed features
        max_signs: Maximum number of signs to process
        min_samples: Minimum samples required per sign
        sequence_length: Number of frames per sequence
    """
    print("🚀 Starting WLASL Processing Pipeline\n")
    
    # Load metadata
    print("📂 Loading WLASL metadata...")
    sign_videos = load_wlasl_metadata(input_dir)
    
    # Filter signs with enough samples
    filtered_signs = {
        sign: videos 
        for sign, videos in sign_videos.items() 
        if len(videos) >= min_samples
    }
    
    print(f"✅ Found {len(filtered_signs)} signs with ≥{min_samples} samples")
    
    # Limit to max_signs
    if len(filtered_signs) > max_signs:
        filtered_signs = dict(list(filtered_signs.items())[:max_signs])
        print(f"⚠️  Limiting to {max_signs} signs")
    
    # Initialize feature extractor
    print("\n🔧 Initializing MediaPipe feature extractor...")
    extractor = StreamingFeatureExtractor()
    
    # Process each sign
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    sign_vocabulary = {}
    processed_count = 0
    
    for sign_idx, (sign_name, video_paths) in enumerate(tqdm(filtered_signs.items(), desc="Processing signs")):
        sign_folder = output_path / sign_name
        sign_folder.mkdir(exist_ok=True)
        
        valid_samples = 0
        
        for video_idx, video_path in enumerate(video_paths):
            try:
                features = extract_features_from_video(video_path, extractor, sequence_length)
                
                if features is not None:
                    # Save as .npy
                    output_file = sign_folder / f"{valid_samples}.npy"
                    np.save(output_file, features)
                    valid_samples += 1
                    
            except Exception as e:
                print(f"\n⚠️  Error processing {video_path}: {e}")
                continue
        
        if valid_samples >= min_samples:
            sign_vocabulary[sign_name] = sign_idx
            processed_count += 1
    
    # Save vocabulary
    vocab_file = output_path / "wlasl_vocab.json"
    with open(vocab_file, 'w') as f:
        json.dump(sign_vocabulary, f, indent=2)
    
    print(f"\n✅ Processing complete!")
    print(f"   - Processed {processed_count} signs")
    print(f"   - Saved to: {output_dir}")
    print(f"   - Vocabulary: {vocab_file}")
    print(f"\n📊 Feature dimension: {extractor.extract(np.zeros((480, 640, 3), dtype=np.uint8)).shape}")


def main():
    parser = argparse.ArgumentParser(description="Process WLASL dataset into landmark features")
    parser.add_argument("--input", default="wlasl_raw", help="Input directory with raw videos")
    parser.add_argument("--output", default="wlasl_processed", help="Output directory for features")
    parser.add_argument("--max_signs", type=int, default=100, help="Maximum signs to process")
    parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples per sign")
    parser.add_argument("--sequence_length", type=int, default=45, help="Frames per sequence")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input directory not found: {args.input}")
        print("\n📥 Download WLASL dataset:")
        print("   1. Go to: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed")
        print("   2. Download and extract to ./wlasl_raw/")
        print("   3. Run this script again")
        return
    
    process_wlasl_dataset(
        args.input,
        args.output,
        args.max_signs,
        args.min_samples,
        args.sequence_length
    )


if __name__ == "__main__":
    main()
