"""
Create Video/GIF from Fly Sequences

This script creates animation frames and optionally compiles them into a video or GIF.

Usage:
    python create_video.py [options]

Options:
    --data PATH          Path to data file (default: fly_group_train.npy)
    --sequence ID        Sequence ID to visualize (default: first sequence)
    --start FRAME        Start frame (default: 0)
    --end FRAME          End frame (default: 500)
    --step STEP          Frame step size (default: 1, use 2-5 for faster)
    --fps FPS            Frames per second for video (default: 10)
    --output DIR         Output directory (default: animation_frames)
    --format FORMAT      Output format: video, gif, or frames (default: frames)
    --dpi DPI            Resolution (default: 100)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path
from plot_mabe_flies import plot_frame

# Define paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent.parent / "data" / "fly_data"
DEFAULT_DATA_FILE = DATA_DIR / "fly_group_train.npy"

def create_frames(data, sequence_id, start_frame, end_frame, step, output_dir, dpi=100):
    """Create individual frame images."""
    # Create subdirectory for this sequence
    seq_output_dir = os.path.join(output_dir, str(sequence_id))
    os.makedirs(seq_output_dir, exist_ok=True)
    
    frames = range(start_frame, end_frame, step)
    total_frames = len(frames)
    
    print(f"Creating {total_frames} frames...")
    print(f"Output directory: {seq_output_dir}")
    
    for i, frame_idx in enumerate(frames):
        if i % 10 == 0:
            print(f"  Progress: {i}/{total_frames} ({100*i/total_frames:.1f}%)")
        
        fig, ax = plot_frame(data, sequence_id, frame_idx=frame_idx)
        
        output_path = os.path.join(seq_output_dir, f'frame_{frame_idx:05d}.png')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Created {total_frames} frames in {seq_output_dir}/")
    return frames, seq_output_dir

def create_video(output_dir, sequence_id, start_frame, end_frame, step, fps):
    """Create video from frames using ffmpeg."""
    import subprocess
    import glob
    
    # Check if frames exist
    frame_files = sorted(glob.glob(os.path.join(output_dir, 'frame_*.png')))
    if not frame_files:
        print("✗ No frames found!")
        return False
    
    print(f"\nFound {len(frame_files)} frames")
    
    # Create animations subdirectory
    animations_dir = 'animations'
    os.makedirs(animations_dir, exist_ok=True)
    
    # Create output filename with sequence ID and frame info
    num_frames = len(frame_files)
    output_name = os.path.join(animations_dir, f"{sequence_id}_frames{start_frame}-{end_frame}_step{step}_{num_frames}frames.mp4")
    
    # Use glob pattern approach (more compatible)
    glob_pattern = os.path.join(output_dir, 'frame_*.png')
    
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),  # frame rate
        '-pattern_type', 'glob',  # use glob pattern
        '-i', glob_pattern,  # input pattern with glob
        '-c:v', 'libopenh264',  # video codec
        '-pix_fmt', 'yuv420p',  # pixel format (for compatibility)
        '-crf', '23',  # quality (lower = better, 23 is default)
        output_name
    ]
    
    print(f"\nCreating video: {output_name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Video created: {output_name}")
        file_size = os.path.getsize(output_name) / 1e6
        print(f"  Size: {file_size:.1f} MB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating video with libopenh264")
        print(e.stderr)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Video created with libopenh264: {output_name}")
            file_size = os.path.getsize(output_name) / 1e6
            print(f"  Size: {file_size:.1f} MB")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"✗ Error creating video:")
            print(e2.stderr)
            return False
    except FileNotFoundError:
        print("✗ ffmpeg not found. Please install ffmpeg:")
        print("  conda install ffmpeg")
        print("  or: sudo apt-get install ffmpeg")
        return False

def create_gif(output_dir, sequence_id, start_frame, end_frame, step, fps):
    """Create GIF from frames using imagemagick or PIL."""
    from PIL import Image
    import glob
    
    # Create animations subdirectory
    animations_dir = 'animations'
    os.makedirs(animations_dir, exist_ok=True)
    
    # Create output filename with sequence ID and frame info
    num_frames_range = len(range(start_frame, end_frame, step))
    output_name = os.path.join(animations_dir, f"{sequence_id}_frames{start_frame}-{end_frame}_step{step}_{num_frames_range}frames.gif")
    
    print(f"\nCreating GIF: {output_name}")
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(output_dir, 'frame_*.png')))
    
    if not frame_files:
        print("✗ No frames found!")
        return False
    
    # Load frames
    print(f"Loading {len(frame_files)} frames...")
    frames = [Image.open(f) for f in frame_files]
    
    # Calculate duration per frame in milliseconds
    duration = int(1000 / fps)
    
    # Save as GIF
    print(f"Saving GIF (this may take a while)...")
    frames[0].save(
        output_name,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    
    print(f"✓ GIF created: {output_name}")
    print(f"  Size: {os.path.getsize(output_name) / 1e6:.1f} MB")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create video/GIF from fly sequences')
    parser.add_argument('--data', default=str(DEFAULT_DATA_FILE), help='Path to data file')
    parser.add_argument('--sequence', default=None, help='Sequence ID (default: first)')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--end', type=int, default=500, help='End frame')
    parser.add_argument('--step', type=int, default=1, help='Frame step size')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--output', default='animation_frames', help='Output directory')
    parser.add_argument('--format', choices=['frames', 'video', 'gif'], default='frames',
                       help='Output format')
    parser.add_argument('--dpi', type=int, default=100, help='Resolution')

    args = parser.parse_args()

    # Load data
    print(f"Loading data: {args.data}")
    data = np.load(args.data, allow_pickle=True).item()
    
    # Get sequence ID
    if args.sequence is None:
        sequence_id = list(data['sequences'].keys())[0]
        print(f"Using first sequence: {sequence_id}")
    else:
        sequence_id = args.sequence
    
    # Validate frame range
    n_frames = data['sequences'][sequence_id]['keypoints'].shape[0]
    if args.end > n_frames:
        print(f"Warning: end frame {args.end} > total frames {n_frames}")
        args.end = n_frames
    
    print(f"\nSequence: {sequence_id}")
    print(f"Frame range: {args.start} to {args.end} (step={args.step})")
    print(f"Total frames to create: {len(range(args.start, args.end, args.step))}")
    
    # Create frames
    frames, seq_output_dir = create_frames(data, sequence_id, args.start, args.end, args.step, args.output, args.dpi)
    
    # Create video or GIF if requested
    if args.format == 'video':
        create_video(seq_output_dir, sequence_id, args.start, args.end, args.step, args.fps)
    elif args.format == 'gif':
        create_gif(seq_output_dir, sequence_id, args.start, args.end, args.step, args.fps)
    else:
        print("\nFrames created. To create video manually:")
        print(f"  ffmpeg -framerate {args.fps} -pattern_type glob -i '{seq_output_dir}/frame_*.png' -c:v libopenh264 -pix_fmt yuv420p {sequence_id}_animation.mp4")

if __name__ == '__main__':
    main()