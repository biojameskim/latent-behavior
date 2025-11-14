"""
Create Video from VQ-VAE Reconstruction Overlays

This script generates a video showing the evolution of VQ-VAE reconstructions
over time. It creates overlay frames at regular intervals and compiles them
into a video using FFmpeg.

Supports two overlay types:
  - sequence: Full arena view with all flies (original vs. reconstructed)
  - window: Individual fly window overlays

Example:
    # Create sequence overlay video
    python create_overlay_video.py \
        --data_file ../../../data/fly_data/fly_group_train.npy \
        --checkpoint ../training/outputs/run_11_01_25_v5/best_model.pt \
        --overlay_type sequence \
        --sequence_id <sequence_id> \
        --start_frame 0 \
        --end_frame 1000 \
        --frame_step 15 \
        --fps 10 \
        --output_dir ./overlay_videos

    # Create window overlay video
    python create_overlay_video.py \
        --data_file ../../../data/fly_data/fly_group_train.npy \
        --checkpoint ../training/outputs/run_11_01_25_v5/best_model.pt \
        --overlay_type window \
        --sequence_id <sequence_id> \
        --fly_idx 0 \
        --start_frame 0 \
        --end_frame 1000 \
        --frame_step 15 \
        --fps 10
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from flies.data.preprocessing import load_and_preprocess_for_vqvae
from flies.data.dataset import FlyKeypointDataset
from flies.visualization.reconstruction import (
    assemble_sequences,
    group_windows_by_fly,
    plot_sequence_overlay,
    plot_window_overlay,
    stitch_fly_windows,
)
from flies.visualization.denormalize import (
    denormalize_reconstructions,
    load_normalization_params_from_original,
)
from flies.vq_vae.vqvae import VQVAE
from flies.vq_vae.vqvae_unified import UnifiedVQVAE

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create overlay video from VQ-VAE reconstructions.")

    # Data and model
    parser.add_argument("--data_file", required=True, help="Path to .npy file containing fly trajectories.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained VQ-VAE checkpoint.")
    parser.add_argument(
        "--fly_split_file",
        default=None,
        help="JSON file containing fly-level splits.",
    )
    parser.add_argument(
        "--val_split_name",
        default="val",
        help="Split name to pull from --fly_split_file (default: val).",
    )

    # Overlay type and target
    parser.add_argument(
        "--overlay_type",
        required=True,
        choices=["sequence", "window"],
        help="Type of overlay: 'sequence' for full arena, 'window' for individual fly.",
    )
    parser.add_argument(
        "--sequence_id",
        required=True,
        help="Sequence ID to visualize.",
    )
    parser.add_argument(
        "--fly_idx",
        type=int,
        default=None,
        help="Fly index (required for window overlays).",
    )

    # Frame range and sampling
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame (default: 0).",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=1000,
        help="End frame (default: 1000).",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=15,
        help="Generate overlay every N frames (default: 15).",
    )

    # Video settings
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for video (default: 10).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for frame images (default: 150).",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        default="./overlay_videos",
        help="Output directory for frames and video (default: ./overlay_videos).",
    )
    parser.add_argument(
        "--keep_frames",
        action="store_true",
        help="Keep individual frame images after creating video.",
    )

    # Model settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader workers (default: 4).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on (default: auto).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=4500,
        help="Number of frames per trajectory when stitching (default: 4500).",
    )
    parser.add_argument(
        "--denormalize",
        action="store_true",
        help="Denormalize to original arena coordinates (for sequence overlays).",
    )

    # Window overlay settings
    parser.add_argument(
        "--window_frames",
        type=int,
        nargs="+",
        default=[0, -1],
        help="Frame indices within each window to show (default: [0, -1]).",
    )

    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    LOG.info("Using device: %s", device)
    return device


def load_fly_ids(path: Optional[str], split_key: Optional[str]) -> Optional[set]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        entries = payload
    else:
        if split_key is None:
            raise ValueError("--val_split_name must be provided when --fly_split_file contains multiple splits.")
        if split_key not in payload:
            raise KeyError(f"Split '{split_key}' not found in {path}. Available keys: {list(payload.keys())}")
        entries = payload[split_key]
    fly_ids = {(item["sequence_id"], int(item["fly_idx"])) for item in entries}
    LOG.info("Loaded %d fly IDs from %s", len(fly_ids), path)
    return fly_ids


def build_model(checkpoint_path: Path, device: torch.device) -> Tuple[VQVAE, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "args" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain training arguments under key 'args'.")
    ckpt_args = checkpoint["args"]

    # Detect quantizer type
    quantizer_type = ckpt_args.get("quantizer_type")
    if quantizer_type is None and "config" in checkpoint:
        quantizer_type = checkpoint["config"].get("method", "vq")
    if quantizer_type is None:
        quantizer_type = "vq"

    if quantizer_type in ("rvq", "fsq", "lfq"):
        quantizer_kwargs = ckpt_args.get("quantizer_kwargs")
        if quantizer_kwargs is None and "config" in checkpoint:
            quantizer_kwargs = checkpoint["config"].get("kwargs", {})
        if quantizer_kwargs is None:
            quantizer_kwargs = {}

        model = UnifiedVQVAE(
            input_dim=ckpt_args["input_dim"],
            hidden_dims=ckpt_args["hidden_dims"],
            embedding_dim=ckpt_args["embedding_dim"],
            num_embeddings=ckpt_args["num_embeddings"],
            sequence_length=ckpt_args["window_size"],
            num_residual_blocks=ckpt_args["num_residual_blocks"],
            commitment_cost=ckpt_args["commitment_cost"],
            quantizer_method=quantizer_type,
            quantizer_kwargs=quantizer_kwargs,
        )
        LOG.info("Loading UnifiedVQVAE with quantizer_type=%s", quantizer_type)
    else:
        model = VQVAE(
            input_dim=ckpt_args["input_dim"],
            hidden_dims=ckpt_args["hidden_dims"],
            embedding_dim=ckpt_args["embedding_dim"],
            num_embeddings=ckpt_args["num_embeddings"],
            sequence_length=ckpt_args["window_size"],
            num_residual_blocks=ckpt_args["num_residual_blocks"],
            commitment_cost=ckpt_args["commitment_cost"],
        )
        LOG.info("Loading legacy VQVAE")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    LOG.info("Loaded model from %s", checkpoint_path)
    return model, ckpt_args


def collate_metadata(batch_metadata: Dict[str, list], idx: int) -> Dict[str, object]:
    record: Dict[str, object] = {}
    for key, value_list in batch_metadata.items():
        value = value_list[idx]
        if isinstance(value, torch.Tensor):
            record[key] = value.item()
        else:
            record[key] = value
    return record


def run_inference(
    model: VQVAE,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, object]]]:
    originals: List[torch.Tensor] = []
    reconstructions: List[torch.Tensor] = []
    metadata: List[Dict[str, object]] = []

    with torch.no_grad():
        for windows, meta in loader:
            windows = windows.to(device)
            recon, _, _, _, _ = model(windows)

            originals.extend(windows.cpu())
            reconstructions.extend(recon.cpu())

            batch_size = windows.shape[0]
            for i in range(batch_size):
                metadata.append(collate_metadata(meta, i))

    LOG.info("Collected %d windows for visualization", len(originals))
    return originals, reconstructions, metadata


def create_sequence_frames(
    arenas_orig: Dict[str, np.ndarray],
    arenas_recon: Dict[str, np.ndarray],
    sequence_id: str,
    start_frame: int,
    end_frame: int,
    frame_step: int,
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    """Generate sequence overlay frames at regular intervals."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if sequence_id not in arenas_orig:
        raise ValueError(f"Sequence '{sequence_id}' not found in reconstructions.")

    original_seq = arenas_orig[sequence_id]
    recon_seq = arenas_recon[sequence_id]

    num_frames = original_seq.shape[0]
    end_frame = min(end_frame, num_frames)

    frame_paths = []
    frames_to_generate = list(range(start_frame, end_frame, frame_step))
    total = len(frames_to_generate)

    LOG.info("Generating %d sequence overlay frames...", total)

    for i, frame_idx in enumerate(frames_to_generate):
        if i % 10 == 0:
            LOG.info("  Progress: %d/%d (%.1f%%)", i, total, 100 * i / total)

        frame_path = frames_dir / f"frame_{frame_idx:05d}.png"
        fig, ax = plot_sequence_overlay(
            original_seq,
            recon_seq,
            frame_idx=frame_idx,
        )
        fig.suptitle(f"Sequence {sequence_id} • Frame {frame_idx}", fontsize=12, y=0.98)
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    LOG.info("✓ Created %d frames in %s", len(frame_paths), frames_dir)
    return frame_paths


def create_window_frames(
    originals: List[torch.Tensor],
    reconstructions: List[torch.Tensor],
    metadata: List[Dict[str, object]],
    sequence_id: str,
    fly_idx: int,
    start_frame: int,
    end_frame: int,
    frame_step: int,
    window_size: int,
    stride: int,
    window_frame_indices: List[int],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    """Generate window overlay frames for a specific fly at regular intervals."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Find windows for the target fly
    fly_windows = []
    for orig, recon, meta in zip(originals, reconstructions, metadata):
        if meta["sequence_id"] == sequence_id and int(meta["fly_idx"]) == fly_idx:
            fly_windows.append({
                "window_idx": int(meta["window_idx"]),
                "original": orig,
                "reconstructed": recon,
            })

    if not fly_windows:
        raise ValueError(f"No windows found for sequence '{sequence_id}', fly {fly_idx}")

    # Sort by window index
    fly_windows.sort(key=lambda x: x["window_idx"])

    LOG.info("Found %d windows for fly %d", len(fly_windows), fly_idx)

    # Determine which windows to visualize based on frame range
    frame_paths = []
    frames_to_generate = []

    for window in fly_windows:
        window_start = window["window_idx"] * stride
        window_end = window_start + window_size

        # Check if this window overlaps with our frame range
        for frame_idx in range(start_frame, end_frame, frame_step):
            if window_start <= frame_idx < window_end:
                frames_to_generate.append({
                    "frame_idx": frame_idx,
                    "window": window,
                    "window_frame": frame_idx - window_start,
                })

    # Remove duplicates (same window might be used for multiple frames)
    seen = set()
    unique_frames = []
    for item in frames_to_generate:
        key = (item["frame_idx"], item["window"]["window_idx"])
        if key not in seen:
            seen.add(key)
            unique_frames.append(item)
    frames_to_generate = unique_frames

    total = len(frames_to_generate)
    LOG.info("Generating %d window overlay frames...", total)

    for i, item in enumerate(frames_to_generate):
        if i % 10 == 0:
            LOG.info("  Progress: %d/%d (%.1f%%)", i, total, 100 * i / total)

        frame_idx = item["frame_idx"]
        window = item["window"]

        frame_path = frames_dir / f"frame_{frame_idx:05d}.png"
        fig, axes = plot_window_overlay(
            window["original"],
            window["reconstructed"],
            frame_indices=window_frame_indices,
        )
        fig.suptitle(
            f"Seq {sequence_id} • Fly {fly_idx} • Window {window['window_idx']} • Frame {frame_idx}",
            fontsize=10,
            y=1.02,
        )
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    LOG.info("✓ Created %d frames in %s", len(frame_paths), frames_dir)
    return frame_paths


def create_video(
    frames_dir: Path,
    output_dir: Path,
    sequence_id: str,
    overlay_type: str,
    fps: int,
    fly_idx: Optional[int] = None,
) -> Path:
    """Create video from frames using FFmpeg."""
    import glob

    frame_files = sorted(glob.glob(str(frames_dir / "frame_*.png")))
    if not frame_files:
        raise RuntimeError("No frames found to create video!")

    LOG.info("Found %d frames for video creation", len(frame_files))

    # Create output filename
    if overlay_type == "sequence":
        video_name = f"{sequence_id}_sequence_overlay.mp4"
    else:
        video_name = f"{sequence_id}_fly{fly_idx}_window_overlay.mp4"

    output_path = output_dir / video_name

    # FFmpeg command
    glob_pattern = str(frames_dir / "frame_*.png")
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", glob_pattern,
        "-c:v", "libopenh264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(output_path),
    ]

    LOG.info("Creating video: %s", output_path)
    LOG.info("Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        file_size = output_path.stat().st_size / 1e6
        LOG.info("✓ Video created: %s (%.1f MB)", output_path, file_size)
        return output_path
    except subprocess.CalledProcessError as e:
        LOG.error("✗ Error creating video:")
        LOG.error(e.stderr)
        raise
    except FileNotFoundError:
        LOG.error("✗ ffmpeg not found. Please install ffmpeg:")
        LOG.error("  conda install ffmpeg")
        LOG.error("  or: sudo apt-get install ffmpeg")
        raise


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()

    # Validate arguments
    if args.overlay_type == "window" and args.fly_idx is None:
        raise ValueError("--fly_idx is required for window overlays")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model, ckpt_args = build_model(checkpoint_path, device)
    window_size = ckpt_args["window_size"]
    stride = ckpt_args["stride"]

    # Load validation data
    val_ids = None
    if args.fly_split_file:
        val_ids = load_fly_ids(args.fly_split_file, args.val_split_name)

    LOG.info("Loading validation trajectories from %s", args.data_file)
    val_trajectories = load_and_preprocess_for_vqvae(args.data_file, allowed_fly_ids=val_ids)
    if not val_trajectories:
        raise RuntimeError("Validation trajectory list is empty.")

    val_dataset = FlyKeypointDataset(
        val_trajectories,
        window_size=window_size,
        stride=stride,
        include_metadata=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Run inference
    LOG.info("Running inference...")
    originals, reconstructions, metadata = run_inference(model, val_loader, device)

    # Generate frames based on overlay type
    if args.overlay_type == "sequence":
        # Stitch windows into full sequences
        LOG.info("Stitching windows into sequences...")
        grouped_orig = group_windows_by_fly(originals, metadata)
        grouped_recon = group_windows_by_fly(reconstructions, metadata)

        stitched_orig = stitch_fly_windows(grouped_orig, args.num_frames, window_size, stride)
        stitched_recon = stitch_fly_windows(grouped_recon, args.num_frames, window_size, stride)

        # Denormalize if requested
        if args.denormalize:
            LOG.info("Denormalizing to original arena coordinates...")
            norm_params = load_normalization_params_from_original(args.data_file)
            stitched_orig = denormalize_reconstructions(stitched_orig, norm_params)
            stitched_recon = denormalize_reconstructions(stitched_recon, norm_params)

        arenas_orig = assemble_sequences(stitched_orig)
        arenas_recon = assemble_sequences(stitched_recon)

        frame_paths = create_sequence_frames(
            arenas_orig,
            arenas_recon,
            args.sequence_id,
            args.start_frame,
            args.end_frame,
            args.frame_step,
            output_dir,
            args.dpi,
        )
    else:  # window overlay
        frame_paths = create_window_frames(
            originals,
            reconstructions,
            metadata,
            args.sequence_id,
            args.fly_idx,
            args.start_frame,
            args.end_frame,
            args.frame_step,
            window_size,
            stride,
            args.window_frames,
            output_dir,
            args.dpi,
        )

    # Create video
    video_path = create_video(
        output_dir / "frames",
        output_dir,
        args.sequence_id,
        args.overlay_type,
        args.fps,
        args.fly_idx,
    )

    # Clean up frames if requested
    if not args.keep_frames:
        import shutil
        frames_dir = output_dir / "frames"
        LOG.info("Removing frame directory: %s", frames_dir)
        shutil.rmtree(frames_dir)

    LOG.info("✓ Complete! Video saved to: %s", video_path)


if __name__ == "__main__":
    main()
