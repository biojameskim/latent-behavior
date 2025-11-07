"""
Run a trained VQ-VAE on a held-out split, stitch reconstructions back into
full fly trajectories, and generate qualitative plots.

Example:
    python visualize_reconstructions.py \
        --data_file ../../../data/fly_data/fly_group_train.npy \
        --checkpoint ../training/outputs/run_11_01_25_v5/best_model.pt \
        --fly_split_file ../data/fly_data/fly_split.json \
        --val_split_name val \
        --output_dir ./visualization/recon_viz \
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    # Allow running the script directly without needing to modify PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
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
from flies.vq_vae.vqvae import VQVAE


LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE reconstructions.")
    parser.add_argument("--data_file", required=True, help="Path to .npy file containing fly trajectories.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained VQ-VAE checkpoint.")
    parser.add_argument(
        "--fly_split_file",
        default=None,
        help="JSON file containing fly-level splits produced by generate_fly_splits.",
    )
    parser.add_argument(
        "--val_split_name",
        default="val",
        help="Split name to pull from --fly_split_file (default: val).",
    )
    parser.add_argument(
        "--val_fly_filter",
        default=None,
        help="Optional JSON array of {sequence_id, fly_idx} entries overriding the validation set.",
    )
    parser.add_argument(
        "--output_dir",
        default="./reconstruction_viz",
        help="Directory where plots and optional arrays will be saved.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers for inference.")
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
        help="Number of frames per full fly trajectory when stitching windows.",
    )
    parser.add_argument(
        "--window_overlays",
        type=int,
        default=5,
        help="Number of individual windows to plot.",
    )
    parser.add_argument(
        "--window_frames",
        type=int,
        nargs="+",
        default=[0, -1],
        help="Frame indices within each window to overlay (allow negatives).",
    )
    parser.add_argument(
        "--sequence_frames",
        type=int,
        nargs="+",
        default=[0],
        help="Arena frames to visualize for each held-out sequence.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=5,
        help="Maximum number of sequences to render arena overlays for (default: 5, use -1 for all).",
    )
    parser.add_argument(
        "--save_pt",
        default=None,
        help="Optional path to write stitched trajectories/metadata using torch.save.",
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
    model = VQVAE(
        input_dim=ckpt_args["input_dim"],
        hidden_dims=ckpt_args["hidden_dims"],
        embedding_dim=ckpt_args["embedding_dim"],
        num_embeddings=ckpt_args["num_embeddings"],
        sequence_length=ckpt_args["window_size"],
        num_residual_blocks=ckpt_args["num_residual_blocks"],
        commitment_cost=ckpt_args["commitment_cost"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    LOG.info(
        "Loaded model from %s (window_size=%d, stride=%d)",
        checkpoint_path,
        ckpt_args["window_size"],
        ckpt_args["stride"],
    )
    return model, ckpt_args


def collate_metadata(batch_metadata: Dict[str, Iterable], idx: int) -> Dict[str, object]:
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


def save_window_plots(
    originals: Sequence[torch.Tensor],
    reconstructions: Sequence[torch.Tensor],
    metadata: Sequence[Dict[str, object]],
    output_dir: Path,
    max_windows: int,
    frame_indices: Sequence[int],
) -> None:
    window_dir = output_dir / "window_overlays"
    window_dir.mkdir(parents=True, exist_ok=True)
    num_windows = min(max_windows, len(originals))
    LOG.info("Saving %d window overlays to %s", num_windows, window_dir)

    for idx in range(num_windows):
        save_path = window_dir / f"window_{idx:04d}.png"
        fig, _ = plot_window_overlay(
            originals[idx],
            reconstructions[idx],
            frame_indices=frame_indices,
        )
        meta = metadata[idx]
        seq = meta.get("sequence_id", "unknown")
        fly = meta.get("fly_idx", "NA")
        window = meta.get("window_idx", "NA")
        fig.suptitle(f"Seq {seq} • Fly {fly} • Window {window}", fontsize=10)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_sequence_plots(
    arenas_original: Dict[str, torch.Tensor],
    arenas_recon: Dict[str, torch.Tensor],
    output_dir: Path,
    frame_indices: Sequence[int],
    max_sequences: int,
) -> None:
    seq_dir = output_dir / "sequence_overlays"
    seq_dir.mkdir(parents=True, exist_ok=True)

    sequence_ids = sorted(arenas_original.keys())
    if max_sequences > 0:
        sequence_ids = sequence_ids[:max_sequences]

    LOG.info("Saving arena overlays for %d sequences to %s", len(sequence_ids), seq_dir)

    for sequence_id in sequence_ids:
        for frame in frame_indices:
            save_path = seq_dir / f"{sequence_id}_frame_{frame}.png"
            fig, _ = plot_sequence_overlay(
                arenas_original[sequence_id],
                arenas_recon[sequence_id],
                frame_idx=frame,
                save_path=save_path,
            )
            plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model, ckpt_args = build_model(checkpoint_path, device)
    window_size = ckpt_args["window_size"]
    stride = ckpt_args["stride"]

    val_ids = None
    if args.fly_split_file:
        val_ids = load_fly_ids(args.fly_split_file, args.val_split_name)
    if args.val_fly_filter:
        LOG.info("Overriding validation split with %s", args.val_fly_filter)
        val_ids = load_fly_ids(args.val_fly_filter, None)

    LOG.info("Loading validation trajectories from %s", args.data_file)
    val_trajectories = load_and_preprocess_for_vqvae(args.data_file, allowed_fly_ids=val_ids)
    if not val_trajectories:
        raise RuntimeError("Validation trajectory list is empty. Check your filters/split configuration.")

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

    originals, reconstructions, metadata = run_inference(model, val_loader, device)

    grouped_orig = group_windows_by_fly(originals, metadata)
    grouped_recon = group_windows_by_fly(reconstructions, metadata)

    stitched_orig = stitch_fly_windows(grouped_orig, args.num_frames, window_size, stride)
    stitched_recon = stitch_fly_windows(grouped_recon, args.num_frames, window_size, stride)

    arenas_orig = assemble_sequences(stitched_orig)
    arenas_recon = assemble_sequences(stitched_recon)

    save_window_plots(
        originals,
        reconstructions,
        metadata,
        output_dir,
        max_windows=args.window_overlays,
        frame_indices=args.window_frames,
    )
    save_sequence_plots(
        arenas_orig,
        arenas_recon,
        output_dir,
        frame_indices=args.sequence_frames,
        max_sequences=args.max_sequences,
    )

    if args.save_pt:
        pt_path = Path(args.save_pt)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stitched_original": stitched_orig,
                "stitched_reconstruction": stitched_recon,
                "metadata": metadata,
            },
            pt_path,
        )
        LOG.info("Saved stitched trajectories to %s", pt_path)

    LOG.info("Visualization complete. Results written to %s", output_dir)


if __name__ == "__main__":
    main()
