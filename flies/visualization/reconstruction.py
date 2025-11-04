"""
Utilities for stitching VQ-VAE reconstructions back into fly trajectories and
visualizing them alongside the original poses.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .plot_mabe_flies import get_Dark3_cmap, plot_arena, plot_fly


FlyKey = Tuple[str, int]


def _to_numpy(window: torch.Tensor) -> np.ndarray:
    if isinstance(window, torch.Tensor):
        window = window.detach().cpu().numpy()
    return np.asarray(window)


def window_to_pose(window: torch.Tensor) -> np.ndarray:
    """
    Convert a dataset window (48, T) or (T, 48) into (T, 24, 2).
    """
    arr = _to_numpy(window)
    if arr.ndim == 3:
        if arr.shape[0] == 48:
            arr = arr
        elif arr.shape[1] == 48:
            arr = arr.transpose(1, 0, 2)
        else:
            raise ValueError(f"Expected 48 feature dimension, got shape {arr.shape}")
    elif arr.ndim == 2:
        if arr.shape[0] == 48:
            arr = arr
        elif arr.shape[1] == 48:
            arr = arr.T
        else:
            raise ValueError(f"Expected 48 feature dimension, got shape {arr.shape}")
    else:
        raise ValueError(f"Unsupported window shape {arr.shape}")

    timesteps = arr.shape[-1]
    pose = arr.reshape(24, 2, timesteps).transpose(2, 0, 1)
    return pose


def group_windows_by_fly(
    windows: Sequence[np.ndarray],
    metadatas: Sequence[Dict[str, int]],
) -> Dict[FlyKey, List[Tuple[int, np.ndarray]]]:
    grouped: Dict[FlyKey, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for window, meta in zip(windows, metadatas):
        key = (meta["sequence_id"], int(meta["fly_idx"]))
        grouped[key].append((int(meta["window_idx"]), window_to_pose(window)))
    return grouped


def stitch_fly_windows(
    grouped: Dict[FlyKey, List[Tuple[int, np.ndarray]]],
    num_frames: int,
    window_size: int,
    stride: int,
) -> Dict[FlyKey, np.ndarray]:
    stitched: Dict[FlyKey, np.ndarray] = {}

    for key, items in grouped.items():
        accum = np.zeros((num_frames, 24, 2), dtype=np.float32)
        counts = np.zeros((num_frames, 24, 2), dtype=np.float32)

        for window_idx, pose in items:
            start = window_idx * stride
            end = min(start + pose.shape[0], num_frames)
            segment = pose[: end - start]
            mask = ~np.isnan(segment)
            accum[start:end][mask] += segment[mask]
            counts[start:end][mask] += 1

        filled = np.full_like(accum, np.nan, dtype=np.float32)
        valid = counts > 0
        filled[valid] = accum[valid] / counts[valid]
        stitched[key] = filled

    return stitched


def assemble_sequences(
    stitched: Dict[FlyKey, np.ndarray],
) -> Dict[str, np.ndarray]:
    sequences: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
    for (sequence_id, fly_idx), pose in stitched.items():
        sequences[sequence_id][fly_idx] = pose

    assembled: Dict[str, np.ndarray] = {}
    for sequence_id, fly_map in sequences.items():
        max_fly = max(fly_map) + 1
        num_frames = next(iter(fly_map.values())).shape[0]
        arena = np.full((num_frames, max_fly, 24, 2), np.nan, dtype=np.float32)
        for fly_idx, pose in fly_map.items():
            arena[:, fly_idx, :, :] = pose
        assembled[sequence_id] = arena

    return assembled


def plot_window_overlay(
    original_window: np.ndarray,
    reconstructed_window: np.ndarray,
    frame_indices: Optional[Sequence[int]] = None,
    show_arena: bool = False,
    save_path: Optional[Path] = None,
):
    original = window_to_pose(original_window)
    reconstructed = window_to_pose(reconstructed_window)

    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed windows must have the same shape")

    num_frames = original.shape[0]
    if frame_indices is None:
        frame_indices = (0, num_frames // 2, num_frames - 1)

    frame_indices = [idx % num_frames for idx in frame_indices]
    n_cols = len(frame_indices)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, frame in zip(axes, frame_indices):
        if show_arena:
            plot_arena(ax=ax)
        plot_fly(
            original[frame],
            ax=ax,
            skelcolor="tab:blue",
            kptcolors="tab:blue",
            kpt_alpha=0.9,
            skel_alpha=0.9,
            kpt_marker="o",
        )
        plot_fly(
            reconstructed[frame],
            ax=ax,
            skelcolor="tab:orange",
            kptcolors="tab:orange",
            kpt_alpha=0.7,
            skel_alpha=0.7,
            kpt_marker="x",
        )
        ax.set_title(f"Frame {frame}")
        ax.set_aspect("equal")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, axes


def plot_sequence_overlay(
    original_sequence: np.ndarray,
    reconstructed_sequence: np.ndarray,
    frame_idx: int,
    show_arena: bool = True,
    save_path: Optional[Path] = None,
):
    if original_sequence.shape != reconstructed_sequence.shape:
        raise ValueError("Original and reconstructed sequences must have the same shape")

    num_frames, num_flies, _, _ = original_sequence.shape
    frame_idx = frame_idx % num_frames

    fig, ax = plt.subplots(figsize=(8, 8))
    if show_arena:
        plot_arena(ax=ax)

    cmap = get_Dark3_cmap()

    for fly_idx in range(num_flies):
        original_pose = original_sequence[frame_idx, fly_idx]
        if np.all(np.isnan(original_pose)):
            continue
        color = cmap(fly_idx % cmap.N)
        plot_fly(
            original_pose,
            ax=ax,
            skelcolor=color,
            kptcolors=color,
            kpt_alpha=0.9,
            skel_alpha=0.9,
            kpt_ms=5,
        )
        recon_pose = reconstructed_sequence[frame_idx, fly_idx]
        if np.all(np.isnan(recon_pose)):
            continue
        plot_fly(
            recon_pose,
            ax=ax,
            skelcolor=color,
            kptcolors=color,
            kpt_alpha=0.45,
            skel_alpha=0.45,
            kpt_marker="x",
            kpt_ms=5,
        )

    ax.set_title(f"Sequence frame {frame_idx}")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax
