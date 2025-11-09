"""
Decode every codebook embedding (and optional user-specified sequences) through
the VQ-VAE decoder to visualize the learned behavior "syllables".

Example:
    python visualize_codebook_embeddings.py \
        --checkpoint ../training/outputs/run_11_01_25_v5/best_model.pt \
        --output_dir codebook_viz \
        --frame_indices 0 74 149
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import torch

from flies.visualization.plot_mabe_flies import plot_arena, plot_fly
from flies.visualization.reconstruction import window_to_pose
from flies.vq_vae.vqvae import VQVAE


LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE codebook embeddings.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained VQ-VAE checkpoint (expects 'args' and 'model_state_dict').",
    )
    parser.add_argument(
        "--output_dir",
        default="./codebook_viz",
        help="Directory where embedding figures will be written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for decoding (default: auto-detect CUDA).",
    )
    parser.add_argument(
        "--frame_indices",
        type=int,
        nargs="+",
        default=[0, -1],
        help="Frame indices (within the decoded window) to plot for each embedding.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=128,
        help="Number of embeddings to decode at once (controls memory usage).",
    )
    parser.add_argument(
        "--codes",
        default=None,
        help="Optional JSON list of custom code sequences to visualize "
             "(e.g. '[[12,42,17]]'). Each sequence is tiled/truncated to the latent length.",
    )
    parser.add_argument(
        "--show_arena",
        action="store_true",
        help="Overlay the arena outline in the generated plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI for saved images.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    LOG.info("Using device: %s", device)
    return device


def build_model(checkpoint_path: Path, device: torch.device) -> tuple[VQVAE, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "args" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} missing 'args' payload.")

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
        "Loaded VQ-VAE (window_size=%d, num_embeddings=%d)",
        ckpt_args["window_size"],
        ckpt_args["num_embeddings"],
    )
    return model, ckpt_args


def latent_length(model: VQVAE, input_dim: int, sequence_length: int, device: torch.device) -> int:
    dummy = torch.zeros(1, input_dim, sequence_length, device=device)
    with torch.no_grad():
        latent = model.encoder(dummy)
    length = latent.shape[-1]
    LOG.info("Latent temporal length: %d", length)
    return length


def expand_sequence(seq: Sequence[int], target_len: int) -> torch.Tensor:
    """
    Expand or tile a custom code sequence to match the latent length.

    Examples:
        seq=[12, 42, 17], target_len=5
        → [12, 42, 17, 12, 42]  (tile and truncate)

        seq=[12, 42, 17, 5, 28], target_len=5
        → [12, 42, 17, 5, 28]  (exact match, no change)

        seq=[42], target_len=5
        → [42, 42, 42, 42, 42]  (repeat single code)

    This allows you to specify custom sequences and have them automatically
    fit the model's expected latent temporal length.
    """
    if len(seq) == 0:
        raise ValueError("Custom code sequence must contain at least one index.")
    tensor = torch.tensor(seq, dtype=torch.long)
    if tensor.numel() == target_len:
        return tensor
    # Tile the sequence enough times to cover target_len, then truncate
    repeats = (target_len + tensor.numel() - 1) // tensor.numel()
    tensor = tensor.repeat(repeats)[:target_len]
    return tensor


def decode_embeddings(
    model: VQVAE,
    indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Decode discrete code indices to continuous behavior trajectories.

    This is the CORE FUNCTION for codebook visualization:
    1. Takes discrete code indices (e.g., [42, 42, 42, 42, 42])
    2. Looks up embeddings from the learned codebook
    3. Passes through the decoder to reconstruct behavior
    4. Returns continuous keypoint trajectories

    Args:
        model: Trained VQ-VAE with decoder and codebook
        indices: (batch, latent_length) - discrete code indices
                 Example: [[42, 42, 42, 42, 42]] to visualize code 42
        device: Where to run computation

    Returns:
        decoded: (batch, 48, 150) - reconstructed fly keypoint trajectories
                 48 = 24 keypoints × 2 coordinates (x,y)
                 150 = sequence length in frames
    """
    with torch.no_grad():
        # model.decode_codes internally:
        # 1. Converts indices to one-hot: [42] → [0,0,...,1,...,0] at position 42
        # 2. Multiplies by codebook weights to get embeddings
        # 3. Passes embeddings through decoder CNN
        # 4. Returns reconstructed behavior
        decoded = model.decode_codes(indices.to(device))
    return decoded.cpu()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_frame_indices(frame_indices: Iterable[int], num_frames: int) -> List[int]:
    normalized = []
    for idx in frame_indices:
        normalized.append(idx % num_frames)
    return normalized


def plot_embedding_window(
    pose_sequence: torch.Tensor,
    indices: Sequence[int],
    embedding_label: str,
    output_path: Path,
    show_arena: bool,
    dpi: int,
) -> None:
    pose_np = window_to_pose(pose_sequence)
    num_frames = pose_np.shape[0]
    frames = normalize_frame_indices(indices, num_frames)

    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
    if len(frames) == 1:
        axes = [axes]

    for ax, frame in zip(axes, frames):
        if show_arena:
            plot_arena(ax=ax)
        plot_fly(
            pose_np[frame],
            ax=ax,
            skelcolor="tab:blue",
            kptcolors="tab:blue",
            kpt_alpha=0.9,
            skel_alpha=0.9,
            kpt_marker="o",
        )
        ax.set_title(f"Frame {frame}")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(embedding_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def visualize_codebook(
    model: VQVAE,
    num_embeddings: int,
    latent_len: int,
    frame_indices: Sequence[int],
    chunk_size: int,
    device: torch.device,
    output_dir: Path,
    show_arena: bool,
    dpi: int,
) -> None:
    """
    Visualize ALL codebook embeddings by decoding each one.

    CONCEPT: For each code in the codebook (0, 1, 2, ..., num_embeddings-1):
    1. Create a sequence where that code is repeated: [42, 42, 42, 42, 42]
    2. Decode it to see what behavior it represents
    3. Visualize selected frames from the decoded behavior

    This answers: "What does code #42 actually mean in terms of fly behavior?"

    Example:
        - Code 12 might represent "walking forward"
        - Code 42 might represent "turning left"
        - Code 17 might represent "grooming"

    Args:
        model: Trained VQ-VAE
        num_embeddings: Size of codebook (e.g., 64, 128, 512)
        latent_len: Temporal length of latent codes (e.g., 5)
                   Computed as: sequence_length / product(strides)
        frame_indices: Which frames to show in each visualization
        chunk_size: Decode this many embeddings at once (memory control)
        device: CPU or CUDA
        output_dir: Where to save images
        show_arena: Whether to overlay arena outline
        dpi: Image resolution
    """
    ensure_directory(output_dir)
    LOG.info("Rendering %d embeddings → %s", num_embeddings, output_dir)

    # all_indices = [0, 1, 2, 3, ..., num_embeddings-1]
    all_indices = torch.arange(num_embeddings, dtype=torch.long)

    # Process in chunks to avoid memory issues
    for start in range(0, num_embeddings, chunk_size):
        end = min(start + chunk_size, num_embeddings)
        batch_indices = all_indices[start:end]  # e.g., [0, 1, 2, ..., 63]

        # CRITICAL: Repeat each code across latent_len timesteps
        # Example: code 42 → [42, 42, 42, 42, 42]
        # Shape: (batch_size, latent_len)
        latent_codes = batch_indices[:, None].repeat(1, latent_len)
        # batch_indices[:, None]: (batch,) → (batch, 1)
        # .repeat(1, latent_len): (batch, 1) → (batch, latent_len)

        # Decode all codes in this chunk
        # Input: (batch, latent_len) discrete indices
        # Output: (batch, 48, 150) continuous keypoint trajectories
        decoded = decode_embeddings(model, latent_codes, device=device)

        # Save one image per embedding
        for idx, window in zip(batch_indices.tolist(), decoded):
            save_path = output_dir / f"embedding_{idx:04d}.png"
            plot_embedding_window(
                window,
                frame_indices,
                embedding_label=f"Code {idx}",
                output_path=save_path,
                show_arena=show_arena,
                dpi=dpi,
            )


def visualize_custom_sequences(
    model: VQVAE,
    sequences: List[Sequence[int]],
    latent_len: int,
    frame_indices: Sequence[int],
    device: torch.device,
    output_dir: Path,
    show_arena: bool,
    dpi: int,
) -> None:
    ensure_directory(output_dir)
    LOG.info("Rendering %d custom code sequences → %s", len(sequences), output_dir)

    tensors = [expand_sequence(seq, latent_len) for seq in sequences]
    codes = torch.stack(tensors, dim=0)
    decoded = decode_embeddings(model, codes, device=device)

    for idx, (seq, window) in enumerate(zip(sequences, decoded), start=1):
        label = f"Custom {idx}: {seq}"
        save_path = output_dir / f"custom_{idx:02d}.png"
        plot_embedding_window(
            window,
            frame_indices,
            embedding_label=label,
            output_path=save_path,
            show_arena=show_arena,
            dpi=dpi,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "codebook_embeddings"
    custom_dir = output_dir / "custom_sequences"

    device = resolve_device(args.device)
    model, ckpt_args = build_model(checkpoint_path, device)
    latent_len = latent_length(model, ckpt_args["input_dim"], ckpt_args["window_size"], device)

    visualize_codebook(
        model=model,
        num_embeddings=ckpt_args["num_embeddings"],
        latent_len=latent_len,
        frame_indices=args.frame_indices,
        chunk_size=args.chunk_size,
        device=device,
        output_dir=embeddings_dir,
        show_arena=args.show_arena,
        dpi=args.dpi,
    )

    if args.codes:
        sequences = json.loads(args.codes)
        if not isinstance(sequences, list):
            raise ValueError("--codes must be a JSON list of sequences.")
        visualize_custom_sequences(
            model=model,
            sequences=sequences,
            latent_len=latent_len,
            frame_indices=args.frame_indices,
            device=device,
            output_dir=custom_dir,
            show_arena=args.show_arena,
            dpi=args.dpi,
        )

    LOG.info("Codebook visualization complete. Results written to %s", output_dir)


if __name__ == "__main__":
    main()
