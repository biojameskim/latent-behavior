"""
Visualize RVQ (Residual Vector Quantization) codebook embeddings.

RVQ is different from standard VQ:
- Standard VQ: One code per timestep → indices shape (B, T)
- RVQ: Multiple codes per timestep (hierarchical) → indices shape (B, T, num_quantizers)

This script provides RVQ-specific visualizations:
1. Individual quantizer contributions (what does each quantizer encode?)
2. Full code combinations (what do all quantizers together produce?)
3. Quantizer ablations (what happens when we zero out certain quantizers?)

Example:
    python visualize_rvq_codebook.py \
        --checkpoint ../training/outputs/rvq_run/best_model.pt \
        --output_dir rvq_viz \
        --mode all
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch

from flies.visualization.plot_mabe_flies import plot_arena, plot_fly
from flies.visualization.reconstruction import window_to_pose
from flies.vq_vae.vqvae_unified import UnifiedVQVAE


LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RVQ codebook.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained UnifiedVQVAE checkpoint")
    parser.add_argument("--output_dir", default="./rvq_viz", help="Output directory")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["individual", "combinations", "ablations", "all"],
        help="Visualization mode",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--frame_indices", type=int, nargs="+", default=[0, -1])
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per visualization type")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show_arena", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model(checkpoint_path: Path, device: torch.device):
    """Load UnifiedVQVAE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "args" not in checkpoint:
        raise KeyError("Checkpoint missing 'args'.")

    args = checkpoint["args"]

    # Check that this is an RVQ model
    if args.get("quantizer_method") != "rvq":
        raise ValueError(
            f"This script is for RVQ models. "
            f"Found quantizer_method={args.get('quantizer_method')}. "
            f"Use visualize_codebook_embeddings.py for standard VQ."
        )

    # Reconstruct model
    model = UnifiedVQVAE(
        input_dim=args["input_dim"],
        hidden_dims=args["hidden_dims"],
        embedding_dim=args["embedding_dim"],
        num_embeddings=args["num_embeddings"],
        sequence_length=args["window_size"],
        num_residual_blocks=args["num_residual_blocks"],
        commitment_cost=args["commitment_cost"],
        quantizer_method=args["quantizer_method"],
        quantizer_kwargs=args.get("quantizer_kwargs", {}),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    num_quantizers = args["quantizer_kwargs"].get("num_quantizers", 4)

    LOG.info(f"Loaded RVQ model:")
    LOG.info(f"  - Codebook size: {args['num_embeddings']}")
    LOG.info(f"  - Num quantizers: {num_quantizers}")
    LOG.info(f"  - Embedding dim: {args['embedding_dim']}")

    return model, args, num_quantizers


def get_latent_length(model, input_dim: int, sequence_length: int, device: torch.device) -> int:
    """Compute latent temporal length."""
    dummy = torch.zeros(1, input_dim, sequence_length, device=device)
    with torch.no_grad():
        latent = model.encoder(dummy)
    return latent.shape[-1]


def visualize_individual_quantizers(
    model: UnifiedVQVAE,
    num_embeddings: int,
    num_quantizers: int,
    latent_len: int,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
    frame_indices: List[int],
    show_arena: bool,
    dpi: int,
):
    """
    Visualize individual quantizer contributions.

    For each quantizer q, we decode codes where:
    - Quantizer q has a specific code
    - All other quantizers are set to 0 (or another baseline)

    This shows what each quantizer learns independently.

    RVQ structure:
        output = quantizer_0(x) + quantizer_1(residual_1) + ... + quantizer_n(residual_n)

    By isolating each quantizer, we see its semantic role.
    """
    output_dir = output_dir / "individual_quantizers"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(f"Visualizing individual quantizer contributions → {output_dir}")

    # Sample codes to visualize
    sample_codes = torch.linspace(0, num_embeddings - 1, num_samples, dtype=torch.long)

    for q_idx in range(num_quantizers):
        LOG.info(f"Processing quantizer {q_idx}/{num_quantizers}")

        for code_idx in sample_codes:
            code_idx = code_idx.item()

            # Create indices: (1, latent_len, num_quantizers)
            # All quantizers = 0, except quantizer q_idx = code_idx
            indices = torch.zeros(1, latent_len, num_quantizers, dtype=torch.long, device=device)
            indices[:, :, q_idx] = code_idx

            # Decode
            with torch.no_grad():
                reconstructed = model.decode_codes(indices)

            # Convert to pose
            pose = window_to_pose(reconstructed[0].cpu())

            # Visualize
            fig, axes = plt.subplots(1, len(frame_indices), figsize=(4 * len(frame_indices), 4))
            if len(frame_indices) == 1:
                axes = [axes]

            for ax, frame in zip(axes, frame_indices):
                frame = frame % pose.shape[0]
                if show_arena:
                    plot_arena(ax=ax)
                plot_fly(
                    pose[frame],
                    ax=ax,
                    skelcolor="tab:blue",
                    kptcolors="tab:blue",
                    kpt_alpha=0.9,
                    skel_alpha=0.9,
                )
                ax.set_title(f"Frame {frame}")
                ax.set_aspect("equal")

            fig.suptitle(f"Quantizer {q_idx}, Code {code_idx:.0f}")
            fig.tight_layout()

            save_path = output_dir / f"quantizer_{q_idx}_code_{code_idx:04.0f}.png"
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def visualize_combinations(
    model: UnifiedVQVAE,
    num_embeddings: int,
    num_quantizers: int,
    latent_len: int,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
    frame_indices: List[int],
    show_arena: bool,
    dpi: int,
):
    """
    Visualize full code combinations.

    Each timestep in RVQ has num_quantizers codes that sum together.
    This shows what different combinations produce.

    We sample random combinations to explore the learned space.
    """
    output_dir = output_dir / "combinations"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(f"Visualizing code combinations → {output_dir}")

    for sample_idx in range(num_samples):
        # Sample random codes for each quantizer
        # Shape: (1, latent_len, num_quantizers)
        indices = torch.randint(0, num_embeddings, (1, latent_len, num_quantizers), device=device)

        # Decode
        with torch.no_grad():
            reconstructed = model.decode_codes(indices)

        # Convert to pose
        pose = window_to_pose(reconstructed[0].cpu())

        # Visualize
        fig, axes = plt.subplots(1, len(frame_indices), figsize=(4 * len(frame_indices), 4))
        if len(frame_indices) == 1:
            axes = [axes]

        for ax, frame in zip(axes, frame_indices):
            frame = frame % pose.shape[0]
            if show_arena:
                plot_arena(ax=ax)
            plot_fly(
                pose[frame],
                ax=ax,
                skelcolor="tab:blue",
                kptcolors="tab:blue",
                kpt_alpha=0.9,
                skel_alpha=0.9,
            )
            ax.set_title(f"Frame {frame}")
            ax.set_aspect("equal")

        # Create title showing the code combination
        codes_str = str(indices[0, 0].cpu().tolist())
        fig.suptitle(f"Sample {sample_idx}: {codes_str}")
        fig.tight_layout()

        save_path = output_dir / f"combination_{sample_idx:03d}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def visualize_ablations(
    model: UnifiedVQVAE,
    num_embeddings: int,
    num_quantizers: int,
    latent_len: int,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
    frame_indices: List[int],
    show_arena: bool,
    dpi: int,
):
    """
    Visualize quantizer ablations.

    Start with random full codes, then progressively zero out quantizers
    to see their contribution.

    Shows: Full → Without Q3 → Without Q2&Q3 → Without Q1&Q2&Q3 → Only Q0
    """
    output_dir = output_dir / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(f"Visualizing quantizer ablations → {output_dir}")

    for sample_idx in range(num_samples):
        # Sample random full codes
        full_indices = torch.randint(0, num_embeddings, (1, latent_len, num_quantizers), device=device)

        # Create ablated versions
        poses = {}
        labels = {}

        # Full (all quantizers)
        with torch.no_grad():
            recon = model.decode_codes(full_indices)
        poses["full"] = window_to_pose(recon[0].cpu())
        labels["full"] = "All quantizers"

        # Progressive ablation (zero out from highest to lowest)
        for keep_n in range(num_quantizers - 1, 0, -1):
            ablated_indices = full_indices.clone()
            ablated_indices[:, :, keep_n:] = 0  # Zero out higher quantizers

            with torch.no_grad():
                recon = model.decode_codes(ablated_indices)
            poses[f"keep_{keep_n}"] = window_to_pose(recon[0].cpu())
            labels[f"keep_{keep_n}"] = f"First {keep_n} quantizer{'s' if keep_n > 1 else ''}"

        # Only first quantizer
        first_only = full_indices.clone()
        first_only[:, :, 1:] = 0
        with torch.no_grad():
            recon = model.decode_codes(first_only)
        poses["first_only"] = window_to_pose(recon[0].cpu())
        labels["first_only"] = "Only first quantizer"

        # Plot all ablations
        n_ablations = len(poses)
        fig, axes = plt.subplots(n_ablations, len(frame_indices), figsize=(4 * len(frame_indices), 3 * n_ablations))

        if n_ablations == 1:
            axes = [axes]
        if len(frame_indices) == 1:
            axes = [[ax] for ax in axes]

        for row_idx, (key, pose) in enumerate(poses.items()):
            for col_idx, frame in enumerate(frame_indices):
                ax = axes[row_idx][col_idx]
                frame = frame % pose.shape[0]

                if show_arena:
                    plot_arena(ax=ax)

                plot_fly(
                    pose[frame],
                    ax=ax,
                    skelcolor="tab:blue",
                    kptcolors="tab:blue",
                    kpt_alpha=0.9,
                    skel_alpha=0.9,
                )

                if col_idx == 0:
                    ax.set_ylabel(labels[key], fontsize=10)
                if row_idx == 0:
                    ax.set_title(f"Frame {frame}")

                ax.set_aspect("equal")

        fig.suptitle(f"Ablation Study {sample_idx}")
        fig.tight_layout()

        save_path = output_dir / f"ablation_{sample_idx:03d}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    # Load model
    model, train_args, num_quantizers = load_model(checkpoint_path, device)
    latent_len = get_latent_length(model, train_args["input_dim"], train_args["window_size"], device)
    num_embeddings = train_args["num_embeddings"]

    LOG.info(f"Latent length: {latent_len}")

    # Run visualizations based on mode
    if args.mode in ["individual", "all"]:
        visualize_individual_quantizers(
            model,
            num_embeddings,
            num_quantizers,
            latent_len,
            device,
            output_dir,
            args.num_samples,
            args.frame_indices,
            args.show_arena,
            args.dpi,
        )

    if args.mode in ["combinations", "all"]:
        visualize_combinations(
            model,
            num_embeddings,
            num_quantizers,
            latent_len,
            device,
            output_dir,
            args.num_samples,
            args.frame_indices,
            args.show_arena,
            args.dpi,
        )

    if args.mode in ["ablations", "all"]:
        visualize_ablations(
            model,
            num_embeddings,
            num_quantizers,
            latent_len,
            device,
            output_dir,
            args.num_samples,
            args.frame_indices,
            args.show_arena,
            args.dpi,
        )

    LOG.info(f"RVQ visualization complete. Results in {output_dir}")


if __name__ == "__main__":
    main()
