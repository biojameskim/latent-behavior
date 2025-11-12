"""
TUTORIAL: VQ-VAE Codebook Visualization

This script demonstrates the CORE CONCEPTS of codebook visualization with
clear step-by-step examples and explanations.

After reading this, you'll understand:
1. How to load a trained VQ-VAE model
2. How to decode individual codebook embeddings
3. How to decode custom code sequences
4. How to visualize the results

For production use, use the full-featured scripts:
- visualize_codebook_embeddings.py (all embeddings + custom sequences)
- visualize_reconstructions.py (reconstruction quality on real data)
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import VQ-VAE model
from flies.vq_vae.vqvae import VQVAE
from flies.visualization.reconstruction import window_to_pose
from flies.visualization.plot_mabe_flies import plot_fly


# ============================================================================
# STEP 1: Load a trained VQ-VAE model
# ============================================================================

def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load a trained VQ-VAE from checkpoint.

    The checkpoint contains:
    - model_state_dict: Learned weights
    - args: Training configuration (needed to reconstruct architecture)

    Returns:
        model: Loaded VQ-VAE in eval mode
        args: Training arguments (dict)
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "args" not in checkpoint:
        raise KeyError("Checkpoint missing 'args'. Cannot reconstruct model architecture.")

    args = checkpoint["args"]

    # Reconstruct model architecture from saved args
    model = VQVAE(
        input_dim=args["input_dim"],              # 48 (24 keypoints × 2)
        hidden_dims=args["hidden_dims"],          # e.g., [64, 128, 256]
        embedding_dim=args["embedding_dim"],      # e.g., 256
        num_embeddings=args["num_embeddings"],    # e.g., 64 or 512
        sequence_length=args["window_size"],      # e.g., 150 frames
        num_residual_blocks=args["num_residual_blocks"],
        commitment_cost=args["commitment_cost"],
    )

    # Load learned weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # Set to evaluation mode (no dropout, etc.)

    print(f"✓ Loaded VQ-VAE:")
    print(f"  - Codebook size: {args['num_embeddings']}")
    print(f"  - Embedding dim: {args['embedding_dim']}")
    print(f"  - Sequence length: {args['window_size']}")

    return model, args


# ============================================================================
# STEP 2: Compute latent length (needed for code sequences)
# ============================================================================

def get_latent_length(model, input_dim: int, sequence_length: int, device: str):
    """
    Compute the temporal length of latent codes.

    The encoder downsamples the input sequence:
        Input: 150 frames → Latent: 5 timesteps

    Latent length = sequence_length / product(strides)
    Example: 150 / (5 × 3 × 2) = 5

    We compute this by passing a dummy input through the encoder.
    """
    print("\nComputing latent length...")

    # Create dummy input: (batch=1, features=48, time=150)
    dummy_input = torch.zeros(1, input_dim, sequence_length, device=device)

    with torch.no_grad():
        # Pass through encoder
        latent = model.encoder(dummy_input)

    latent_length = latent.shape[-1]
    print(f"✓ Latent length: {latent_length}")
    print(f"  (Each code covers ~{sequence_length // latent_length} frames)")

    return latent_length


# ============================================================================
# STEP 3: Decode a single codebook embedding
# ============================================================================

def decode_single_code(model, code_idx: int, latent_length: int, device: str):
    """
    Decode a SINGLE codebook embedding to see what behavior it represents.

    Steps:
    1. Create sequence where the same code is repeated: [42, 42, 42, 42, 42]
    2. Pass through decoder to reconstruct behavior
    3. Convert to pose format for visualization

    Args:
        model: Trained VQ-VAE
        code_idx: Which code to decode (0 to num_embeddings-1)
        latent_length: Temporal length of latent codes
        device: CPU or CUDA

    Returns:
        pose: (time, 24, 2) - reconstructed fly poses
    """
    print(f"\nDecoding code {code_idx}...")

    # Step 1: Create sequence of repeated codes
    # Shape: (1, latent_length) - batch size 1
    code_sequence = torch.full((1, latent_length), code_idx, dtype=torch.long, device=device)
    # Example: code_idx=42, latent_length=5 → [[42, 42, 42, 42, 42]]

    print(f"  Code sequence: {code_sequence.tolist()}")

    # Step 2: Decode to continuous behavior
    with torch.no_grad():
        # decode_codes internally:
        # - Converts indices to one-hot
        # - Looks up embeddings from codebook
        # - Passes through decoder
        # Output shape: (1, 48, 150) - reconstructed keypoint trajectories
        reconstructed = model.decode_codes(code_sequence)

    print(f"  Reconstructed shape: {reconstructed.shape}")

    # Step 3: Convert to pose format for visualization
    # (1, 48, 150) → (150, 24, 2)
    pose = window_to_pose(reconstructed[0])  # Remove batch dimension

    print(f"  Pose shape: {pose.shape}")
    print(f"✓ Successfully decoded code {code_idx}")

    return pose


# ============================================================================
# STEP 4: Decode a custom sequence of codes
# ============================================================================

def decode_custom_sequence(model, code_sequence: list, latent_length: int, device: str):
    """
    Decode a CUSTOM SEQUENCE of codes to compose behaviors.

    Instead of repeating the same code, we can specify different codes
    at each latent timestep to create composite behaviors.

    Example:
        code_sequence = [12, 42, 17, 5, 28]
        This creates a behavior that transitions through these codes.

    Args:
        model: Trained VQ-VAE
        code_sequence: List of code indices (will be tiled/truncated to latent_length)
        latent_length: Temporal length of latent codes
        device: CPU or CUDA

    Returns:
        pose: (time, 24, 2) - reconstructed fly poses
    """
    print(f"\nDecoding custom sequence: {code_sequence}")

    # Tile or truncate sequence to match latent_length
    if len(code_sequence) < latent_length:
        # Repeat to fill
        repeats = (latent_length + len(code_sequence) - 1) // len(code_sequence)
        code_sequence = (code_sequence * repeats)[:latent_length]
    elif len(code_sequence) > latent_length:
        # Truncate
        code_sequence = code_sequence[:latent_length]

    print(f"  Adjusted to length {latent_length}: {code_sequence}")

    # Convert to tensor
    code_tensor = torch.tensor([code_sequence], dtype=torch.long, device=device)

    # Decode
    with torch.no_grad():
        reconstructed = model.decode_codes(code_tensor)

    # Convert to pose
    pose = window_to_pose(reconstructed[0])

    print(f"✓ Successfully decoded custom sequence")
    return pose


# ============================================================================
# STEP 5: Visualize poses
# ============================================================================

def visualize_pose(pose, frame_indices, title: str, output_path: str):
    """
    Visualize selected frames from a pose sequence.

    Args:
        pose: (time, 24, 2) - pose array
        frame_indices: Which frames to show (e.g., [0, 74, 149])
        title: Plot title
        output_path: Where to save the image
    """
    print(f"\nVisualizing: {title}")
    print(f"  Frames: {frame_indices}")

    num_frames = len(frame_indices)
    fig, axes = plt.subplots(1, num_frames, figsize=(4 * num_frames, 4))

    if num_frames == 1:
        axes = [axes]

    for ax, frame_idx in zip(axes, frame_indices):
        # Handle negative indices
        if frame_idx < 0:
            frame_idx = pose.shape[0] + frame_idx

        # Plot fly pose for this frame
        plot_fly(
            pose[frame_idx],
            ax=ax,
            skelcolor="tab:blue",
            kptcolors="tab:blue",
            kpt_alpha=0.9,
            skel_alpha=0.9,
            kpt_marker="o",
        )

        ax.set_title(f"Frame {frame_idx}")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title)
    fig.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved to {output_path}")


# ============================================================================
# MAIN: Put it all together
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tutorial: VQ-VAE Codebook Visualization"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained VQ-VAE checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output_dir",
        default="./tutorial_outputs",
        help="Where to save visualization images",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference",
    )
    args = parser.parse_args()

    # ========================================================================
    # Example 1: Decode specific codebook embeddings
    # ========================================================================

    print("=" * 80)
    print("EXAMPLE 1: Decode individual codebook embeddings")
    print("=" * 80)

    # Load model
    model, train_args = load_model(args.checkpoint, args.device)

    # Get latent length
    latent_length = get_latent_length(
        model,
        train_args["input_dim"],
        train_args["window_size"],
        args.device,
    )

    # Decode a few example codes
    example_codes = [0, 10, 20, 30]  # Adjust based on your codebook size

    for code_idx in example_codes:
        if code_idx >= train_args["num_embeddings"]:
            print(f"Skipping code {code_idx} (exceeds codebook size)")
            continue

        # Decode the code
        pose = decode_single_code(model, code_idx, latent_length, args.device)

        # Visualize 3 frames: start, middle, end
        visualize_pose(
            pose,
            frame_indices=[0, pose.shape[0] // 2, -1],
            title=f"Codebook Embedding {code_idx}",
            output_path=f"{args.output_dir}/example1_code_{code_idx:03d}.png",
        )

    # ========================================================================
    # Example 2: Decode custom sequences
    # ========================================================================

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Decode custom code sequences")
    print("=" * 80)

    # Example custom sequences
    custom_sequences = [
        [0, 10, 20, 30, 40],     # Linear progression
        [42, 42, 42, 42, 42],    # Single repeated code
        [5, 15, 25, 15, 5],      # Symmetric sequence
    ]

    for i, seq in enumerate(custom_sequences):
        # Filter codes that exceed codebook size
        seq = [c for c in seq if c < train_args["num_embeddings"]]

        if not seq:
            print(f"Skipping sequence {i} (all codes exceed codebook size)")
            continue

        # Decode
        pose = decode_custom_sequence(model, seq, latent_length, args.device)

        # Visualize
        visualize_pose(
            pose,
            frame_indices=[0, pose.shape[0] // 2, -1],
            title=f"Custom Sequence {i+1}: {seq}",
            output_path=f"{args.output_dir}/example2_custom_{i+1}.png",
        )

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Visualizations saved to: {args.output_dir}")
    print()
    print("Key takeaways:")
    print("1. decode_codes() converts discrete indices → continuous behavior")
    print("2. Repeating the same code shows the 'pure' behavior for that code")
    print("3. Custom sequences let you compose different behaviors")
    print("4. Each code covers multiple frames (~30 in this example)")
    print()
    print("Next steps:")
    print("- Examine the output images to understand what each code represents")
    print("- Use visualize_codebook_embeddings.py to decode ALL codes at once")
    print("- Use visualize_reconstructions.py to evaluate reconstruction quality")


if __name__ == "__main__":
    main()
