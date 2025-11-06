"""
Debug script to verify the normalization fix is working.
Use this AFTER training a new model from scratch with the fixed code.

This will show you:
1. Encoder output scale (should be normalized with std ~ 1)
2. Codebook scale (should match encoder outputs)
3. Distances (should be reasonable, not in thousands)
4. VQ loss components (should be < 10 total)
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import create_dataloaders
from vq_vae.vqvae import VQVAE

print("=" * 80)
print("NORMALIZATION FIX VERIFICATION")
print("=" * 80)

# Small data sample for debugging
print("\nLoading small data sample...")
train_loader = create_dataloaders(
    train_data_file="../../../../data/fly_data/fly_group_train.npy",
    window_size=150,
    stride=150,  # Non-overlapping for speed
    batch_size=32,
    num_workers=0,
    train_fly_ids=None
)

# Create a NEW model with the fixed architecture
print("\nCreating NEW model with GroupNorm fix...")
model = VQVAE(
    input_dim=48,
    hidden_dims=[64, 128, 256],
    embedding_dim=128,  # Increased from 32
    num_embeddings=64,   # Start with 64 codes
    sequence_length=150,
    num_residual_blocks=2,
    commitment_cost=0.25
)

print(f"Model architecture:")
print(f"  - Encoder: 48 → [64, 128, 256] → 128")
print(f"  - Pre-quantizer norm: GroupNorm (THIS IS THE FIX!)")
print(f"  - Quantizer: 64 codes × 128 dims")
print(f"  - Decoder: 128 → [256, 128, 64] → 48")

# Get a batch
x = next(iter(train_loader))
print(f"\n{'='*80}")
print("TESTING WITH RANDOM WEIGHTS (before training)")
print("=" * 80)

print(f"\nInput batch shape: {x.shape}")
print(f"Input stats:")
print(f"  Mean: {x.mean():.4f}")
print(f"  Std:  {x.std():.4f}")
print(f"  Range: [{x.min():.2f}, {x.max():.2f}]")

model.eval()
with torch.no_grad():
    # Encoder output BEFORE normalization
    z_before_norm = model.encoder(x)
    print(f"\nEncoder output (BEFORE GroupNorm):")
    print(f"  Shape: {z_before_norm.shape}")
    print(f"  Mean: {z_before_norm.mean():.4f}")
    print(f"  Std:  {z_before_norm.std():.4f}")
    print(f"  Range: [{z_before_norm.min():.2f}, {z_before_norm.max():.2f}]")

    # Encoder output AFTER normalization
    z_after_norm = model.pre_quantizer_norm(z_before_norm)
    print(f"\nEncoder output (AFTER GroupNorm) ← THIS IS THE FIX:")
    print(f"  Shape: {z_after_norm.shape}")
    print(f"  Mean: {z_after_norm.mean():.4f}")
    print(f"  Std:  {z_after_norm.std():.4f}")
    print(f"  Range: [{z_after_norm.min():.2f}, {z_after_norm.max():.2f}]")

    # Check codebook
    codebook = model.quantizer.embedding.weight
    print(f"\nCodebook (initialized with normal(0,1)):")
    print(f"  Shape: {codebook.shape}")
    print(f"  Mean: {codebook.mean():.4f}")
    print(f"  Std:  {codebook.std():.4f}")
    print(f"  Range: [{codebook.min():.2f}, {codebook.max():.2f}]")

    # Full forward pass
    x_recon, vq_loss, perplexity, _, encoding_indices = model(x)

    print(f"\nVQ Loss components (with random weights):")
    print(f"  VQ loss: {vq_loss.item():.4f}")
    print(f"  Perplexity: {perplexity.item():.2f} / {model.quantizer.num_embeddings}")

    # Calculate scale ratio
    z_scale = z_after_norm.abs().mean()
    codebook_scale = codebook.abs().mean()
    ratio = z_scale / codebook_scale

    print(f"\n{'='*80}")
    print("SCALE ANALYSIS")
    print("=" * 80)
    print(f"Encoder output scale (after norm): {z_scale:.4f}")
    print(f"Codebook scale:                    {codebook_scale:.4f}")
    print(f"Ratio:                             {ratio:.2f}x")

    if 0.5 < ratio < 2.0:
        print("✅ GOOD: Scales are well-matched!")
        print("   The GroupNorm is working correctly.")
    else:
        print("⚠️  Scales are mismatched even after norm")
        print("   This might indicate a problem with the fix.")

print(f"\n{'='*80}")
print("AFTER TRAINING")
print("=" * 80)
print("You should see:")
print("  1. VQ loss < 10 (not 600+)")
print("  2. Perplexity > 50 (good codebook utilization)")
print("  3. Encoder outputs normalized to std ~ 1")
print("  4. Codebook and encoder scales matched")
print("  5. Smooth training curves without exploding losses")
print("=" * 80)
