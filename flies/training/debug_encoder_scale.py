"""
Debug script to check encoder output scale vs codebook scale.
This helps diagnose why VQ loss is so high.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if __package__ in (None, ""):
    # Allow running the script directly without needing to modify PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.dataset import create_dataloaders
from vq_vae.vqvae import VQVAE

# Load checkpoint
checkpoint_path = "training/outputs/run_11_05_25_v1/checkpoint_epoch_250.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
args = checkpoint['args']

print("Loading data...")
train_loader = create_dataloaders(
    train_data_file=args['train_data'],
    window_size=args['window_size'],
    stride=args['stride'],
    batch_size=32,  # Small batch for debugging
    num_workers=0,
    train_fly_ids=None
)

print("Creating model...")
model = VQVAE(
    input_dim=args['input_dim'],
    hidden_dims=args['hidden_dims'],
    embedding_dim=args['embedding_dim'],
    num_embeddings=args['num_embeddings'],
    sequence_length=args['window_size'],
    num_residual_blocks=args['num_residual_blocks'],
    commitment_cost=args['commitment_cost']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("\nAnalyzing encoder outputs and codebook...")
print("=" * 60)

# Get a batch of data
x = next(iter(train_loader))
print(f"Input shape: {x.shape}")
print(f"Input stats: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")

# Encode
with torch.no_grad():
    z = model.encoder(x)
    print(f"\nEncoder output (z) shape: {z.shape}")
    print(f"Encoder output stats:")
    print(f"  Mean: {z.mean():.4f}")
    print(f"  Std:  {z.std():.4f}")
    print(f"  Min:  {z.min():.4f}")
    print(f"  Max:  {z.max():.4f}")
    print(f"  Mean absolute value: {z.abs().mean():.4f}")

# Check codebook
codebook = model.quantizer.embedding.weight
print(f"\nCodebook shape: {codebook.shape}")
print(f"Codebook stats:")
print(f"  Mean: {codebook.mean():.4f}")
print(f"  Std:  {codebook.std():.4f}")
print(f"  Min:  {codebook.min():.4f}")
print(f"  Max:  {codebook.max():.4f}")
print(f"  Mean absolute value: {codebook.abs().mean():.4f}")

# Calculate distances
z_permuted = z.permute(0, 2, 1).contiguous()
z_flat = z_permuted.view(-1, args['embedding_dim'])

distances = (
    torch.sum(z_flat**2, dim=1, keepdim=True) +
    torch.sum(codebook**2, dim=1) -
    2 * torch.matmul(z_flat, codebook.t())
)

min_distances = distances.min(dim=1)[0]
print(f"\nDistance to nearest codebook entry:")
print(f"  Mean: {min_distances.mean():.4f}")
print(f"  Std:  {min_distances.std():.4f}")
print(f"  Min:  {min_distances.min():.4f}")
print(f"  Max:  {min_distances.max():.4f}")

# Check codebook usage
encoding_indices = torch.argmin(distances, dim=1)
unique_codes, counts = torch.unique(encoding_indices, return_counts=True)
print(f"\nCodebook usage (out of {args['num_embeddings']} codes):")
print(f"  Codes used: {len(unique_codes)}")
print(f"  Usage distribution:")
for code, count in zip(unique_codes.numpy(), counts.numpy()):
    percentage = 100 * count / len(encoding_indices)
    print(f"    Code {code:2d}: {count:6d} times ({percentage:5.2f}%)")

# Calculate actual VQ loss components
z_q = torch.matmul(
    torch.nn.functional.one_hot(encoding_indices, args['num_embeddings']).float(),
    codebook
).view(z_permuted.shape)

codebook_loss = torch.nn.functional.mse_loss(z_q, z_permuted.detach())
commitment_loss = torch.nn.functional.mse_loss(z_q.detach(), z_permuted)
vq_loss = codebook_loss + args['commitment_cost'] * commitment_loss

print(f"\nVQ Loss components:")
print(f"  Codebook loss:   {codebook_loss:.4f}")
print(f"  Commitment loss: {commitment_loss:.4f} (× {args['commitment_cost']} = {commitment_loss * args['commitment_cost']:.4f})")
print(f"  Total VQ loss:   {vq_loss:.4f}")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

scale_ratio = z.abs().mean() / codebook.abs().mean()
print(f"Scale ratio (encoder / codebook): {scale_ratio:.2f}x")

if scale_ratio > 10:
    print("❌ PROBLEM: Encoder outputs are much larger than codebook!")
    print("   This causes high VQ loss and poor quantization.")
elif scale_ratio < 0.1:
    print("❌ PROBLEM: Codebook is much larger than encoder outputs!")
    print("   This can also cause quantization issues.")
else:
    print("✅ Scale ratio looks reasonable.")

if min_distances.mean() > 10:
    print(f"❌ PROBLEM: Average distance to nearest code is {min_distances.mean():.2f}")
    print("   This is very high - quantization is poor.")
elif min_distances.mean() > 1:
    print(f"⚠️  WARNING: Average distance to nearest code is {min_distances.mean():.2f}")
    print("   This is somewhat high - quantization could be better.")
else:
    print(f"✅ Distance to nearest code looks good ({min_distances.mean():.2f}).")

if len(unique_codes) < args['num_embeddings'] * 0.5:
    print(f"⚠️  WARNING: Only {len(unique_codes)}/{args['num_embeddings']} codes used")
    print("   Consider reducing codebook size or improving training.")
