"""
Training script for VQ-VAE on fly behavior sequences.

Usage:
    python train.py --train_data /path/to/train.npy --val_data /path/to/val.npy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path
import sys
import os
import math

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import create_dataloaders
from vq_vae.vqvae import VQVAE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)


def random_rotate_batch(windows):
    """
    Randomly rotate each trajectory window around the arena center (0, 0).

    This makes the learned behavior codes rotation-invariant, appropriate for
    circular fly arenas where absolute orientation is arbitrary.

    Args:
        windows: (batch_size, 48, timesteps) tensor of keypoint coordinates

    Returns:
        Rotated windows with same shape
    """
    batch_size, features, timesteps = windows.shape
    num_keypoints = features // 2  # 24 keypoints

    # Reshape to (batch, num_keypoints, 2, timesteps) then permute to (B, T, 24, 2)
    coords = windows.view(batch_size, num_keypoints, 2, timesteps)
    coords = coords.permute(0, 3, 1, 2).contiguous()  # (B, T, 24, 2)

    # Generate random rotation angle for each sample in batch
    theta = torch.rand(batch_size, device=windows.device) * (2 * math.pi)  # (B,)
    cos_theta = torch.cos(theta)  # (B,)
    sin_theta = torch.sin(theta)  # (B,)

    # Build rotation matrix for each sample: [[cos, -sin], [sin, cos]]
    # Shape: (B, 2, 2)
    rotation = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=1),  # First row
        torch.stack([sin_theta, cos_theta], dim=1)    # Second row
    ], dim=1)

    # Flatten coords for batch matrix multiplication: (B, T*24, 2)
    coords_flat = coords.view(batch_size, -1, 2)

    # Apply rotation: (B, T*24, 2) @ (B, 2, 2) = (B, T*24, 2)
    rotated_flat = torch.bmm(coords_flat, rotation)

    # Reshape back to (B, T, 24, 2)
    rotated = rotated_flat.view(batch_size, timesteps, num_keypoints, 2)

    # Convert back to (B, 48, T) format
    rotated = rotated.permute(0, 2, 3, 1).contiguous()  # (B, 24, 2, T)
    rotated = rotated.view(batch_size, features, timesteps)

    return rotated


def train_epoch(model, train_loader, optimizer, device, epoch, use_rotation_aug=False):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0

    for batch_idx, x in enumerate(train_loader):
        x = x.to(device)  # (batch, 48, time)

        if use_rotation_aug:
            with torch.no_grad():
                x = random_rotate_batch(x)

        # Forward pass
        x_recon, vq_loss, perplexity, _, _ = model(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Total loss
        loss = recon_loss + vq_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()

        # Log progress
        if batch_idx % 100 == 0:
            LOG.info(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] | "
                f"Loss: {loss.item():.4f} | "
                f"Recon: {recon_loss.item():.4f} | "
                f"VQ: {vq_loss.item():.4f} | "
                f"Perplexity: {perplexity.item():.2f}"
            )

    # Average metrics
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'perplexity': total_perplexity / n_batches
    }


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)

            # Forward pass
            x_recon, vq_loss, perplexity, _, _ = model(x)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x)

            # Total loss
            loss = recon_loss + vq_loss

            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()

    # Average metrics
    n_batches = len(val_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'perplexity': total_perplexity / n_batches
    }


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Output directory: {output_dir}")

    # Create dataloaders
    LOG.info("Creating dataloaders...")
    if args.val_data:
        train_loader, val_loader = create_dataloaders(
            train_data_file=args.train_data,
            val_data_file=args.val_data,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        train_loader = create_dataloaders(
            train_data_file=args.train_data,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        val_loader = None

    LOG.info(f"Train batches: {len(train_loader)}")
    if val_loader:
        LOG.info(f"Val batches: {len(val_loader)}")

    # Create model
    LOG.info("Creating VQ-VAE model...")
    model = VQVAE(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        sequence_length=args.window_size,
        num_residual_blocks=args.num_residual_blocks,
        commitment_cost=args.commitment_cost
    )
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info(f"Model has {n_params:,} trainable parameters")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    LOG.info("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        LOG.info(f"\n{'='*50}")
        LOG.info(f"Epoch {epoch}/{args.epochs}")
        LOG.info(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            use_rotation_aug=args.augment_rotation
        )
        LOG.info(
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"Recon: {train_metrics['recon_loss']:.4f} | "
            f"VQ: {train_metrics['vq_loss']:.4f} | "
            f"Perplexity: {train_metrics['perplexity']:.2f}"
        )

        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, device)
            LOG.info(
                f"Val   | Loss: {val_metrics['loss']:.4f} | "
                f"Recon: {val_metrics['recon_loss']:.4f} | "
                f"VQ: {val_metrics['vq_loss']:.4f} | "
                f"Perplexity: {val_metrics['perplexity']:.2f}"
            )

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = output_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'args': vars(args)
                }, checkpoint_path)
                LOG.info(f"Saved best model to {checkpoint_path}")

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'args': vars(args)
            }, checkpoint_path)
            LOG.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, final_path)
    LOG.info(f"Saved final model to {final_path}")
    LOG.info("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VQ-VAE on fly behavior data')

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training .npy file')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation .npy file')
    parser.add_argument('--window_size', type=int, default=150,
                        help='Window size for sequences (strides auto-computed to match length)')
    parser.add_argument('--stride', type=int, default=150,
                        help='Stride for sliding windows (use window_size for non-overlapping)')

    # Model arguments
    parser.add_argument('--input_dim', type=int, default=48,
                        help='Input dimension (24 keypoints * 2 coords)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128, 256],
                        help='Hidden dimensions for encoder/decoder')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--num_embeddings', type=int, default=512,
                        help='Number of codebook entries (behavior syllables)')
    parser.add_argument('--num_residual_blocks', type=int, default=2,
                        help='Number of residual blocks per layer')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                        help='Commitment cost (beta) for VQ loss')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--augment_rotation', action='store_true',
                        help='Apply random rotational augmentation during training')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Log arguments
    LOG.info("Arguments:")
    for arg, value in vars(args).items():
        LOG.info(f"  {arg}: {value}")

    main(args)
