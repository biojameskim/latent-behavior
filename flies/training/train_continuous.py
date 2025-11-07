"""
Unified training script for all model types:
1. VQ-VAE (discrete)
2. VAE (continuous)
3. Transformer/LSTM forecaster (continuous-to-continuous)
4. Hybrid (continuous → discrete token extraction)

Usage:
    python train_continuous.py --model_type vqvae --config configs/vqvae.yaml
    python train_continuous.py --model_type vae --config configs/vae.yaml
    python train_continuous.py --model_type transformer --config configs/transformer.yaml
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import load_and_preprocess_for_vqvae
from data.dataset import FlyKeypointDataset
from vq_vae.vqvae import VQVAE
from vq_vae.vae_continuous import ContinuousVAE, BetaVAE, AnnealedVAE
from forecasting.continuous_forecaster import TransformerForecaster, LSTMForecaster


def create_model(model_type: str, config: Dict) -> nn.Module:
    """Create model based on type."""
    if model_type == 'vqvae':
        return VQVAE(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            embedding_dim=config['embedding_dim'],
            num_embeddings=config['num_embeddings'],
            num_residual_blocks=config['num_residual_blocks'],
            commitment_cost=config['commitment_cost'],
            sequence_length=config['window_size'],
        )
    elif model_type == 'vae':
        return ContinuousVAE(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            latent_dim=config['latent_dim'],
            num_residual_blocks=config['num_residual_blocks'],
            kl_weight=config.get('kl_weight', 1.0),
            sequence_length=config['window_size'],
        )
    elif model_type == 'beta_vae':
        return BetaVAE(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            latent_dim=config['latent_dim'],
            num_residual_blocks=config['num_residual_blocks'],
            kl_weight=config.get('kl_weight', 4.0),
            sequence_length=config['window_size'],
        )
    elif model_type == 'annealed_vae':
        return AnnealedVAE(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            latent_dim=config['latent_dim'],
            num_residual_blocks=config['num_residual_blocks'],
            kl_weight_max=config.get('kl_weight', 1.0),
            sequence_length=config['window_size'],
        )
    elif model_type == 'transformer':
        return TransformerForecaster(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            context_length=config['context_length'],
            forecast_length=config['forecast_length'],
        )
    elif model_type == 'lstm':
        return LSTMForecaster(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            context_length=config['context_length'],
            forecast_length=config['forecast_length'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    model_type: str,
    epoch: int = 0,
    total_epochs: int = 100,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {}

    pbar = tqdm(dataloader, desc=f"Training epoch {epoch+1}")
    for batch in pbar:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)

        optimizer.zero_grad()

        # Forward pass based on model type
        if model_type == 'vqvae':
            x_recon, vq_loss, perplexity, encodings = model(x)
            recon_loss = nn.functional.mse_loss(x_recon, x)
            loss = recon_loss + vq_loss

            # Track components
            loss_components.setdefault('recon_loss', []).append(recon_loss.item())
            loss_components.setdefault('vq_loss', []).append(vq_loss.item())
            loss_components.setdefault('perplexity', []).append(perplexity.item())

        elif model_type in ['vae', 'beta_vae', 'annealed_vae']:
            # Update KL weight for annealed VAE
            if model_type == 'annealed_vae':
                model.update_kl_weight(epoch, total_epochs)

            x_recon, info = model(x)
            loss, loss_dict = model.compute_loss(x, x_recon, info)

            for k, v in loss_dict.items():
                loss_components.setdefault(k, []).append(v)

        elif model_type in ['transformer', 'lstm']:
            loss, loss_dict = model.compute_loss(x)

            for k, v in loss_dict.items():
                loss_components.setdefault(k, []).append(v)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: sum(v) / len(v) for k, v in loss_components.items()}

    return {'total_loss': avg_loss, **avg_components}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    model_type: str,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0
    loss_components = {}

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            # Forward pass based on model type
            if model_type == 'vqvae':
                x_recon, vq_loss, perplexity, encodings = model(x)
                recon_loss = nn.functional.mse_loss(x_recon, x)
                loss = recon_loss + vq_loss

                loss_components.setdefault('recon_loss', []).append(recon_loss.item())
                loss_components.setdefault('vq_loss', []).append(vq_loss.item())
                loss_components.setdefault('perplexity', []).append(perplexity.item())

            elif model_type in ['vae', 'beta_vae', 'annealed_vae']:
                x_recon, info = model(x)
                loss, loss_dict = model.compute_loss(x, x_recon, info)

                for k, v in loss_dict.items():
                    loss_components.setdefault(k, []).append(v)

            elif model_type in ['transformer', 'lstm']:
                loss, loss_dict = model.compute_loss(x)

                for k, v in loss_dict.items():
                    loss_components.setdefault(k, []).append(v)

            total_loss += loss.item()

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: sum(v) / len(v) for k, v in loss_components.items()}

    return {'total_loss': avg_loss, **avg_components}


def train(
    model_type: str,
    config: Dict,
    output_dir: str,
    use_wandb: bool = False,
    wandb_project: str = 'fly-behavior',
):
    """Main training loop."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb
    if use_wandb:
        wandb.init(project=wandb_project, config=config, name=f"{model_type}_{config.get('run_name', '')}")

    # Load data
    print("Loading data...")
    data_file = config['data_file']
    fly_split_file = config['fly_split_file']

    # For forecasting models, window size should be context + forecast
    if model_type in ['transformer', 'lstm']:
        window_size = config['context_length'] + config['forecast_length']
    else:
        window_size = config['window_size']

    train_dataset = FlyKeypointDataset(
        data_file=data_file,
        fly_split_file=fly_split_file,
        split_name='train',
        window_size=window_size,
        stride=config.get('stride', window_size),
    )

    val_dataset = FlyKeypointDataset(
        data_file=data_file,
        fly_split_file=fly_split_file,
        split_name='val',
        window_size=window_size,
        stride=window_size,  # Non-overlapping for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create model
    print(f"Creating {model_type} model...")
    model = create_model(model_type, config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.0),
        betas=config.get('betas', (0.9, 0.99)),
    )

    # Training loop
    best_val_loss = float('inf')
    epochs = config['epochs']

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, model_type, epoch, epochs)
        print(f"Train loss: {train_metrics['total_loss']:.4f}")
        for k, v in train_metrics.items():
            if k != 'total_loss':
                print(f"  {k}: {v:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, device, model_type)
        print(f"Val loss: {val_metrics['total_loss']:.4f}")
        for k, v in val_metrics.items():
            if k != 'total_loss':
                print(f"  {k}: {v:.4f}")

        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
            })

        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'config': config,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"Saved best model (val_loss={best_val_loss:.4f})")

        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'config': config,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    if use_wandb:
        wandb.finish()

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['vqvae', 'vae', 'beta_vae', 'annealed_vae', 'transformer', 'lstm'])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='fly-behavior')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Train
    train(
        model_type=args.model_type,
        config=config,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
