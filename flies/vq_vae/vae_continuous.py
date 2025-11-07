"""
Standard VAE with continuous latent space for behavior modeling.

This serves as a direct comparison to VQ-VAE:
- VQ-VAE: Encoder → Discrete codes → Decoder
- VAE: Encoder → Continuous z ~ N(μ, σ) → Decoder

Key differences:
1. No discrete bottleneck (no quantization)
2. Probabilistic latent space with KL regularization
3. Can capture subtle continuous variations in behavior
4. Latent space is smooth and interpolatable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .seq_encoder import SeqEncoder
from .seq_decoder import SeqDecoder


class ContinuousVAE(nn.Module):
    """
    Variational Autoencoder with continuous latent space.

    Architecture:
        Input (batch, 48, 150) → Encoder → μ, log_σ² (batch, latent_dim, T)
        → Sample z ~ N(μ, σ) → Decoder → Output (batch, 48, 150)

    Args:
        input_dim: Number of input features (48 for 24 keypoints × 2)
        hidden_dims: List of hidden dimensions for encoder/decoder layers
        latent_dim: Dimension of continuous latent space (replaces embedding_dim in VQ-VAE)
        num_residual_blocks: Number of residual blocks per layer
        kl_weight: Weight for KL divergence loss (β in β-VAE)
    """

    def __init__(
        self,
        input_dim: int = 48,
        hidden_dims: list = [64, 128, 256],
        latent_dim: int = 128,
        num_residual_blocks: int = 2,
        kl_weight: float = 1.0,
        sequence_length: int = 150,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Encoder: produces continuous embeddings
        self.encoder = SeqEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=latent_dim,
            num_residual_blocks=num_residual_blocks,
            sequence_length=sequence_length,
        )

        # Compute compressed sequence length
        self.compressed_length = self.encoder.compute_compressed_length(sequence_length)

        # Split encoder output into mean and log-variance
        # We'll project from latent_dim to 2*latent_dim, then split
        self.fc_mu = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)
        self.fc_logvar = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)

        # Decoder: reconstructs from continuous latent
        self.decoder = SeqDecoder(
            embedding_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            output_dim=input_dim,
            num_residual_blocks=num_residual_blocks,
            sequence_length=sequence_length,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor (batch, input_dim, seq_len)

        Returns:
            mu: Mean of latent distribution (batch, latent_dim, compressed_len)
            logvar: Log-variance of latent distribution (batch, latent_dim, compressed_len)
        """
        h = self.encoder(x)  # (batch, latent_dim, compressed_len)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, 1)

        Args:
            mu: Mean (batch, latent_dim, compressed_len)
            logvar: Log-variance (batch, latent_dim, compressed_len)

        Returns:
            z: Sampled latent vector (batch, latent_dim, compressed_len)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.

        Args:
            z: Latent vector (batch, latent_dim, compressed_len)

        Returns:
            x_recon: Reconstructed input (batch, input_dim, seq_len)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass: encode → sample → decode

        Args:
            x: Input tensor (batch, input_dim, seq_len)

        Returns:
            x_recon: Reconstructed input (batch, input_dim, seq_len)
            info: Dictionary containing:
                - 'mu': Latent mean
                - 'logvar': Latent log-variance
                - 'z': Sampled latent vector
                - 'kl_loss': KL divergence loss
        """
        # Encode to latent distribution
        mu, logvar = self.encode(x)

        # Sample latent vector
        z = self.reparameterize(mu, logvar)

        # Decode to reconstruction
        x_recon = self.decode(z)

        # Compute KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2])
        kl_loss = kl_loss.mean()  # Average over batch

        info = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'kl_loss': kl_loss,
        }

        return x_recon, info

    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                     info: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VAE loss: reconstruction + weighted KL divergence.

        Loss = MSE(x, x_recon) + β * KL(q(z|x) || p(z))

        Args:
            x: Original input
            x_recon: Reconstructed input
            info: Dictionary from forward() containing kl_loss

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)

        # KL divergence loss (already computed in forward)
        kl_loss = info['kl_loss']

        # Total loss (β-VAE formulation)
        total_loss = recon_loss + self.kl_weight * kl_loss

        loss_dict = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
        }

        return total_loss, loss_dict

    def get_latent_codes(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """
        Extract continuous latent representations.

        Args:
            x: Input tensor (batch, input_dim, seq_len)
            use_mean: If True, return mean μ; if False, sample from distribution

        Returns:
            Latent codes (batch, latent_dim, compressed_len)
        """
        mu, logvar = self.encode(x)
        if use_mean:
            return mu
        else:
            return self.reparameterize(mu, logvar)


class BetaVAE(ContinuousVAE):
    """
    β-VAE variant with adjustable KL weight for disentanglement.

    Higher β (e.g., 4-10) encourages more disentangled representations
    at the cost of reconstruction quality.

    Reference: Higgins et al. 2017 - "β-VAE: Learning Basic Visual Concepts
    with a Constrained Variational Framework"
    """
    def __init__(self, *args, kl_weight: float = 4.0, **kwargs):
        super().__init__(*args, kl_weight=kl_weight, **kwargs)


class AnnealedVAE(ContinuousVAE):
    """
    VAE with KL annealing schedule.

    Gradually increases KL weight from 0 to target value over training.
    Helps avoid posterior collapse and improves training stability.

    Usage:
        model = AnnealedVAE(...)
        for epoch in range(num_epochs):
            model.update_kl_weight(epoch, total_epochs=100)
            # ... training loop
    """
    def __init__(self, *args, kl_weight_max: float = 1.0, **kwargs):
        super().__init__(*args, kl_weight=0.0, **kwargs)
        self.kl_weight_max = kl_weight_max

    def update_kl_weight(self, current_epoch: int, total_epochs: int):
        """Linear annealing schedule."""
        self.kl_weight = self.kl_weight_max * min(1.0, current_epoch / (total_epochs * 0.5))
