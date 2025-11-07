"""
Unified quantizer wrapper supporting multiple quantization methods.

This module provides a consistent interface for:
1. Standard VQ with improvements (lower codebook_dim, cosine_sim, dead code expiry)
2. Finite Scalar Quantization (FSQ) - no codebook, simple rounding
3. Residual VQ (RVQ) - hierarchical multi-stage quantization
4. Lookup Free Quantization (LFQ) - binary latents without lookup

All quantizers return the same output format for easy comparison:
    (quantized, indices, loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import VectorQuantizer


class UnifiedQuantizer(nn.Module):
    """
    Unified wrapper for different quantization methods.

    Args:
        method (str): One of ['vq', 'vq_improved', 'fsq', 'rvq', 'lfq']
        embedding_dim (int): Dimension of encoder output
        num_embeddings (int): Codebook size (for VQ-based methods)
        commitment_cost (float): Weight for commitment loss
        **method_specific_kwargs: Additional parameters for specific methods

    Example:
        # Standard VQ
        quantizer = UnifiedQuantizer('vq', embedding_dim=256, num_embeddings=64)

        # Improved VQ with lower codebook dimension
        quantizer = UnifiedQuantizer('vq_improved', embedding_dim=256,
                                    num_embeddings=64, codebook_dim=32,
                                    use_cosine_sim=True, kmeans_init=True)

        # FSQ (no codebook!)
        quantizer = UnifiedQuantizer('fsq', embedding_dim=256,
                                    levels=[8, 5, 5, 5])  # ~1000 codes

        # Residual VQ
        quantizer = UnifiedQuantizer('rvq', embedding_dim=256,
                                    num_embeddings=32, num_quantizers=4)

        # LFQ
        quantizer = UnifiedQuantizer('lfq', embedding_dim=16,  # LFQ needs smaller dim
                                    codebook_size=64)
    """

    def __init__(
        self,
        method='vq',
        embedding_dim=256,
        num_embeddings=64,
        commitment_cost=0.25,
        **method_kwargs
    ):
        super().__init__()

        self.method = method
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Create the appropriate quantizer based on method
        if method == 'vq':
            # Standard VQ (your current implementation)
            self.quantizer = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost
            )
            self._forward = self._forward_vq

        elif method == 'vq_improved':
            # Improved VQ using vector-quantize-pytorch library
            # Supports: lower codebook_dim, cosine_sim, dead code expiry, kmeans init
            try:
                from vector_quantize_pytorch import VectorQuantize
            except ImportError:
                raise ImportError(
                    "vector-quantize-pytorch not installed. "
                    "Run: pip install vector-quantize-pytorch"
                )

            self.quantizer = VectorQuantize(
                dim=embedding_dim,
                codebook_size=num_embeddings,
                codebook_dim=method_kwargs.get('codebook_dim', 32),  # Lower dim codebook
                use_cosine_sim=method_kwargs.get('use_cosine_sim', True),
                threshold_ema_dead_code=method_kwargs.get('threshold_ema_dead_code', 2),
                kmeans_init=method_kwargs.get('kmeans_init', True),
                kmeans_iters=method_kwargs.get('kmeans_iters', 10),
                decay=method_kwargs.get('decay', 0.8),
                commitment_weight=commitment_cost,
                accept_image_fmap=False,  # We're using sequences, not images
                channel_last=True  # True because we permute to (B, T, C) in _forward_vq_improved
            )
            self._forward = self._forward_vq_improved

        elif method == 'fsq':
            # Finite Scalar Quantization - no codebook!
            try:
                from vector_quantize_pytorch import FSQ
            except ImportError:
                raise ImportError(
                    "vector-quantize-pytorch not installed. "
                    "Run: pip install vector-quantize-pytorch"
                )

            levels = method_kwargs.get('levels', [8, 5, 5, 5])  # ~1000 codes
            # FSQ requires dim to match number of levels
            assert len(levels) <= embedding_dim, \
                f"FSQ levels ({len(levels)}) must be <= embedding_dim ({embedding_dim})"

            # Store levels for get_codebook_size()
            self.fsq_levels = levels

            # We need a projection to go from embedding_dim to len(levels)
            self.pre_fsq_proj = nn.Linear(embedding_dim, len(levels))
            self.post_fsq_proj = nn.Linear(len(levels), embedding_dim)

            self.quantizer = FSQ(
                levels=levels,
                dim=len(levels),
                num_codebooks=1,
                keep_num_codebooks_dim=False
            )
            self._forward = self._forward_fsq

        elif method == 'rvq':
            # Residual VQ - hierarchical quantization
            try:
                from vector_quantize_pytorch import ResidualVQ
            except ImportError:
                raise ImportError(
                    "vector-quantize-pytorch not installed. "
                    "Run: pip install vector-quantize-pytorch"
                )

            self.quantizer = ResidualVQ(
                dim=embedding_dim,
                num_quantizers=method_kwargs.get('num_quantizers', 4),
                codebook_size=num_embeddings,
                codebook_dim=method_kwargs.get('codebook_dim', None),
                kmeans_init=method_kwargs.get('kmeans_init', True),
                kmeans_iters=method_kwargs.get('kmeans_iters', 10),
                threshold_ema_dead_code=method_kwargs.get('threshold_ema_dead_code', 2),
                shared_codebook=method_kwargs.get('shared_codebook', False),
                stochastic_sample_codes=method_kwargs.get('stochastic_sample_codes', False),
                commitment_weight=commitment_cost
                # Note: ResidualVQ doesn't support channel_first parameter
            )
            self._forward = self._forward_rvq

        elif method == 'lfq':
            # Lookup Free Quantization - binary latents
            try:
                from vector_quantize_pytorch import LFQ
            except ImportError:
                raise ImportError(
                    "vector-quantize-pytorch not installed. "
                    "Run: pip install vector-quantize-pytorch"
                )

            # LFQ works best with smaller dimensions
            lfq_dim = method_kwargs.get('lfq_dim', 16)
            codebook_size = method_kwargs.get('codebook_size', 64)

            # Need projections to/from LFQ dim
            self.pre_lfq_proj = nn.Linear(embedding_dim, lfq_dim)
            self.post_lfq_proj = nn.Linear(lfq_dim, embedding_dim)

            self.quantizer = LFQ(
                dim=lfq_dim,
                codebook_size=codebook_size,
                entropy_loss_weight=method_kwargs.get('entropy_loss_weight', 0.1),
                diversity_gamma=method_kwargs.get('diversity_gamma', 1.0),
                commitment_loss_weight=commitment_cost,
                num_codebooks=method_kwargs.get('num_codebooks', 1),
                keep_num_codebooks_dim=False
                # Note: LFQ doesn't support channel_first parameter
                # Expects (B, ..., dim) format which we provide via permutation
            )
            self._forward = self._forward_lfq

        else:
            raise ValueError(
                f"Unknown quantization method: {method}. "
                f"Choose from: ['vq', 'vq_improved', 'fsq', 'rvq', 'lfq']"
            )

    def forward(self, z):
        """
        Forward pass - delegates to method-specific implementation.

        Args:
            z (torch.Tensor): Encoder output (B, C, T)

        Returns:
            z_q (torch.Tensor): Quantized output (B, C, T)
            indices (torch.Tensor): Quantized indices
            loss (torch.Tensor): Quantization loss
            perplexity (torch.Tensor): Codebook usage metric (if available)
        """
        return self._forward(z)

    def _forward_vq(self, z):
        """Standard VQ forward (your current implementation)"""
        z_q, loss, perplexity, encodings, indices = self.quantizer(z)
        return z_q, indices, loss, perplexity

    def _forward_vq_improved(self, z):
        """Improved VQ forward using vector-quantize-pytorch"""
        # VectorQuantize expects (B, T, C) or (B, C, H, W)
        # We have (B, C, T) so we need to permute
        z = z.permute(0, 2, 1)  # (B, T, C)

        # VectorQuantize returns (quantized, indices, loss)
        z_q, indices, loss = self.quantizer(z)

        # Permute back to (B, C, T)
        z_q = z_q.permute(0, 2, 1)

        # Compute perplexity from indices
        perplexity = self._compute_perplexity(indices)

        return z_q, indices, loss, perplexity

    def _forward_fsq(self, z):
        """FSQ forward - no codebook, just rounding"""
        # z: (B, C, T)
        z = z.permute(0, 2, 1)  # (B, T, C)

        # Project to FSQ dimension
        z_proj = self.pre_fsq_proj(z)  # (B, T, levels)

        # FSQ quantization (no loss in FSQ!)
        z_q, indices = self.quantizer(z_proj)

        # Project back to embedding dimension
        z_q = self.post_fsq_proj(z_q)  # (B, T, C)

        # Permute back
        z_q = z_q.permute(0, 2, 1)  # (B, C, T)

        # FSQ has no auxiliary loss, but we compute commitment loss for consistency
        loss = F.mse_loss(z_q.detach(), z.permute(0, 2, 1)) * self.quantizer.codebook_size

        perplexity = self._compute_perplexity(indices)

        return z_q, indices, loss, perplexity

    def _forward_rvq(self, z):
        """Residual VQ forward - hierarchical quantization"""
        # z: (B, C, T)
        z = z.permute(0, 2, 1)  # (B, T, C)

        # ResidualVQ returns (quantized, indices, loss)
        # indices shape: (B, T, num_quantizers)
        z_q, indices, loss = self.quantizer(z)

        # Permute back
        z_q = z_q.permute(0, 2, 1)  # (B, C, T)

        # For RVQ, we compute perplexity across all quantizers
        perplexity = self._compute_perplexity(indices)

        # Loss might be per-quantizer, so take mean
        if loss.dim() > 0:
            loss = loss.mean()

        return z_q, indices, loss, perplexity

    def _forward_lfq(self, z):
        """LFQ forward - binary latents"""
        # z: (B, C, T)
        z = z.permute(0, 2, 1)  # (B, T, C)

        # Project to LFQ dimension
        z_proj = self.pre_lfq_proj(z)  # (B, T, lfq_dim)

        # LFQ returns (quantized, indices, entropy_loss)
        z_q, indices, loss = self.quantizer(z_proj)

        # Project back
        z_q = self.post_lfq_proj(z_q)  # (B, T, C)

        # Permute back
        z_q = z_q.permute(0, 2, 1)  # (B, C, T)

        perplexity = self._compute_perplexity(indices)

        return z_q, indices, loss, perplexity

    def _compute_perplexity(self, indices):
        """Compute perplexity from indices to measure codebook usage"""
        # Get the actual codebook size for this method
        codebook_size = self.get_codebook_size()

        # Handle different index shapes
        if indices.dim() == 3:  # (B, T, num_quantizers) for RVQ
            indices = indices.reshape(-1, indices.shape[-1])
            # Compute perplexity for each quantizer and average
            perplexities = []
            for q in range(indices.shape[-1]):
                q_indices = indices[:, q]
                encodings = F.one_hot(q_indices, codebook_size).float()
                avg_probs = encodings.mean(dim=0)
                perp = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
                perplexities.append(perp)
            return torch.stack(perplexities).mean()
        else:  # (B, T) for standard VQ/FSQ/LFQ
            indices_flat = indices.reshape(-1)
            encodings = F.one_hot(indices_flat, codebook_size).float()
            avg_probs = encodings.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            return perplexity

    def get_codebook_size(self):
        """Return effective codebook size"""
        if self.method == 'fsq':
            # FSQ codebook size is product of levels
            import numpy as np
            return int(np.prod(self.fsq_levels))
        elif self.method == 'lfq':
            return self.quantizer.codebook_size
        else:
            return self.num_embeddings

    def __repr__(self):
        return (
            f"UnifiedQuantizer(method={self.method}, "
            f"embedding_dim={self.embedding_dim}, "
            f"codebook_size={self.get_codebook_size()})"
        )
