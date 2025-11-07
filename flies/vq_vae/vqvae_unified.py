"""
Unified VQ-VAE supporting multiple quantization methods.

This is an extension of the standard VQ-VAE that supports:
- Standard VQ
- Improved VQ (lower codebook dim, cosine sim, dead code expiry)
- Finite Scalar Quantization (FSQ)
- Residual VQ (RVQ)
- Lookup Free Quantization (LFQ)

Simply change the quantizer_method parameter to switch between methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .seq_encoder import SequenceEncoder
from .seq_decoder import SequenceDecoder
from .unified_quantizer import UnifiedQuantizer
from .vqvae import compute_strides_for_length


class UnifiedVQVAE(nn.Module):
    """
    Unified VQ-VAE that supports multiple quantization methods.

    Args:
        input_dim (int): Input feature dimension (e.g., 48 for fly behavior)
        hidden_dims (list of int): Hidden dimensions for encoder
        embedding_dim (int): Dimension of latent embeddings
        num_embeddings (int): Codebook size
        sequence_length (int): Expected input sequence length
        num_residual_blocks (int): Number of residual blocks per layer
        commitment_cost (float): Weight for commitment loss
        strides (list of int, optional): Manual stride values
        quantizer_method (str): Quantization method - one of:
            - 'vq': Standard VQ (current implementation)
            - 'vq_improved': VQ with lower codebook_dim + cosine sim + dead code expiry
            - 'fsq': Finite Scalar Quantization (no codebook!)
            - 'rvq': Residual VQ (hierarchical)
            - 'lfq': Lookup Free Quantization (binary latents)
        quantizer_kwargs (dict): Method-specific parameters

    Example:
        # Standard VQ (same as before)
        model = UnifiedVQVAE(..., quantizer_method='vq')

        # Improved VQ
        model = UnifiedVQVAE(..., quantizer_method='vq_improved',
                            quantizer_kwargs={'codebook_dim': 32,
                                            'use_cosine_sim': True,
                                            'kmeans_init': True})

        # FSQ (no codebook!)
        model = UnifiedVQVAE(..., quantizer_method='fsq',
                            quantizer_kwargs={'levels': [8, 5, 5, 5]})

        # Residual VQ
        model = UnifiedVQVAE(..., quantizer_method='rvq',
                            quantizer_kwargs={'num_quantizers': 4})

        # LFQ
        model = UnifiedVQVAE(..., quantizer_method='lfq',
                            quantizer_kwargs={'lfq_dim': 16, 'codebook_size': 64})
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        embedding_dim,
        num_embeddings,
        sequence_length,
        num_residual_blocks=2,
        commitment_cost=0.25,
        strides=None,
        quantizer_method='vq',
        quantizer_kwargs=None
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.quantizer_method = quantizer_method

        # Compute optimal strides if not provided
        if strides is None:
            strides = compute_strides_for_length(sequence_length, len(hidden_dims))
            import logging
            logging.info(f"Auto-computed strides for sequence_length={sequence_length}: {strides}")

        self.strides = strides

        self.encoder = SequenceEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            num_residual_blocks=num_residual_blocks,
            strides=strides
        )

        # Add normalization before quantizer to prevent scale mismatch
        self.pre_quantizer_norm = nn.GroupNorm(num_groups=1, num_channels=embedding_dim)

        # Create unified quantizer
        if quantizer_kwargs is None:
            quantizer_kwargs = {}

        self.quantizer = UnifiedQuantizer(
            method=quantizer_method,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            **quantizer_kwargs
        )

        # Decoder mirrors the encoder
        self.decoder = SequenceDecoder(
            embedding_dim=embedding_dim,
            hidden_dims=list(reversed(hidden_dims)),
            output_dim=input_dim,
            output_length=sequence_length,
            num_residual_blocks=num_residual_blocks,
            strides=list(reversed(strides))
        )

    def forward(self, x):
        """
        Forward pass through Unified VQ-VAE.

        Args:
            x (torch.Tensor): Input sequences (batch_size, input_dim, sequence_length)

        Returns:
            x_recon (torch.Tensor): Reconstructed sequences
            vq_loss (torch.Tensor): VQ loss
            perplexity (torch.Tensor): Codebook usage measure
            encodings (None): Placeholder for compatibility
            encoding_indices (torch.Tensor): Indices into codebook
        """
        # Encode
        z = self.encoder(x)

        # Normalize before quantization
        z = self.pre_quantizer_norm(z)

        # Quantize - unified interface returns (z_q, indices, loss, perplexity)
        z_q, indices, vq_loss, perplexity = self.quantizer(z)

        # Decode
        x_recon = self.decoder(z_q)

        # Return same format as original VQVAE for compatibility
        return x_recon, vq_loss, perplexity, None, indices

    def encode(self, x):
        """
        Encode input and return discrete code indices.

        Args:
            x (torch.Tensor): Input sequences (batch_size, input_dim, sequence_length)

        Returns:
            encoding_indices (torch.Tensor): Discrete code indices
        """
        z = self.encoder(x)
        z = self.pre_quantizer_norm(z)
        _, indices, _, _ = self.quantizer(z)
        return indices

    def decode_codes(self, encoding_indices):
        """
        Decode from discrete code indices back to sequences.

        Note: This method needs to be adapted for different quantizer types
        as they may have different ways of converting indices to embeddings.
        For now, it works with standard VQ and improved VQ.

        Args:
            encoding_indices (torch.Tensor): Code indices (batch_size, time_steps)
                                           or (batch_size, time_steps, num_quantizers) for RVQ

        Returns:
            x_recon (torch.Tensor): Reconstructed sequences
        """
        if self.quantizer_method in ['vq', 'vq_improved']:
            # Standard approach for VQ
            batch_size = encoding_indices.shape[0]
            time_steps = encoding_indices.shape[1] if encoding_indices.dim() == 2 else encoding_indices.shape[1]

            if self.quantizer_method == 'vq':
                # Use original VectorQuantizer's embedding table
                encodings = F.one_hot(encoding_indices, self.quantizer.num_embeddings).float()
                encodings = encodings.view(-1, self.quantizer.num_embeddings)
                z_q = torch.matmul(encodings, self.quantizer.quantizer.embedding.weight)
                z_q = z_q.view(batch_size, time_steps, self.quantizer.embedding_dim)
                z_q = z_q.permute(0, 2, 1).contiguous()

            else:  # vq_improved
                # VectorQuantize has indices_to_codes method
                z_q = self.quantizer.quantizer.indices_to_codes(encoding_indices)
                # z_q is (B, T, C), need (B, C, T)
                z_q = z_q.permute(0, 2, 1).contiguous()

        elif self.quantizer_method == 'rvq':
            # ResidualVQ has get_output_from_indices
            z_q = self.quantizer.quantizer.get_output_from_indices(encoding_indices)
            # z_q is (B, T, C), need (B, C, T)
            z_q = z_q.permute(0, 2, 1).contiguous()

        elif self.quantizer_method == 'fsq':
            # FSQ has indices_to_codes
            z_q_fsq = self.quantizer.quantizer.indices_to_codes(encoding_indices)
            # Project back to embedding dim
            if z_q_fsq.dim() == 2:
                z_q_fsq = z_q_fsq.unsqueeze(0)  # Add batch if needed
            z_q = self.quantizer.post_fsq_proj(z_q_fsq)
            z_q = z_q.permute(0, 2, 1).contiguous()

        elif self.quantizer_method == 'lfq':
            # LFQ has indices_to_codes
            z_q_lfq = self.quantizer.quantizer.indices_to_codes(encoding_indices)
            # Project back to embedding dim
            if z_q_lfq.dim() == 2:
                z_q_lfq = z_q_lfq.unsqueeze(0)
            z_q = self.quantizer.post_lfq_proj(z_q_lfq)
            z_q = z_q.permute(0, 2, 1).contiguous()

        else:
            raise NotImplementedError(f"decode_codes not implemented for {self.quantizer_method}")

        # Decode
        x_recon = self.decoder(z_q)
        return x_recon

    def __repr__(self):
        return (
            f"UnifiedVQVAE(\n"
            f"  quantizer_method={self.quantizer_method},\n"
            f"  codebook_size={self.quantizer.get_codebook_size()},\n"
            f"  embedding_dim={self.quantizer.embedding_dim},\n"
            f"  sequence_length={self.sequence_length},\n"
            f"  strides={self.strides}\n"
            f")"
        )
