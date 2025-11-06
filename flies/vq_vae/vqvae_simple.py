"""
VQ-VAE with ONLY initialization fix (no pre-quantizer normalization).
This is closer to the original paper.

Use this if you want the simplest possible fix.
"""
import torch
import torch.nn as nn
from .seq_encoder import SequenceEncoder
from .seq_decoder import SequenceDecoder
from .quantizer_simple import VectorQuantizerSimple

class VQVAESimple(nn.Module):
    """
    VQ-VAE with only the codebook initialization fix.
    No pre-quantizer normalization.

    This is closer to the original VQ-VAE paper approach.
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
        strides=None
    ):
        super().__init__()

        self.sequence_length = sequence_length

        # Import here to avoid circular dependency
        from .vqvae import compute_strides_for_length

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

        # NO pre-quantizer normalization!
        # Just use the simple quantizer with fixed initialization

        self.quantizer = VectorQuantizerSimple(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )

        self.decoder = SequenceDecoder(
            embedding_dim=embedding_dim,
            hidden_dims=list(reversed(hidden_dims)),
            output_dim=input_dim,
            output_length=sequence_length,
            num_residual_blocks=num_residual_blocks,
            strides=list(reversed(strides))
        )

    def forward(self, x):
        # Encode
        z = self.encoder(x)

        # Quantize (no normalization!)
        z_q, vq_loss, perplexity, encodings, encoding_indices = self.quantizer(z)

        # Decode
        x_recon = self.decoder(z_q)

        return x_recon, vq_loss, perplexity, encodings, encoding_indices

    def encode(self, x):
        z = self.encoder(x)
        _, _, _, _, encoding_indices = self.quantizer(z)
        return encoding_indices

    def decode_codes(self, encoding_indices):
        batch_size, time_steps = encoding_indices.shape
        encodings = torch.nn.functional.one_hot(encoding_indices, self.quantizer.num_embeddings).float()
        encodings = encodings.view(-1, self.quantizer.num_embeddings)
        z_q = torch.matmul(encodings, self.quantizer.embedding.weight)
        z_q = z_q.view(batch_size, time_steps, self.quantizer.embedding_dim)
        z_q = z_q.permute(0, 2, 1).contiguous()
        x_recon = self.decoder(z_q)
        return x_recon
