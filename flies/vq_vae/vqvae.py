import torch
import torch.nn as nn
import torch.nn.functional as F
from .seq_encoder import SequenceEncoder
from .seq_decoder import SequenceDecoder
from .quantizer import VectorQuantizer


def compute_strides_for_length(sequence_length, num_layers, prefer_small=True):
    """
    Compute stride values that divide the sequence length evenly.

    For example, 150 frames with 3 layers: [5, 3, 2] → 150→30→10→5

    Args:
        sequence_length: Target sequence length
        num_layers: Number of downsampling layers
        prefer_small: If True, prefer smaller strides (more gradual downsampling)

    Returns:
        List of strides (one per layer)

    Raises:
        ValueError: If sequence_length cannot be evenly divided by the computed strides
    """
    # Find all factors of sequence_length
    def factorize(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    factors = factorize(sequence_length)

    # Distribute factors across layers
    if len(factors) < num_layers:
        # Not enough factors - need to check if this will work
        strides = [2] * num_layers

        # Verify that the strides evenly divide sequence_length
        product = 1
        for s in strides:
            product *= s

        if sequence_length % product != 0:
            raise ValueError(
                f"Cannot create {num_layers} downsampling layers for sequence_length={sequence_length}. "
                f"The sequence length has insufficient factors ({factors}). "
                f"Suggested alternatives: Use window_size with more factors (e.g., 120, 144, 150, 160, 192, 200), "
                f"or reduce the number of hidden layers (currently {num_layers})."
            )
    else:
        # Distribute factors
        if prefer_small:
            factors = sorted(factors, reverse=True)  # Larger strides first
        else:
            factors = sorted(factors)

        strides = [1] * num_layers
        for i, f in enumerate(factors):
            strides[i % num_layers] *= f

    # Final validation: verify strides evenly divide sequence_length
    test_length = sequence_length
    for stride in strides:
        if test_length % stride != 0:
            raise ValueError(
                f"Computed strides {strides} do not evenly divide sequence_length={sequence_length}. "
                f"This is a bug in stride computation. Please report this issue."
            )
        test_length //= stride

    return strides


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder for temporal sequence modeling.

    This model learns to:
    1. Encode continuous sequences into latent representations (Encoder)
    2. Quantize to discrete codes from a learned codebook (VectorQuantizer)
    3. Decode back to reconstruct the original sequence (Decoder)

    For fly behavior, this discovers a discrete vocabulary of behavior "syllables"
    that can be combined to describe complex behavioral sequences.

    Args:
        input_dim (int): Input feature dimension (e.g., 48 for 24 keypoints * 2 coordinates)
        hidden_dims (list of int): Hidden dimensions for encoder (e.g., [64, 128, 256])
        embedding_dim (int): Dimension of latent embeddings (e.g., 512)
        num_embeddings (int): Codebook size - number of discrete behavior codes (e.g., 512)
        sequence_length (int): Expected input sequence length (e.g., 150 frames)
            - Must have sufficient prime factors for the number of layers
            - Good: 120, 144, 150, 160, 192, 200 (highly composite)
            - Bad: Primes (97, 101) or numbers with few factors
        num_residual_blocks (int): Number of residual blocks per layer (default: 2)
        commitment_cost (float): Weight for commitment loss (default: 0.25)
        strides (list of int, optional): Manual stride values (auto-computed if None)
            - Must evenly divide sequence_length when multiplied together
            - Example: strides=[5,3,2] for sequence_length=150

    Forward returns:
        x_recon: Reconstructed sequence
        vq_loss: Vector quantization loss (codebook + commitment)
        perplexity: Codebook usage metric
        encodings: One-hot encodings of used codes
        encoding_indices: Indices of behavior codes for each time step
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

        # Compute optimal strides if not provided
        if strides is None:
            strides = compute_strides_for_length(sequence_length, len(hidden_dims))
            # Log the computed strides for user awareness
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

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )

        # Decoder mirrors the encoder (reverse hidden_dims and strides)
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
        Forward pass through VQ-VAE.

        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, input_dim, sequence_length)

        Returns:
            x_recon (torch.Tensor): Reconstructed sequences, same shape as x
            vq_loss (torch.Tensor): VQ loss (codebook + commitment)
            perplexity (torch.Tensor): Measure of codebook usage (higher = more codes used)
            encodings (torch.Tensor): One-hot vectors of which codes were selected
            encoding_indices (torch.Tensor): Indices into codebook for each time step
        """
        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, vq_loss, perplexity, encodings, encoding_indices = self.quantizer(z)

        # Decode
        x_recon = self.decoder(z_q)

        return x_recon, vq_loss, perplexity, encodings, encoding_indices

    def encode(self, x):
        """
        Encode input and return discrete code indices.

        Useful for analysis: extract behavior syllable sequences.

        Args:
            x (torch.Tensor): Input sequences (batch_size, input_dim, sequence_length)

        Returns:
            encoding_indices (torch.Tensor): Discrete code indices (batch_size, reduced_time_steps)
        """
        z = self.encoder(x)
        _, _, _, _, encoding_indices = self.quantizer(z)
        return encoding_indices

    def decode_codes(self, encoding_indices):
        """
        Decode from discrete code indices back to sequences.

        Useful for generating behavior from learned syllables.

        Args:
            encoding_indices (torch.Tensor): Code indices (batch_size, time_steps)

        Returns:
            x_recon (torch.Tensor): Reconstructed sequences
        """
        # Get embeddings from codebook
        batch_size, time_steps = encoding_indices.shape

        # Convert indices to one-hot
        encodings = F.one_hot(encoding_indices, self.quantizer.num_embeddings).float()
        encodings = encodings.view(-1, self.quantizer.num_embeddings)

        # Get quantized embeddings
        z_q = torch.matmul(encodings, self.quantizer.embedding.weight)
        z_q = z_q.view(batch_size, time_steps, self.quantizer.embedding_dim)

        # Convert to (B, C, T) for decoder
        z_q = z_q.permute(0, 2, 1).contiguous()

        # Decode
        x_recon = self.decoder(z_q)

        return x_recon
