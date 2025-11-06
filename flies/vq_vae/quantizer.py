import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer layer that discretizes continuous representations.

    This is the core of VQ-VAE - it learns a discrete codebook of embeddings
    and maps encoder outputs to the nearest codebook entry.

    Args:
        num_embeddings (int): Size of the codebook (number of discrete codes).
        embedding_dim (int): Dimension of each codebook entry.
        commitment_cost (float): Weight for the commitment loss (beta in the paper).
                                Encourages encoder output to stay close to chosen codes.

    The codebook learns behavior "syllables" - each entry represents a prototypical
    temporal pattern that can be reused across different sequences.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook with uniform distribution
        # Use a larger scale to better match typical encoder outputs after normalization
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # After GroupNorm, encoder outputs typically have std ~ 1, so init codebook similarly
        self.embedding.weight.data.normal_(0, 1.0)

    def forward(self, z):
        """
        Forward pass: quantize continuous encoder outputs.

        Args:
            z (torch.Tensor): Encoder output of shape (batch_size, embedding_dim, time_steps)

        Returns:
            z_q (torch.Tensor): Quantized embeddings with same shape as z
            loss (torch.Tensor): VQ loss (codebook + commitment loss)
            perplexity (torch.Tensor): Measure of codebook usage
            encodings (torch.Tensor): One-hot encoding of which codes were used
            encoding_indices (torch.Tensor): Indices of nearest codebook entries
        """
        # Convert from (B, C, T) to (B, T, C) for easier processing
        z = z.permute(0, 2, 1).contiguous()  # (B, T, C)

        # Flatten to (B*T, C)
        z_flattened = z.view(-1, self.embedding_dim)

        # Calculate distances from z to each codebook entry
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight**2, dim=1) -
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )  # (B*T, num_embeddings)

        # Find nearest codebook entry for each encoder output
        encoding_indices = torch.argmin(distances, dim=1)  # (B*T,)

        # Convert to one-hot encodings
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # (B*T, num_embeddings)

        # Quantize: replace z with nearest codebook entry
        z_q = torch.matmul(encodings, self.embedding.weight)  # (B*T, C)
        z_q = z_q.view(z.shape)  # (B, T, C)

        # VQ Loss has two components:
        # 1. Codebook loss: moves codebook entries closer to encoder outputs
        #    (only gradients to codebook, not encoder)
        #    MSE(z_q, z.detach()) → gradients only flow to z_q (codebook embeddings)
        codebook_loss = F.mse_loss(z_q, z.detach())

        # 2. Commitment loss: encourages encoder to commit to codebook entries
        #    (only gradients to encoder, not codebook)
        #    MSE(z_q.detach(), z) → gradients only flow to z (encoder outputs)
        commitment_loss = F.mse_loss(z_q.detach(), z)

        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z
        # This allows gradients to flow through the quantization operation
        z_q = z + (z_q - z).detach()

        # Convert back to (B, C, T)
        z_q = z_q.permute(0, 2, 1).contiguous()

        # Calculate perplexity (measure of how many codes are being used)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Reshape encoding_indices back to (B, T)
        batch_size = z.shape[0]
        time_steps = z.shape[1]
        encoding_indices = encoding_indices.view(batch_size, time_steps)

        return z_q, loss, perplexity, encodings, encoding_indices
