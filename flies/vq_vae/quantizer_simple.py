"""
Alternative quantizer implementation with just initialization fix (no normalization).

This is closer to the original VQ-VAE paper and might be sufficient
if the initialization is correct.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerSimple(nn.Module):
    """
    Vector Quantizer with ONLY the initialization fix.
    No additional normalization layers.

    This is closer to the original VQ-VAE paper approach.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook with reasonable scale
        # FIXED: Use normal(0, 1) instead of uniform(-1/K, 1/K)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_(0, 1.0)

        # Alternative: could also use Xavier/Kaiming initialization
        # nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, z):
        """Same as original quantizer - no changes to forward pass."""
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight**2, dim=1) -
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)

        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 2, 1).contiguous()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        batch_size = z.shape[0]
        time_steps = z.shape[1]
        encoding_indices = encoding_indices.view(batch_size, time_steps)

        return z_q, loss, perplexity, encodings, encoding_indices
