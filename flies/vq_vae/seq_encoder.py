import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .residual import ResidualBlock1D

class SequenceEncoder(nn.Module):
    """
    Encoder that takes input x (temporal sequences) and maps to latent space z.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dims (list of int): List of hidden layer dimensions.
        embedding_dim (int): Dimension of the output embedding.
        num_residual_blocks (int): Number of residual blocks to add after each downsampling layer.

    Returns:
        torch.Tensor: Encoded latent representation z.
    """

    def __init__(self, input_dim, hidden_dims, embedding_dim, num_residual_blocks=2, strides=None):
        super().__init__()

        # Handle strides: can be single int or list of ints (one per layer)
        if strides is None:
            strides = [2] * len(hidden_dims)
        elif isinstance(strides, int):
            strides = [strides] * len(hidden_dims)
        else:
            assert len(strides) == len(hidden_dims), "strides must match number of hidden layers"

        layers = []
        prev_dim = input_dim
        for hidden_dim, stride in zip(hidden_dims, strides):
            # Downsampling conv block with adaptive kernel/padding
            # Adapts kernel_size and padding based on stride to handle length properly
            kernel_size = stride * 2
            padding = stride // 2 if stride % 2 == 0 else stride // 2 + 1

            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            # Add residual blocks after downsampling
            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock1D(hidden_dim))
            prev_dim = hidden_dim

        # Final projection to embedding dimension
        layers.append(nn.Conv1d(prev_dim, embedding_dim, kernel_size=3, stride=1, padding=1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim, sequence_length)
        
        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, embedding_dim, reduced_sequence_length)
        """
        z = self.encoder(x)
        return z