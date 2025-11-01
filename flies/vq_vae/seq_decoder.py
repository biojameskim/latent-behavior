import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import ResidualBlock1D

class SequenceDecoder(nn.Module):
    """
    Decoder that takes quantized latent representation z_q and reconstructs the input x.

    Args:
        embedding_dim (int): Dimension of the input embedding (from quantizer).
        hidden_dims (list of int): List of hidden layer dimensions (in reverse order from encoder).
        output_dim (int): Dimension of the output features (should match encoder input_dim).
        output_length (int): Target output sequence length (to match encoder input).
        num_residual_blocks (int): Number of residual blocks to add after each upsampling layer.

    Returns:
        torch.Tensor: Reconstructed sequence.

    Note:
        Uses Upsample + Conv instead of ConvTranspose1d for cleaner upsampling.
        This avoids checkerboard artifacts and output_padding complexity.
    """

    def __init__(self, embedding_dim, hidden_dims, output_dim, output_length, num_residual_blocks=2, strides=None):
        super().__init__()

        self.output_length = output_length
        self.hidden_dims = hidden_dims

        # Handle strides: can be single int or list of ints (one per layer)
        # Decoder strides should be reversed encoder strides
        if strides is None:
            strides = [2] * len(hidden_dims)
        elif isinstance(strides, int):
            strides = [strides] * len(hidden_dims)
        else:
            assert len(strides) == len(hidden_dims), "strides must match number of hidden layers"

        # Initial projection
        self.initial_conv = nn.Conv1d(embedding_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1)

        # Build upsampling blocks using Upsample + Conv (cleaner than ConvTranspose1d)
        self.upsample_blocks = nn.ModuleList()
        prev_dim = hidden_dims[0]

        for i, (hidden_dim, stride) in enumerate(zip(hidden_dims[1:], strides[:-1]), 1):
            block = nn.ModuleDict({
                'residuals': nn.ModuleList([ResidualBlock1D(prev_dim) for _ in range(num_residual_blocks)]),
                'upsample': nn.Sequential(
                    nn.Upsample(scale_factor=stride, mode='nearest'),
                    nn.Conv1d(prev_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
                ),
                'relu': nn.ReLU(),
                'bn': nn.BatchNorm1d(hidden_dim)
            })
            self.upsample_blocks.append(block)
            prev_dim = hidden_dim

        # Final residual blocks and upsampling
        self.final_residuals = nn.ModuleList([ResidualBlock1D(prev_dim) for _ in range(num_residual_blocks)])
        final_stride = strides[-1]
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=final_stride, mode='nearest'),
            nn.Conv1d(prev_dim, output_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z_q):
        """
        Forward pass through the decoder.

        Args:
            z_q (torch.Tensor): Quantized latent tensor of shape (batch_size, embedding_dim, reduced_sequence_length)

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, output_dim, output_length)
        """
        # Initial projection
        x = self.initial_conv(z_q)

        # Upsampling blocks
        for block in self.upsample_blocks:
            # Apply residual blocks
            for residual in block['residuals']:
                x = residual(x)
            # Upsample
            x = block['upsample'](x)
            x = block['relu'](x)
            x = block['bn'](x)

        # Final residual blocks
        for residual in self.final_residuals:
            x = residual(x)

        # Final upsampling
        x_recon = self.final_upsample(x)

        # Verify output length matches (should be exact with correct output_padding)
        if x_recon.shape[2] != self.output_length:
            # This should rarely happen now, but crop/pad as safety
            if x_recon.shape[2] > self.output_length:
                x_recon = x_recon[:, :, :self.output_length]
            else:
                # If still too short, we have a bug - raise error instead of padding
                raise ValueError(
                    f"Decoder output length {x_recon.shape[2]} does not match "
                    f"target {self.output_length}. Check output_padding settings."
                )

        return x_recon
