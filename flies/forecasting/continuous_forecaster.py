"""
Continuous-to-continuous behavior forecasting models.

These models operate directly on continuous keypoint sequences without discretization,
learning to predict future behavior from past observations.

Approach: Train model on continuous data → optionally extract discrete tokens post-hoc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class TransformerForecaster(nn.Module):
    """
    Transformer-based forecasting on continuous keypoints.

    Architecture:
        Past keypoints (batch, 48, context_len) → Embedding
        → Transformer encoder → Forecast future (batch, 48, forecast_len)

    This learns continuous dynamics directly without tokenization.
    """

    def __init__(
        self,
        input_dim: int = 48,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        context_length: int = 75,  # Past frames to condition on
        forecast_length: int = 75,  # Future frames to predict
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.context_length = context_length
        self.forecast_length = forecast_length

        # Input projection: (batch, 48, time) → (batch, time, d_model)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=context_length + forecast_length)

        # Transformer encoder (processes context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: (batch, time, d_model) → (batch, time, 48)
        self.output_proj = nn.Linear(d_model, input_dim)

        # Learnable queries for forecast timesteps
        self.forecast_queries = nn.Parameter(torch.randn(forecast_length, d_model))

    def forward(self, x_context: torch.Tensor, x_future: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict future keypoints from past context.

        Args:
            x_context: Past keypoints (batch, input_dim, context_len)
            x_future: Optional future keypoints for teacher forcing (batch, input_dim, forecast_len)

        Returns:
            x_pred: Predicted future keypoints (batch, input_dim, forecast_len)
        """
        batch_size = x_context.size(0)

        # Transpose to (batch, time, features)
        x_context = x_context.transpose(1, 2)  # (batch, context_len, 48)

        # Project to model dimension
        context_emb = self.input_proj(x_context)  # (batch, context_len, d_model)

        # Create forecast queries (batch, forecast_len, d_model)
        forecast_queries = self.forecast_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate context + forecast queries
        full_seq = torch.cat([context_emb, forecast_queries], dim=1)  # (batch, context_len+forecast_len, d_model)

        # Add positional encoding
        full_seq = self.pos_encoder(full_seq)

        # Create causal mask: each position can only attend to past
        seq_len = full_seq.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x_context.device)

        # Apply transformer
        transformer_out = self.transformer(full_seq, mask=causal_mask)  # (batch, seq_len, d_model)

        # Extract forecast portion and project to output
        forecast_out = transformer_out[:, -self.forecast_length:, :]  # (batch, forecast_len, d_model)
        x_pred = self.output_proj(forecast_out)  # (batch, forecast_len, 48)

        # Transpose back to (batch, 48, forecast_len)
        x_pred = x_pred.transpose(1, 2)

        return x_pred

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute forecasting loss by predicting second half from first half.

        Args:
            x: Full sequence (batch, 48, context_length + forecast_length)

        Returns:
            loss: MSE loss
            loss_dict: Dictionary of loss components
        """
        # Split into context and future
        x_context = x[:, :, :self.context_length]
        x_future = x[:, :, self.context_length:]

        # Predict future
        x_pred = self.forward(x_context)

        # MSE loss
        loss = F.mse_loss(x_pred, x_future)

        loss_dict = {
            'loss': loss.item(),
            'forecast_mse': loss.item(),
        }

        return loss, loss_dict


class LSTMForecaster(nn.Module):
    """
    LSTM-based forecasting on continuous keypoints.

    Simpler baseline that uses recurrent architecture.
    """

    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        context_length: int = 75,
        forecast_length: int = 75,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_length = context_length
        self.forecast_length = forecast_length

        # LSTM encoder (processes context)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_context: torch.Tensor) -> torch.Tensor:
        """
        Predict future keypoints autoregressively.

        Args:
            x_context: Past keypoints (batch, input_dim, context_len)

        Returns:
            x_pred: Predicted future keypoints (batch, input_dim, forecast_len)
        """
        batch_size = x_context.size(0)

        # Transpose to (batch, time, features)
        x_context = x_context.transpose(1, 2)  # (batch, context_len, 48)

        # Encode context
        lstm_out, (h_n, c_n) = self.lstm(x_context)

        # Autoregressive generation
        predictions = []
        current_input = x_context[:, -1:, :]  # Last context frame

        for _ in range(self.forecast_length):
            # Predict next frame
            lstm_out, (h_n, c_n) = self.lstm(current_input, (h_n, c_n))
            next_frame = self.fc_out(lstm_out)  # (batch, 1, 48)

            predictions.append(next_frame)
            current_input = next_frame

        # Concatenate predictions
        x_pred = torch.cat(predictions, dim=1)  # (batch, forecast_len, 48)

        # Transpose back
        x_pred = x_pred.transpose(1, 2)  # (batch, 48, forecast_len)

        return x_pred

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Same as TransformerForecaster."""
        x_context = x[:, :, :self.context_length]
        x_future = x[:, :, self.context_length:]

        x_pred = self.forward(x_context)
        loss = F.mse_loss(x_pred, x_future)

        loss_dict = {
            'loss': loss.item(),
            'forecast_mse': loss.item(),
        }

        return loss, loss_dict


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
