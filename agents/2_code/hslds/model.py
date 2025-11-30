"""
Hierarchical Switching Linear Dynamical System (HSLDS) for Behavior Discovery
Implements the DiscoveryPipeline universal interface
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class BehaviorPreprocessor:
    """
    Critical preprocessing for behavior invariance
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def ego_centric_alignment(self, keypoints):
        """
        Center on reference point (thorax - assumed to be keypoint 0)
        Input: (Batch, Time, 48) where 48 = 24 keypoints * 2 coords (x, y)
        Output: (Batch, Time, 48) ego-centered
        """
        # Reshape to (Batch, Time, 24, 2)
        batch_size, time_steps, _ = keypoints.shape
        kp_reshaped = keypoints.reshape(batch_size, time_steps, 24, 2)

        # Reference point: thorax (keypoint 0) for each fly
        # Assuming first 12 keypoints are fly 1, last 12 are fly 2
        ref_fly1 = kp_reshaped[:, :, 0:1, :]  # (B, T, 1, 2)
        ref_fly2 = kp_reshaped[:, :, 12:13, :]  # (B, T, 1, 2)

        # Center fly 1 keypoints
        kp_reshaped[:, :, 0:12, :] = kp_reshaped[:, :, 0:12, :] - ref_fly1
        # Center fly 2 keypoints
        kp_reshaped[:, :, 12:24, :] = kp_reshaped[:, :, 12:24, :] - ref_fly2

        # Flatten back to (B, T, 48)
        return kp_reshaped.reshape(batch_size, time_steps, 48)

    def compute_velocities(self, keypoints):
        """
        Add temporal derivatives
        Input: (Batch, Time, 48)
        Output: (Batch, Time, 48) velocities
        """
        # Compute velocity: v_t = x_t - x_{t-1}
        # Pad the first frame with zeros
        velocities = torch.zeros_like(keypoints)
        velocities[:, 1:, :] = keypoints[:, 1:, :] - keypoints[:, :-1, :]
        return velocities

    def engineer_features(self, keypoints):
        """
        Domain-specific features for Drosophila
        Input: (Batch, Time, 48)
        Output: (Batch, Time, N) engineered features
        """
        batch_size, time_steps, _ = keypoints.shape
        kp = keypoints.reshape(batch_size, time_steps, 24, 2)

        features = []

        # 1. Inter-keypoint distances (within each fly)
        # Fly 1: distances between adjacent keypoints
        for i in range(11):
            dist = torch.norm(kp[:, :, i, :] - kp[:, :, i+1, :], dim=-1, keepdim=True)
            features.append(dist)

        # Fly 2: distances between adjacent keypoints
        for i in range(12, 23):
            dist = torch.norm(kp[:, :, i, :] - kp[:, :, i+1, :], dim=-1, keepdim=True)
            features.append(dist)

        # 2. Inter-fly distance (between thorax points)
        inter_fly_dist = torch.norm(kp[:, :, 0, :] - kp[:, :, 12, :], dim=-1, keepdim=True)
        features.append(inter_fly_dist)

        # 3. Heading angles (using first two keypoints as body axis)
        # Fly 1 heading
        heading1 = torch.atan2(kp[:, :, 1, 1] - kp[:, :, 0, 1],
                               kp[:, :, 1, 0] - kp[:, :, 0, 0]).unsqueeze(-1)
        features.append(heading1)

        # Fly 2 heading
        heading2 = torch.atan2(kp[:, :, 13, 1] - kp[:, :, 12, 1],
                               kp[:, :, 13, 0] - kp[:, :, 12, 0]).unsqueeze(-1)
        features.append(heading2)

        # Concatenate all features: (B, T, 25) = 11 + 11 + 1 + 1 + 1
        return torch.cat(features, dim=-1)

    def normalize(self, features):
        """
        Z-score normalization
        Input: (Batch, Time, Features)
        Output: (Batch, Time, Features) normalized
        """
        # Compute mean and std across batch and time dimensions
        if self.mean is None:
            # Flatten batch and time for computing statistics
            flat_features = features.reshape(-1, features.shape[-1])
            self.mean = flat_features.mean(dim=0, keepdim=True)
            self.std = flat_features.std(dim=0, keepdim=True) + 1e-8

        # Normalize
        normalized = (features - self.mean) / self.std
        return normalized

    def preprocess(self, raw_trajectories):
        """
        Full preprocessing pipeline
        Input: (Batch, Time, 48) raw keypoints
        Output: (Batch, Time, Features) processed features
        """
        aligned = self.ego_centric_alignment(raw_trajectories)
        velocities = self.compute_velocities(aligned)
        spatial_features = self.engineer_features(aligned)

        # Combine: aligned positions + velocities + engineered features
        # (B, T, 48) + (B, T, 48) + (B, T, 25) = (B, T, 121)
        combined = torch.cat([aligned, velocities, spatial_features], dim=-1)

        # Normalize
        normalized = self.normalize(combined)
        return normalized


class GraphEncoder(nn.Module):
    """
    Graph Neural Network encoder for keypoint structure
    Maps (Batch*Time, 24, 2) -> (Batch*Time, latent_dim)
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # GCN layers
        self.conv1 = GCNConv(2, 16)  # 2D coordinates -> 16D
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, latent_dim)

        # Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def create_edge_index(self, num_keypoints=12):
        """
        Create anatomical connectivity graph for one fly (12 keypoints)
        Assumes chain structure: 0-1-2-...-11
        """
        edges = []
        # Chain connectivity
        for i in range(num_keypoints - 1):
            edges.append([i, i+1])
            edges.append([i+1, i])  # Bidirectional

        # Additional connections (e.g., legs to body)
        # Simplified: just use chain for now
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, keypoints):
        """
        Input: (Batch, 48) keypoints for one timestep
        Output: (Batch, latent_dim)
        """
        batch_size = keypoints.shape[0]
        kp = keypoints.reshape(batch_size, 24, 2)

        # Process each fly separately, then concatenate
        edge_index = self.create_edge_index(12).to(keypoints.device)

        # Fly 1: keypoints 0-11
        fly1_data = []
        for i in range(batch_size):
            fly1_data.append(Data(x=kp[i, 0:12, :], edge_index=edge_index))
        fly1_batch = Batch.from_data_list(fly1_data)

        # Fly 2: keypoints 12-23
        fly2_data = []
        for i in range(batch_size):
            fly2_data.append(Data(x=kp[i, 12:24, :], edge_index=edge_index))
        fly2_batch = Batch.from_data_list(fly2_data)

        # Encode fly 1
        h1 = F.relu(self.conv1(fly1_batch.x, fly1_batch.edge_index))
        h1 = F.relu(self.conv2(h1, fly1_batch.edge_index))
        h1 = self.conv3(h1, fly1_batch.edge_index)
        h1 = global_mean_pool(h1, fly1_batch.batch)  # (Batch, latent_dim)

        # Encode fly 2
        h2 = F.relu(self.conv1(fly2_batch.x, fly2_batch.edge_index))
        h2 = F.relu(self.conv2(h2, fly2_batch.edge_index))
        h2 = self.conv3(h2, fly2_batch.edge_index)
        h2 = global_mean_pool(h2, fly2_batch.batch)  # (Batch, latent_dim)

        # Combine
        combined = h1 + h2  # Simple aggregation
        return self.readout(combined)


class SwitchingPolicy(nn.Module):
    """
    Recurrent policy for discrete state transitions
    P(s_t | s_{t-1}, z_t, context)
    """
    def __init__(self, n_states, latent_dim, hidden_dim=64):
        super().__init__()
        self.n_states = n_states

        # GRU for temporal context
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)

        # State transition network
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim + n_states, 128),
            nn.ReLU(),
            nn.Linear(128, n_states)
        )

        # Sticky prior (bias towards self-transitions)
        self.sticky_bias = nn.Parameter(torch.ones(n_states) * 2.0)

    def forward(self, latent_seq, prev_states=None):
        """
        Input:
            latent_seq: (Batch, Time, latent_dim)
            prev_states: (Batch, Time, n_states) one-hot encoded
        Output: (Batch, Time, n_states) logits
        """
        batch_size, seq_len, _ = latent_seq.shape

        # Get temporal context from GRU
        context, _ = self.gru(latent_seq)  # (B, T, hidden_dim)

        if prev_states is None:
            prev_states = F.one_hot(torch.zeros(batch_size, seq_len, dtype=torch.long,
                                                device=latent_seq.device),
                                   self.n_states).float()

        # Concatenate context and previous state
        combined = torch.cat([context, prev_states], dim=-1)

        # Predict state logits
        logits = self.transition(combined)

        # Add sticky bias (encourage self-transitions)
        logits = logits + prev_states * self.sticky_bias.unsqueeze(0).unsqueeze(0)

        return logits


class DiscreteStateDecoder(nn.Module):
    """
    Decoder that ONLY uses discrete states (no continuous bypass)
    Forces model to encode all information into state sequence
    This is the CRITICAL fix to prevent codebook collapse
    """
    def __init__(self, n_states, output_dim, hidden_dim=128):
        super().__init__()
        self.n_states = n_states
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # State embedding (each state has its own learned embedding)
        self.state_embed = nn.Embedding(n_states, hidden_dim)

        # Temporal decoder (autoregressive over states)
        self.temporal_decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, states):
        """
        Input: states (Batch, Time) - discrete state indices
        Output: (Batch, Time, output_dim) - reconstructed features
        """
        batch_size, seq_len = states.shape

        # Embed states
        state_emb = self.state_embed(states)  # (B, T, hidden_dim)

        # Apply temporal dynamics (GRU captures state transitions)
        temporal_features, _ = self.temporal_decoder(state_emb)  # (B, T, hidden_dim)

        # Project to output
        output = self.output_proj(temporal_features)  # (B, T, output_dim)

        return output


class LinearDynamicsDecoder(nn.Module):
    """
    State-dependent linear dynamics decoder (DEPRECATED - causes collapse)
    Each discrete state has its own A matrix and bias
    KEEPING FOR BACKWARDS COMPATIBILITY BUT NOT USED
    """
    def __init__(self, n_states, latent_dim, output_dim):
        super().__init__()
        self.n_states = n_states
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # State-specific dynamics matrices
        self.A_matrices = nn.Parameter(torch.randn(n_states, latent_dim, latent_dim) * 0.01)
        self.b_vectors = nn.Parameter(torch.zeros(n_states, latent_dim))

        # Emission network (latent -> observation)
        self.emission = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, latent, states):
        """
        Input:
            latent: (Batch, Time, latent_dim)
            states: (Batch, Time) discrete state indices
        Output: (Batch, Time, output_dim)
        """
        batch_size, seq_len, _ = latent.shape

        # Apply state-dependent linear dynamics
        # For each timestep, use the A matrix for that state
        # This is a simplified version; full SLDS would propagate z_t = A_{s_t} z_{t-1}

        # Gather A matrices and b vectors for each state
        A = self.A_matrices[states]  # (B, T, latent_dim, latent_dim)
        b = self.b_vectors[states]   # (B, T, latent_dim)

        # Apply dynamics: z_t' = A_t @ z_t + b_t
        # Using batch matrix multiplication
        latent_transformed = torch.matmul(latent.unsqueeze(-2), A).squeeze(-2) + b

        # Emit observations
        output = self.emission(latent_transformed)
        return output


class DiscoveryPipeline(nn.Module):
    """
    HSLDS implementation of the DiscoveryPipeline interface
    """
    def __init__(self, input_dim=48, latent_dim=32, n_states=12, output_dim=121):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_states = n_states
        self.n_codes = n_states  # For interface compatibility
        self.output_dim = output_dim

        # Components
        self.preprocessor = BehaviorPreprocessor()

        # Temporal convolution for adding temporal context (5 frames = 167ms at 30Hz)
        self.temporal_conv = nn.Conv1d(48, 48, kernel_size=5, padding=2, groups=48)  # Depthwise

        self.graph_encoder = GraphEncoder(latent_dim=latent_dim)
        self.switching_policy = SwitchingPolicy(n_states, latent_dim)

        # CRITICAL FIX: Use DiscreteStateDecoder (no latent bypass)
        self.decoder = DiscreteStateDecoder(n_states, output_dim, hidden_dim=128)

        # For loss tracking
        self.commitment_loss = 0.0

        # For temperature annealing
        self.current_epoch = 0

    def encode_to_latent(self, x):
        """
        Encode input to continuous latent representation with TEMPORAL CONTEXT
        Input: (Batch, Time, Features) - preprocessed features (121D)
        Output: (Batch, Time, latent_dim)
        """
        batch_size, seq_len, feature_dim = x.shape

        # Extract raw keypoint positions (first 48 features) for graph encoder
        # The preprocessed data is [48 positions, 48 velocities, 25 engineered features]
        x_positions = x[:, :, :48]  # (B, T, 48)

        # Add temporal convolution BEFORE graph encoder
        # This gives each frame a receptive field of ±2 frames (150ms at 30Hz)
        x_positions_t = x_positions.permute(0, 2, 1)  # (B, 48, T)
        x_temporal = self.temporal_conv(x_positions_t)  # (B, 48, T)
        x_temporal = x_temporal.permute(0, 2, 1)  # (B, T, 48)

        # Reshape to (Batch*Time, 48) for graph encoder
        x_flat = x_temporal.reshape(batch_size * seq_len, 48)

        # Encode
        latent_flat = self.graph_encoder(x_flat)  # (B*T, latent_dim)

        # Reshape back
        latent = latent_flat.reshape(batch_size, seq_len, self.latent_dim)
        return latent

    def encode(self, x):
        """
        Map input trajectories to discrete state codes
        Input: (Batch, Time, Features)
        Output: (Batch, Time) Integer codes
        """
        # Get continuous latent
        latent = self.encode_to_latent(x)

        # Get state logits from switching policy
        state_logits = self.switching_policy(latent)

        # Greedy decoding (could use Viterbi for better sequence decoding)
        states = torch.argmax(state_logits, dim=-1)

        return states

    def decode(self, codes):
        """
        Map discrete codes back to trajectory space
        Input: (Batch, Time) Integer codes
        Output: (Batch, Time, Features)
        """
        # Decode directly from states (no latent needed with new decoder)
        reconstructed = self.decoder(codes)
        return reconstructed

    def forward(self, x):
        """
        Standard training forward pass
        Returns: reconstructed, codes
        """
        # Encode to continuous latent
        latent = self.encode_to_latent(x)

        # Get state distribution
        state_logits = self.switching_policy(latent)

        # Sample states (Gumbel-Softmax for differentiability during training)
        # Anneal temperature: start hot (explore), end cold (commit)
        epoch_fraction = min(self.current_epoch / 80, 1.0)  # Anneal over 80 epochs (SLOWED)
        temperature = max(1.0 - 0.9 * epoch_fraction, 0.1)  # 1.0 → 0.1
        state_probs = F.gumbel_softmax(state_logits, tau=temperature, hard=True)
        states_hard = torch.argmax(state_probs, dim=-1)

        # CRITICAL FIX: Decode from states ONLY (no latent bypass)
        reconstructed = self.decoder(states_hard)

        # Compute commitment loss (encourage latent to commit to states)
        # This is analogous to VQ-VAE commitment loss
        with torch.no_grad():
            optimal_states = torch.argmax(state_logits, dim=-1)
        self.commitment_loss = F.cross_entropy(
            state_logits.reshape(-1, self.n_states),
            optimal_states.reshape(-1)
        )

        return reconstructed, states_hard

    def generate(self, n_samples, length):
        """
        Autoregressive generation of new behavior sequences
        Input: n_samples, length
        Output: (n_samples, length, Features) Synthetic trajectories
        """
        device = next(self.parameters()).device

        # Initialize with random latent state
        latent = torch.randn(n_samples, 1, self.latent_dim, device=device) * 0.1

        # Initialize with random state
        current_state = torch.randint(0, self.n_states, (n_samples,), device=device)

        generated_frames = []
        state_sequence = []

        for t in range(length):
            # Get state transition probabilities
            state_logits = self.switching_policy(latent)[:, -1, :]  # (n_samples, n_states)

            # Sample next state
            state_probs = F.softmax(state_logits, dim=-1)
            current_state = torch.multinomial(state_probs, 1).squeeze(-1)
            state_sequence.append(current_state)

            # Decode current state
            frame = self.decoder(latent[:, -1:, :], current_state.unsqueeze(-1))
            generated_frames.append(frame)

            # Update latent (simple random walk for now)
            # In full implementation, would use learned dynamics
            latent = torch.cat([latent,
                               latent[:, -1:, :] + torch.randn_like(latent[:, -1:, :]) * 0.1],
                              dim=1)

        # Stack frames
        generated = torch.cat(generated_frames, dim=1)  # (n_samples, length, output_dim)

        return generated
