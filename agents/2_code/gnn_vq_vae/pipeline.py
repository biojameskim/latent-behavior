import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

# ==========================================
# 1. Preprocessing Module
# ==========================================

class BehaviorPreprocessor:
    """
    Critical preprocessing for behavior invariance
    """
    def __init__(self, keypoint_names=None):
        self.keypoint_names = keypoint_names
        # Indices for specific body parts to use as reference (e.g., thorax/shoulders)
        # Assuming standard ordering if names not provided, but better to be explicit.
        # For now, we'll assume the first few features are body center or similar if available,
        # otherwise we calculate centroid.
        pass

    def ego_centric_alignment(self, trajectories):
        """
        Center on reference point (centroid of current frame) and align orientation.
        Input: (Batch, Time, Features)
        Assumes Features are [x1, y1, x2, y2, ...]
        """
        B, T, F_dim = trajectories.shape
        # Reshape to (B, T, K, 2)
        keypoints = trajectories.reshape(B, T, -1, 2)
        
        # 1. Centering: Subtract mean of each frame (centroid)
        centroid = keypoints.mean(dim=2, keepdim=True) # (B, T, 1, 2)
        centered = keypoints - centroid
        
        # 2. Orientation Alignment (Optional but recommended for "ego-centric")
        # For simple centering, we return centered flattened.
        # To do full ego-centric, we'd need to rotate so head-tail axis is up.
        # Given "Discovery Mode" and lack of strict body part mapping guarantees in this snippet,
        # we will stick to translation invariance (centering) for now.
        
        return centered.reshape(B, T, F_dim)

    def compute_velocities(self, trajectories):
        """Add temporal derivatives"""
        # trajectories: (B, T, F)
        # Pad first frame to keep length T
        velocities = torch.diff(trajectories, dim=1, prepend=trajectories[:, :1, :])
        return velocities

    def engineer_features(self, trajectories):
        """Domain-specific features: Pairwise distances"""
        # trajectories: (B, T, F) -> (B, T, K, 2)
        B, T, F_dim = trajectories.shape
        K = F_dim // 2
        kps = trajectories.reshape(B, T, K, 2)
        
        # Compute pairwise distances for a subset of keypoints to avoid O(K^2) explosion if K is large
        # For K=24, K^2 is fine (~576).
        # We'll compute distances from centroid (already implicitly encoded in centered coords)
        # and maybe limb lengths.
        # For simplicity in this universal pipeline, we pass through raw aligned coords + velocities.
        # If we wanted explicit distances:
        # dists = torch.norm(kps[:, :, :, None, :] - kps[:, :, None, :, :], dim=-1)
        # return dists.reshape(B, T, -1)
        
        return trajectories # Returning aligned trajectories directly for GNN to handle

    def normalize(self, features):
        """Scale normalization (Z-score)"""
        mean = features.mean(dim=(0, 1), keepdim=True)
        std = features.std(dim=(0, 1), keepdim=True) + 1e-6
        return (features - mean) / std

    def preprocess(self, raw_trajectories):
        """Full pipeline"""
        # Convert to tensor if numpy
        if isinstance(raw_trajectories, np.ndarray):
            raw_trajectories = torch.from_numpy(raw_trajectories).float()
            
        aligned = self.ego_centric_alignment(raw_trajectories)
        velocities = self.compute_velocities(aligned)
        
        # Concatenate position and velocity
        # (B, T, F) + (B, T, F) -> (B, T, 2F)
        combined = torch.cat([aligned, velocities], dim=-1)
        
        # engineered = self.engineer_features(aligned) # GNN handles geometry
        
        normalized = self.normalize(combined)
        return normalized

# ==========================================
# 2. Universal Model Architecture (GNN-VQ-VAE)
# ==========================================

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_embedding', torch.randn(self._num_embeddings, self._embedding_dim))
        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embeddings))
        self.register_buffer('_ema_w', torch.randn(self._num_embeddings, self._embedding_dim))
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # inputs: (B, T, D)
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self._embedding).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            # Laplace smoothing of the cluster size
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the codebook vectors
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) /
                (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw
            
            self._embedding = self._ema_w / self._ema_cluster_size.unsqueeze(1)
            
            # RESTART DEAD CODES
            # Identify codes with very low usage (approx < 1% of uniform expectation)
            n_active = self._ema_cluster_size.sum()
            usage = self._ema_cluster_size / (n_active + 1e-7)
            threshold = (1.0 / self._num_embeddings) * 0.1 
            dead_codes = torch.where(usage < threshold)[0]
            
            if len(dead_codes) > 0:
                # Pick random inputs from current batch to reset them
                flat_input = inputs.view(-1, self._embedding_dim)
                n_dead = len(dead_codes)
                # Ensure we don't sample more than available
                n_sample = min(n_dead, flat_input.size(0))
                if n_sample > 0:
                    rand_indices = torch.randint(0, flat_input.size(0), (n_sample,), device=inputs.device)
                    rand_inputs = flat_input[rand_indices].detach()
                    
                    # Update embedding and EMA buffers
                    # We only update the first n_sample dead codes if batch is small
                    target_codes = dead_codes[:n_sample]
                    
                    with torch.no_grad():
                        self._embedding[target_codes] = rand_inputs
                        # Reset EMA stats for these codes to average
                        self._ema_w[target_codes] = rand_inputs * self._ema_cluster_size[target_codes].unsqueeze(1)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, commitment_loss, encoding_indices.view(input_shape[0], input_shape[1])

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = input_dim // num_nodes
        
        # Simple GNN: MLP per node + aggregation
        # In a real GNN we'd use adjacency, but for fully connected or implicit, we can use attention or shared MLP.
        # Here we use a shared MLP per node (PointNet style) followed by temporal convs.
        
        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temporal Encoder (TCN)
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim * num_nodes, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, latent_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        # Reshape to (B, T, Nodes, NodeFeats)
        x_nodes = x.view(B, T, self.num_nodes, -1)
        
        # Apply Node MLP
        x_emb = self.node_mlp(x_nodes) # (B, T, Nodes, Hidden)
        
        # Flatten nodes for TCN
        x_flat = x_emb.view(B, T, -1).permute(0, 2, 1) # (B, C, T)
        
        z = self.tcn(x_flat).permute(0, 2, 1) # (B, T, Latent)
        return z

class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        
        self.tcn = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * num_nodes, kernel_size=3, padding=1)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim // num_nodes)
        )

    def forward(self, z):
        # z: (B, T, Latent)
        B, T, _ = z.shape
        
        z_flat = z.permute(0, 2, 1) # (B, C, T)
        x_nodes_flat = self.tcn(z_flat).permute(0, 2, 1) # (B, T, Hidden*Nodes)
        
        x_nodes = x_nodes_flat.view(B, T, self.num_nodes, -1)
        out = self.node_mlp(x_nodes) # (B, T, Nodes, OutFeats)
        
        return out.view(B, T, -1)

class TransformerDynamics(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, embedding_dim)) # Max len 500
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, indices):
        # indices: (B, T)
        x = self.embedding(indices) + self.pos_encoder[:, :indices.size(1), :]
        
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(indices.size(1)).to(indices.device)
        
        out = self.transformer(x, mask=mask, is_causal=True)
        logits = self.fc_out(out)
        return logits

class DiscoveryPipeline(nn.Module):
    """
    Standardized wrapper for GNN-VQ-VAE with Transformer Dynamics.
    """
    def __init__(self, input_dim=96, num_nodes=24, hidden_dim=64, latent_dim=32, num_embeddings=64, embedding_dim=32):
        super().__init__()
        self.n_codes = num_embeddings
        
        # 1. Encoder (GNN + TCN)
        self.encoder = GNNEncoder(input_dim, hidden_dim, latent_dim, num_nodes)
        
        # 2. Bottleneck (VQ)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.pre_vq = nn.Linear(latent_dim, embedding_dim)
        self.post_vq = nn.Linear(embedding_dim, latent_dim)
        
        # 3. Decoder (TCN + GNN)
        self.decoder = GNNDecoder(latent_dim, hidden_dim, input_dim, num_nodes)
        
        # 4. Dynamics (Transformer Prior)
        # We train this jointly or separately. For simplicity, we include it and train jointly with auxiliary loss.
        self.dynamics = TransformerDynamics(num_embeddings, embedding_dim)
        
        self.commitment_loss = 0.0

    def encode(self, x):
        """
        Map input trajectories to discrete tokens/codes.
        Input: (Batch, Time, Features)
        Output: (Batch, Time) -> Integer codes
        """
        z = self.encoder(x)
        z = self.pre_vq(z)
        _, _, indices = self.vq(z)
        return indices

    def decode(self, codes):
        """
        Map discrete tokens back to trajectory space.
        Input: (Batch, Time) -> Integer codes
        Output: (Batch, Time, Features)
        """
        # Get embeddings from indices
        z_q = F.embedding(codes, self.vq._embedding)
        z = self.post_vq(z_q)
        return self.decoder(z)

    def forward(self, x):
        """
        Standard training forward pass.
        Returns: reconstructed, codes
        """
        # 1. Encode
        z = self.encoder(x)
        z = self.pre_vq(z)
        
        # 2. Quantize
        z_q, commit_loss, indices = self.vq(z)
        self.commitment_loss = commit_loss
        
        # 3. Decode
        z_dec = self.post_vq(z_q)
        recon = self.decoder(z_dec)
        
        # 4. Dynamics (Predict next token)
        # Input: indices[:, :-1], Target: indices[:, 1:]
        # We compute logits here to be used in loss if needed, 
        # but the main discovery_loss signature doesn't take logits.
        # We will store them to be accessible if we extend the loss, 
        # or we can add a separate training step for dynamics.
        # For the "DiscoveryPipeline" interface, we focus on recon + codes.
        self.dynamics_logits = self.dynamics(indices)
        
        return recon, indices

    def generate(self, n_samples, length):
        """
        Autoregressive generation of new behavior.
        Input: n_samples, length
        Output: (n_samples, length, Features) -> Synthetic trajectories
        """
        device = next(self.parameters()).device
        
        # Start with random token or specific start token
        # Assuming 0 is not special, just random
        curr_seq = torch.randint(0, self.n_codes, (n_samples, 1)).to(device)
        
        with torch.no_grad():
            for _ in range(length - 1):
                logits = self.dynamics(curr_seq)
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                curr_seq = torch.cat([curr_seq, next_token], dim=1)
        
        # Decode
        return self.decode(curr_seq)

# ==========================================
# 3. Discovery-Specific Loss Function
# ==========================================

def discovery_loss(original, reconstructed, codes, model, alpha=1.0, beta=0.25, gamma=0.1, delta=0.5):
    """
    Multi-objective loss for unsupervised discovery.
    """
    # 1. Reconstruction
    recon_loss = F.mse_loss(reconstructed, original)
    
    # 2. Commitment
    if hasattr(model, 'commitment_loss'):
        commit_loss = model.commitment_loss
    else:
        commit_loss = torch.tensor(0.0).to(original.device)
    
    # 3. Codebook utilization
    code_probs = torch.bincount(codes.flatten(), minlength=model.n_codes).float() / codes.numel()
    code_probs = code_probs + 1e-10
    codebook_entropy = -torch.sum(code_probs * torch.log(code_probs))
    target_entropy = np.log(model.n_codes)
    codebook_loss = target_entropy - codebook_entropy
    
    # 4. Temporal coherence
    code_changes = (codes[:, 1:] != codes[:, :-1]).float().mean()
    temporal_loss = code_changes
    
    # Optional: Dynamics Loss (Cross Entropy) if model has dynamics_logits
    dynamics_loss = torch.tensor(0.0).to(original.device)
    if hasattr(model, 'dynamics_logits'):
        # Predict next token
        logits = model.dynamics_logits[:, :-1, :].reshape(-1, model.n_codes)
        targets = codes[:, 1:].reshape(-1)
        dynamics_loss = F.cross_entropy(logits, targets)
    
    # Combined
    total_loss = (
        alpha * recon_loss +
        beta * commit_loss +
        gamma * codebook_loss +
        delta * temporal_loss +
        0.5 * dynamics_loss # Add dynamics weight
    )
    
    return total_loss, {
        'total': total_loss.item(),
        'reconstruction': recon_loss.item(),
        'commitment': commit_loss.item(),
        'codebook': codebook_loss.item(),
        'temporal': temporal_loss.item(),
        'dynamics': dynamics_loss.item()
    }

# ==========================================
# 4. Failure Mode Detection
# ==========================================

class FailureModeDetector:
    """Detect and warn about common failure modes"""
    
    def check_all(self, codes, original, reconstructed, n_codes):
        """Run all checks"""
        issues = []
        if self.check_codebook_collapse(codes, n_codes): issues.append("CODEBOOK_COLLAPSE")
        if self.check_temporal_flickering(codes): issues.append("TEMPORAL_FLICKERING")
        if self.check_degenerate_segmentation(codes): issues.append("DEGENERATE_SEGMENTATION")
        if self.check_poor_reconstruction(original, reconstructed): issues.append("POOR_RECONSTRUCTION")
        return issues
    
    def check_codebook_collapse(self, codes, n_codes):
        unique = len(np.unique(codes.cpu().numpy()))
        return unique < n_codes * 0.10 # Strict threshold
    
    def check_temporal_flickering(self, codes):
        change_rate = (codes[:, 1:] != codes[:, :-1]).float().mean()
        return change_rate > 0.5
    
    def check_degenerate_segmentation(self, codes):
        freqs = np.bincount(codes.flatten().cpu().numpy()) / codes.numel()
        return freqs.max() > 0.9 # One code dominates 90% of data
    
    def check_poor_reconstruction(self, original, reconstructed):
        var_orig = torch.var(original)
        var_recon = torch.var(reconstructed)
        return var_recon < (var_orig * 0.1) # Posterior collapse to mean

# ==========================================
# 5. Training Loop
# ==========================================

def train_pipeline(model, data, epochs=50, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    history = []
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        metrics_sum = {}
        
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            
            recon, codes = model(x)
            loss, metrics = discovery_loss(x, recon, codes, model)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items()}
        history.append(avg_metrics)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total={avg_metrics['total']:.4f}, Recon={avg_metrics['reconstruction']:.4f}, Codebook={avg_metrics['codebook']:.4f}")
            
    return model, history
