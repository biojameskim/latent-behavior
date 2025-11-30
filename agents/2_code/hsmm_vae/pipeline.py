import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader, TensorDataset


class BehaviorPreprocessor:
    """
    Preprocess fly keypoint trajectories for invariance and richer signals.
    Operates on raw trajectories shaped (batch, time, features) with 48 features (24 keypoints x/y).
    """

    def __init__(self, anchor_idx: int = 0, eps: float = 1e-6):
        self.anchor_idx = anchor_idx
        self.eps = eps
        self.mean = None
        self.std = None

    def ego_centric_alignment(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Center on reference point (thorax proxy) to remove global translation.
        Expects keypoints reshaped to (batch, time, n_kp, 2).
        """
        anchor = keypoints[..., self.anchor_idx, :]  # (B, T, 2)
        centered = keypoints - anchor[..., None, :]
        return centered

    def compute_velocities(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute velocity and acceleration; pad with zeros to preserve length."""
        vel = np.diff(keypoints, axis=1, prepend=keypoints[:, :1])
        acc = np.diff(vel, axis=1, prepend=vel[:, :1])
        return vel, acc

    def engineer_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Derive pairwise distances and angles from anchor to each keypoint.
        Returns feature array shaped (B, T, base + velocity + acceleration + distances + angles).
        """
        B, T, K, _ = keypoints.shape
        vel, acc = self.compute_velocities(keypoints)

        # Pairwise distances between all keypoints (symmetric upper triangle)
        diffs = keypoints[:, :, :, None, :] - keypoints[:, :, None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices(K, k=1)
        pairwise = dists[:, :, iu[0], iu[1]]  # (B, T, K*(K-1)/2)

        # Angles from anchor to each keypoint (relative orientation)
        anchor_vectors = keypoints - keypoints[:, :, self.anchor_idx : self.anchor_idx + 1, :]
        angles = np.arctan2(anchor_vectors[..., 1], anchor_vectors[..., 0])  # (B, T, K)

        base = keypoints.reshape(B, T, -1)
        vel_flat = vel.reshape(B, T, -1)
        acc_flat = acc.reshape(B, T, -1)
        feats = np.concatenate([base, vel_flat, acc_flat, pairwise, angles], axis=-1)
        return feats.astype(np.float32)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Fit z-score on first call; reuse stats for consistency."""
        if self.mean is None or self.std is None:
            self.mean = features.mean(axis=(0, 1), keepdims=True)
            self.std = features.std(axis=(0, 1), keepdims=True) + self.eps
        return (features - self.mean) / self.std

    def preprocess(self, raw_trajectories: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline."""
        B, T, F = raw_trajectories.shape
        keypoints = raw_trajectories.reshape(B, T, -1, 2)
        aligned = self.ego_centric_alignment(keypoints)
        features = self.engineer_features(aligned)
        normalized = self.normalize(features)
        return normalized


class Encoder(nn.Module):
    """Temporal encoder with local convolutions plus bidirectional GRU for context."""

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, latent_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.proj(h)


class Decoder(nn.Module):
    """GRU decoder conditioned on quantized embeddings."""

    def __init__(self, code_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gru = nn.GRU(code_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(z_q)
        return self.head(h)


class DiscoveryPipeline(nn.Module):
    """
    Hierarchical Semi-Markov VAE with discrete codes and duration-aware prior.
    Adheres to the universal DiscoveryPipeline interface.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        code_dim: int = 64,
        n_codes: int = 32,
        max_duration: int = 45,
        gumbel_temp: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.n_codes = n_codes
        self.max_duration = max_duration
        self.gumbel_temp = gumbel_temp

        self.encoder = Encoder(feature_dim, hidden_dim, code_dim).to(device)
        self.codebook = nn.Embedding(n_codes, code_dim)
        self.decoder = Decoder(code_dim, hidden_dim, feature_dim).to(device)

        # Semi-Markov dynamics: transition matrix and per-state duration logits
        self.transition_logits = nn.Parameter(torch.zeros(n_codes, n_codes))
        self.duration_logits = nn.Parameter(torch.zeros(n_codes, max_duration))

        self.commitment_loss = torch.tensor(0.0, device=device)

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vector quantization with straight-through estimator.
        Returns codes (B, T) and quantized vectors (B, T, D).
        """
        # Compute distances to codebook
        codebook = self.codebook.weight  # (K, D)
        z_flat = z_e.reshape(-1, z_e.size(-1))
        dist = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flat, codebook.t())
            + torch.sum(codebook**2, dim=1)
        )  # (B*T, K)
        codes = torch.argmin(dist, dim=1)
        z_q = codebook[codes].reshape(z_e.shape)

        # Commitment and embedding losses (VQ-VAE style)
        commit_loss = F.mse_loss(z_e.detach(), z_q)
        embed_loss = F.mse_loss(z_e, z_q.detach())
        self.commitment_loss = commit_loss + embed_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        codes = codes.reshape(z_e.shape[0], z_e.shape[1])
        return codes, z_q_st

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        codes, _ = self.quantize(z_e)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = self.codebook(codes)
        recon = self.decoder(z_q)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        codes, z_q = self.quantize(z_e)
        recon = self.decoder(z_q)
        return recon, codes

    def _sample_durations(self, curr_state: torch.Tensor, batch: int) -> torch.Tensor:
        durations = []
        duration_probs = F.softmax(self.duration_logits, dim=-1)
        for b in range(batch):
            probs = duration_probs[curr_state[b]].detach().cpu().numpy()
            dur = np.random.choice(self.max_duration, p=probs) + 1  # durations start at 1
            durations.append(dur)
        return torch.tensor(durations, device=self.device)

    def _sample_next_states(self, curr_state: torch.Tensor, batch: int) -> torch.Tensor:
        next_states = []
        trans = F.softmax(self.transition_logits, dim=-1)
        for b in range(batch):
            probs = trans[curr_state[b]].detach().cpu().numpy()
            next_states.append(np.random.choice(self.n_codes, p=probs))
        return torch.tensor(next_states, device=self.device)

    def generate(self, n_samples: int, length: int) -> torch.Tensor:
        """
        Autoregressively generate codes via semi-Markov prior, then decode to trajectories.
        """
        codes = torch.zeros((n_samples, length), dtype=torch.long, device=self.device)
        state = torch.randint(0, self.n_codes, (n_samples,), device=self.device)
        positions = torch.zeros(n_samples, dtype=torch.long, device=self.device)

        while (positions < length).any():
            durations = self._sample_durations(state, n_samples)
            for b in range(n_samples):
                if positions[b] >= length:
                    continue
                dur = int(durations[b].item())
                end = min(positions[b] + dur, length)
                codes[b, positions[b] : end] = state[b]
                positions[b] = end
            state = self._sample_next_states(state, n_samples)
        traj = self.decode(codes)
        return traj


def discovery_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    codes: torch.Tensor,
    model: DiscoveryPipeline,
    alpha: float = 1.0,
    beta: float = 0.25,
    gamma: float = 10.0,
    delta: float = 5.0,
    min_change_rate: float = 0.15,
    usage_floor: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    recon_loss = F.mse_loss(reconstructed, original)

    if hasattr(model, "commitment_loss"):
        commit_loss = model.commitment_loss
    else:
        commit_loss = torch.tensor(0.0, device=original.device)

    with torch.no_grad():
        code_probs = torch.bincount(codes.flatten(), minlength=model.n_codes).float() / codes.numel()
        code_probs = code_probs + 1e-10
        codebook_entropy = -torch.sum(code_probs * torch.log(code_probs))
        target_entropy = math.log(model.n_codes)
        usage_penalty = F.relu(usage_floor - (code_probs > 0).float().mean())
    codebook_loss = target_entropy - codebook_entropy + usage_penalty

    change_rate = (codes[:, 1:] != codes[:, :-1]).float().mean()
    temporal_loss = F.relu(min_change_rate - change_rate)

    total_loss = alpha * recon_loss + beta * commit_loss + gamma * codebook_loss + delta * temporal_loss

    return total_loss, {
        "total": total_loss.item(),
        "reconstruction": recon_loss.item(),
        "commitment": commit_loss.item(),
        "codebook": codebook_loss.item(),
        "temporal": temporal_loss.item(),
    }


class FailureModeDetector:
    """Detect and warn about common failure modes."""

    def check_all(self, codes: torch.Tensor, original: torch.Tensor, reconstructed: torch.Tensor, n_codes: int) -> List[str]:
        issues = []
        if self.check_codebook_collapse(codes, n_codes):
            issues.append("CODEBOOK_COLLAPSE")
        if self.check_temporal_flickering(codes):
            issues.append("TEMPORAL_FLICKERING")
        if self.check_degenerate_segmentation(codes):
            issues.append("DEGENERATE_SEGMENTATION")
        if self.check_poor_reconstruction(original, reconstructed):
            issues.append("POOR_RECONSTRUCTION")
        return issues

    def check_codebook_collapse(self, codes: torch.Tensor, n_codes: int) -> bool:
        unique = len(torch.unique(codes))
        return unique < n_codes * 0.10

    def check_temporal_flickering(self, codes: torch.Tensor) -> bool:
        change_rate = (codes[:, 1:] != codes[:, :-1]).float().mean()
        return change_rate > 0.5

    def check_degenerate_segmentation(self, codes: torch.Tensor) -> bool:
        freqs = torch.bincount(codes.flatten()).float() / codes.numel()
        return torch.max(freqs) > 0.9

    def check_poor_reconstruction(self, original: torch.Tensor, reconstructed: torch.Tensor) -> bool:
        var_orig = torch.var(original)
        var_recon = torch.var(reconstructed)
        return var_recon < (var_orig * 0.1)


def train_pipeline(
    model: DiscoveryPipeline,
    data: np.ndarray,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[DiscoveryPipeline, List[Dict[str, float]]]:
    model.to(device)
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        epoch_losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, codes = model(batch)
            loss, logs = discovery_loss(batch, recon, codes, model)
            loss.backward()
            optimizer.step()
            epoch_losses.append(logs)
        mean_logs = {k: float(np.mean([l[k] for l in epoch_losses])) for k in epoch_losses[0].keys()}
        history.append(mean_logs)
    return model, history


class IntrinsicEvaluator:
    """
    Standardized evaluation for ANY behavior model.
    """

    def compute_reconstruction_mse(self, model: DiscoveryPipeline, real_data: torch.Tensor) -> float:
        with torch.no_grad():
            recon, _ = model(real_data)
            return F.mse_loss(recon, real_data).item()

    def compute_mean_bout_length(self, codes: torch.Tensor) -> float:
        codes_np = codes.cpu().numpy()
        bout_lengths = []
        for seq in codes_np:
            run_len = 1
            for t in range(1, len(seq)):
                if seq[t] == seq[t - 1]:
                    run_len += 1
                else:
                    bout_lengths.append(run_len)
                    run_len = 1
            bout_lengths.append(run_len)
        return float(np.mean(bout_lengths))

    def compute_mmd(self, real: torch.Tensor, synth: torch.Tensor, sigma: float = None, max_points: int = 4000) -> float:
        # Flatten temporal dimension and optionally subsample for tractability
        X = real.reshape(-1, real.size(-1))
        Y = synth.reshape(-1, synth.size(-1))
        if X.size(0) > max_points:
            idx = torch.randperm(X.size(0), device=X.device)[:max_points]
            X = X[idx]
        if Y.size(0) > max_points:
            idx = torch.randperm(Y.size(0), device=Y.device)[:max_points]
            Y = Y[idx]
        with torch.no_grad():
            if sigma is None:
                # Median heuristic
                sample = torch.cat([X[:: max(1, X.size(0) // 500)], Y[:: max(1, Y.size(0) // 500)]], dim=0)
                pdist = torch.cdist(sample, sample)
                sigma = torch.median(pdist[pdist > 0]).item() + 1e-6
            gamma = 1.0 / (2 * sigma**2)
            K_xx = torch.exp(-gamma * torch.cdist(X, X) ** 2)
            K_yy = torch.exp(-gamma * torch.cdist(Y, Y) ** 2)
            K_xy = torch.exp(-gamma * torch.cdist(X, Y) ** 2)
            mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
            return float(mmd.item())

    def compute_acf_error(self, real: torch.Tensor, synth: torch.Tensor, max_lag: int = 10) -> float:
        def acf(x: torch.Tensor) -> torch.Tensor:
            x = x - x.mean(dim=1, keepdim=True)
            acfs = []
            for lag in range(1, max_lag + 1):
                corr = (x[:, :-lag] * x[:, lag:]).mean(dim=(0, 1)) / (x.var(dim=(0, 1)) + 1e-6)
                acfs.append(corr)
            return torch.stack(acfs, dim=0)  # (lag, feat)

        with torch.no_grad():
            acf_real = acf(real)
            acf_synth = acf(synth)
            return float(torch.mean(torch.abs(acf_real - acf_synth)).item())

    def evaluate_all(self, model: DiscoveryPipeline, real_data: torch.Tensor) -> Dict[str, float]:
        results: Dict[str, float] = {}

        results["reconstruction_mse"] = self.compute_reconstruction_mse(model, real_data)

        codes = model.encode(real_data)
        results["codebook_usage"] = len(np.unique(codes.cpu().numpy())) / model.n_codes

        results["mean_bout_length"] = self.compute_mean_bout_length(codes)

        synthetic_data = model.generate(n_samples=real_data.shape[0], length=real_data.shape[1]).detach()

        results["mmd_score"] = self.compute_mmd(real_data, synthetic_data)

        results["acf_error"] = self.compute_acf_error(real_data, synthetic_data)

        results["discovery_score"] = (
            (1.0 / (results["reconstruction_mse"] + 1e-6)) * 0.3
            + (results["codebook_usage"]) * 0.2
            + (1.0 / (results["mmd_score"] + 1e-6)) * 0.3
            + (1.0 / (results["acf_error"] + 1e-6)) * 0.2
        )

        return results


class ExtrinsicEvaluator:
    def evaluate_with_labels(self, model: DiscoveryPipeline, data: torch.Tensor, labels: np.ndarray) -> Dict[str, float]:
        codes = model.encode(data)
        labels_flat = np.asarray(labels).flatten()
        codes_flat = codes.flatten().cpu().numpy()
        valid = ~np.isnan(labels_flat)
        labels_valid = labels_flat[valid]
        codes_valid = codes_flat[valid]
        if labels_valid.size == 0:
            ari = float("nan")
        else:
            ari = adjusted_rand_score(labels_valid, codes_valid)
        return {"ari": float(ari)}


def load_dataset(path: str, preprocessor: BehaviorPreprocessor) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    trajectories = data["trajectories"]
    labels = data["labels"]
    processed = preprocessor.preprocess(trajectories)
    return processed, labels


def run_training(
    data_path: str,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 8,
    n_codes: int = 32,
):
    preprocessor = BehaviorPreprocessor()
    processed, labels = load_dataset(data_path, preprocessor)
    feature_dim = processed.shape[-1]
    model = DiscoveryPipeline(feature_dim=feature_dim, n_codes=n_codes, device=device)
    model, history = train_pipeline(model, processed, epochs=epochs, batch_size=batch_size, device=device)

    tensor_data = torch.tensor(processed, dtype=torch.float32, device=device)
    intrinsic = IntrinsicEvaluator().evaluate_all(model, tensor_data)
    extrinsic = ExtrinsicEvaluator().evaluate_with_labels(model, tensor_data, labels)
    return model, history, intrinsic, extrinsic


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HSMM-VAE Discovery Pipeline")
    parser.add_argument("--data", type=str, default="data/fly_data/mabe22_subset_for_claude.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_codes", type=int, default=32)
    args = parser.parse_args()

    model, history, intrinsic, extrinsic = run_training(
        data_path=args.data,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_codes=args.n_codes,
    )

    print("Training complete. Last epoch losses:", history[-1])
    print("Intrinsic evaluation:", intrinsic)
    print("Extrinsic evaluation:", extrinsic)
