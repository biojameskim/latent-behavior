import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import adjusted_rand_score

# ==========================================
# Phase 2: Intrinsic Evaluation (Standardized)
# ==========================================

class IntrinsicEvaluator:
    """
    Standardized evaluation for ANY behavior model.
    """
    
    def evaluate_all(self, model, real_data):
        results = {}
        model.eval()
        
        # 1. Reconstruction quality
        results['reconstruction_mse'] = self.compute_reconstruction_mse(model, real_data)
        
        # 2. Code statistics
        with torch.no_grad():
            codes = model.encode(real_data)
        results['codebook_usage'] = len(np.unique(codes.cpu().numpy())) / model.n_codes
        
        # 3. Temporal properties
        results['mean_bout_length'] = self.compute_mean_bout_length(codes)
        
        # 4. Generative quality (CRITICAL)
        # Generate synthetic data of same shape as real_data
        # Note: Generating full dataset might be slow, so we sample a subset if needed.
        # Here we generate same size.
        with torch.no_grad():
            synthetic_data = model.generate(n_samples=real_data.shape[0], length=real_data.shape[1])
        
        # Compute MMD (Maximum Mean Discrepancy) - Kinematic Distribution match
        results['mmd_score'] = self.compute_mmd(real_data, synthetic_data)
        
        # Compute ACF (Autocorrelation) match - Temporal Dynamics match
        results['acf_error'] = self.compute_acf_error(real_data, synthetic_data)
        
        # 5. Combined Discovery Score
        # We want Low MSE, High Usage, Low MMD, Low ACF Error
        results['discovery_score'] = (
            (1.0 / (results['reconstruction_mse'] + 1e-6)) * 0.3 +
            (results['codebook_usage']) * 0.2 +
            (1.0 / (results['mmd_score'] + 1e-6)) * 0.3 +
            (1.0 / (results['acf_error'] + 1e-6)) * 0.2
        )
        
        return results

    def compute_reconstruction_mse(self, model, real_data):
        with torch.no_grad():
            recon, _ = model(real_data)
        return F.mse_loss(recon, real_data).item()

    def compute_mean_bout_length(self, codes):
        # codes: (B, T)
        codes_np = codes.cpu().numpy()
        bout_lengths = []
        for seq in codes_np:
            # Find runs of same code
            diffs = np.diff(seq)
            change_points = np.where(diffs != 0)[0] + 1
            # Add start and end
            change_points = np.concatenate(([0], change_points, [len(seq)]))
            lengths = np.diff(change_points)
            bout_lengths.extend(lengths)
        return np.mean(bout_lengths)

    def compute_mmd(self, real, synthetic):
        """
        Maximum Mean Discrepancy between real and synthetic distributions.
        Using a simple Gaussian kernel.
        """
        # Flatten time: (B*T, F) to compare frame distributions
        # Subsample if too large
        n_samples = min(2000, real.shape[0] * real.shape[1])
        
        real_flat = real.reshape(-1, real.shape[-1])
        syn_flat = synthetic.reshape(-1, synthetic.shape[-1])
        
        idx_r = torch.randperm(real_flat.shape[0])[:n_samples]
        idx_s = torch.randperm(syn_flat.shape[0])[:n_samples]
        
        x = real_flat[idx_r]
        y = syn_flat[idx_s]
        
        return self._mmd_linear(x, y).item()

    def _mmd_linear(self, x, y):
        # Linear MMD approximation for speed
        delta = x.mean(0) - y.mean(0)
        return torch.dot(delta, delta)

    def compute_acf_error(self, real, synthetic):
        """
        Compare Autocorrelation Functions.
        """
        # Compute ACF for each feature averaged over batch
        # real: (B, T, F)
        def get_acf(data, max_lag=50):
            # data: (B, T, F)
            B, T, F = data.shape
            acfs = []
            for b in range(min(B, 10)): # Average over first 10 samples
                for f in range(F):
                    series = data[b, :, f]
                    mean = series.mean()
                    var = series.var() + 1e-6
                    acf = []
                    for lag in range(max_lag):
                        if lag == 0:
                            c = 1.0
                        else:
                            c = ((series[lag:] - mean) * (series[:-lag] - mean)).mean() / var
                        acf.append(c)
                    acfs.append(acf)
            return torch.tensor(acfs).mean(dim=0) # (Lags,)

        real_acf = get_acf(real)
        syn_acf = get_acf(synthetic)
        
        return F.mse_loss(real_acf, syn_acf).item()

# ==========================================
# Phase 3: Validation (Ground Truth Reveal)
# ==========================================

class ExtrinsicEvaluator:
    def evaluate_with_labels(self, model, data, labels):
        model.eval()
        with torch.no_grad():
            codes = model.encode(data)
        
        # ARI: How well do codes match human labels?
        labels_flat = np.asarray(labels).flatten()
        codes_flat = codes.flatten().cpu().numpy()
        # Mask out NaNs in labels (and corresponding codes) to avoid invalid ARI
        valid = ~np.isnan(labels_flat)
        labels_valid = labels_flat[valid]
        codes_valid = codes_flat[valid]
        if labels_valid.size == 0:
            ari = float("nan")
        else:
            ari = adjusted_rand_score(labels_valid, codes_valid)
        return {'ari': ari}
