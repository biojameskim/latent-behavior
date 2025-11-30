"""
Intrinsic and Extrinsic Evaluators
Standardized evaluation for ANY behavior model
"""

import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy import signal
from tqdm import tqdm


class IntrinsicEvaluator:
    """
    Standardized evaluation for ANY behavior model.
    Does NOT use ground truth labels.
    """

    def __init__(self):
        pass

    def evaluate_all(self, model, real_data, device='cuda'):
        """
        Compute all intrinsic metrics

        Args:
            model: DiscoveryPipeline instance
            real_data: torch.Tensor (N, Time, Features) - original raw data
            device: 'cuda' or 'cpu'

        Returns:
            results: dict of metrics
        """
        model.eval()
        model = model.to(device)
        real_data = real_data.to(device)

        results = {}

        print("Running intrinsic evaluation...")

        # 1. Reconstruction quality
        print("  - Computing reconstruction MSE...")
        results['reconstruction_mse'] = self.compute_reconstruction_mse(model, real_data)

        # 2. Code statistics
        print("  - Analyzing code usage...")
        codes = model.encode(model.preprocessor.preprocess(real_data))
        results['codebook_usage'] = len(np.unique(codes.cpu().numpy())) / model.n_codes

        # 3. Temporal properties
        print("  - Computing temporal statistics...")
        results['mean_bout_length'] = self.compute_mean_bout_length(codes)

        # 4. Generative quality (CRITICAL)
        print("  - Generating synthetic data...")
        synthetic_data = model.generate(n_samples=real_data.shape[0], length=real_data.shape[1])

        # Preprocess real data for fair comparison
        real_data_processed = model.preprocessor.preprocess(real_data)

        print("  - Computing MMD score...")
        results['mmd_score'] = self.compute_mmd(real_data_processed, synthetic_data)

        print("  - Computing ACF error...")
        results['acf_error'] = self.compute_acf_error(real_data_processed, synthetic_data)

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
        """
        Compute mean squared error between original and reconstructed data
        """
        with torch.no_grad():
            # Preprocess
            processed = model.preprocessor.preprocess(real_data)

            # Encode and decode
            reconstructed, _ = model(processed)

            # MSE
            mse = torch.mean((processed - reconstructed) ** 2).item()

        return mse

    def compute_mean_bout_length(self, codes):
        """
        Compute average duration of behavioral bouts (runs of same code)

        Args:
            codes: (N, Time) tensor of discrete codes

        Returns:
            mean_bout_length: average number of consecutive frames with same code
        """
        codes_np = codes.cpu().numpy()
        bout_lengths = []

        for seq in codes_np:
            # Find where codes change
            changes = np.where(seq[1:] != seq[:-1])[0] + 1
            # Add start and end
            boundaries = np.concatenate([[0], changes, [len(seq)]])
            # Compute bout lengths
            lengths = np.diff(boundaries)
            bout_lengths.extend(lengths)

        return float(np.mean(bout_lengths))

    def compute_mmd(self, real_data, synthetic_data, kernel='rbf', bandwidth=1.0):
        """
        Maximum Mean Discrepancy - measures distribution similarity
        Uses RBF kernel

        Args:
            real_data: (N, Time, Features)
            synthetic_data: (N, Time, Features)
            kernel: 'rbf' or 'linear'
            bandwidth: kernel bandwidth

        Returns:
            mmd: scalar distance between distributions
        """
        # Flatten time and features for distribution comparison
        real_flat = real_data.reshape(-1, real_data.shape[-1])
        synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])

        # Subsample for computational efficiency
        max_samples = 5000
        if real_flat.shape[0] > max_samples:
            indices_real = np.random.choice(real_flat.shape[0], max_samples, replace=False)
            real_flat = real_flat[indices_real]
        if synth_flat.shape[0] > max_samples:
            indices_synth = np.random.choice(synth_flat.shape[0], max_samples, replace=False)
            synth_flat = synth_flat[indices_synth]

        # Convert to numpy for efficiency
        X = real_flat.cpu().numpy()
        Y = synth_flat.cpu().numpy()

        # Compute kernel matrices
        def rbf_kernel(X, Y, bandwidth):
            # Squared Euclidean distances
            XX = np.sum(X**2, axis=1)[:, np.newaxis]
            YY = np.sum(Y**2, axis=1)[np.newaxis, :]
            XY = np.dot(X, Y.T)
            dists = XX + YY - 2*XY
            return np.exp(-dists / (2 * bandwidth**2))

        if kernel == 'rbf':
            K_XX = rbf_kernel(X, X, bandwidth)
            K_YY = rbf_kernel(Y, Y, bandwidth)
            K_XY = rbf_kernel(X, Y, bandwidth)
        else:  # linear
            K_XX = np.dot(X, X.T)
            K_YY = np.dot(Y, Y.T)
            K_XY = np.dot(X, Y.T)

        # MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
        m = X.shape[0]
        n = Y.shape[0]

        mmd_squared = (K_XX.sum() - np.trace(K_XX)) / (m * (m-1)) + \
                      (K_YY.sum() - np.trace(K_YY)) / (n * (n-1)) - \
                      2 * K_XY.sum() / (m * n)

        return float(np.sqrt(max(mmd_squared, 0)))

    def compute_acf_error(self, real_data, synthetic_data, max_lag=50):
        """
        Autocorrelation Function error - measures temporal dynamics similarity

        Computes ACF for each feature dimension and averages the error

        Args:
            real_data: (N, Time, Features)
            synthetic_data: (N, Time, Features)
            max_lag: maximum lag to compute ACF

        Returns:
            acf_error: mean absolute difference in ACF
        """
        real_np = real_data.cpu().numpy()
        synth_np = synthetic_data.cpu().numpy()

        # Compute ACF for each feature dimension
        n_features = real_np.shape[-1]
        acf_errors = []

        # Sample a subset of features for efficiency
        feature_indices = np.random.choice(n_features, min(20, n_features), replace=False)

        for feat_idx in feature_indices:
            # Extract feature across all sequences
            real_feat = real_np[:, :, feat_idx].flatten()
            synth_feat = synth_np[:, :, feat_idx].flatten()

            # Compute autocorrelation
            real_acf = self._compute_acf(real_feat, max_lag)
            synth_acf = self._compute_acf(synth_feat, max_lag)

            # Mean absolute error
            acf_errors.append(np.mean(np.abs(real_acf - synth_acf)))

        return float(np.mean(acf_errors))

    def _compute_acf(self, x, max_lag):
        """
        Compute autocorrelation function

        Args:
            x: 1D array
            max_lag: maximum lag

        Returns:
            acf: autocorrelation values for lags 0 to max_lag
        """
        # Demean
        x = x - np.mean(x)

        # Compute ACF using FFT (efficient)
        acf = np.correlate(x, x, mode='full')
        acf = acf[len(acf)//2:]  # Take positive lags
        acf = acf / acf[0]  # Normalize

        return acf[:max_lag+1]


class ExtrinsicEvaluator:
    """
    Evaluation using ground truth labels
    ONLY used after intrinsic evaluation for validation
    """

    def evaluate_with_labels(self, model, data, labels, device='cuda'):
        """
        Evaluate discovered codes against ground truth labels

        Args:
            model: DiscoveryPipeline instance
            data: torch.Tensor (N, Time, Features) - original raw data
            labels: torch.Tensor or np.array (N, Time) - ground truth labels
            device: 'cuda' or 'cpu'

        Returns:
            results: dict with ARI and other metrics
        """
        model.eval()
        model = model.to(device)
        data = data.to(device)

        print("Running extrinsic evaluation (with ground truth)...")

        # Encode data
        with torch.no_grad():
            processed = model.preprocessor.preprocess(data)
            codes = model.encode(processed)

        # Convert to numpy
        codes_np = codes.cpu().numpy().flatten()

        if torch.is_tensor(labels):
            labels_np = labels.cpu().numpy().flatten()
        else:
            labels_np = labels.flatten()

        # Compute Adjusted Rand Index
        ari = adjusted_rand_score(labels_np, codes_np)

        # Additional metrics
        # Normalized Mutual Information
        from sklearn.metrics import normalized_mutual_info_score
        nmi = normalized_mutual_info_score(labels_np, codes_np)

        # Homogeneity and Completeness
        from sklearn.metrics import homogeneity_completeness_v_measure
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            labels_np, codes_np
        )

        results = {
            'ari': ari,
            'nmi': nmi,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'v_measure': v_measure
        }

        return results
