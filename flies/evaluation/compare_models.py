"""
Comprehensive comparison framework for discrete vs continuous behavior models.

Compares:
1. VQ-VAE (discrete tokenization)
2. VAE (continuous latent space)
3. Transformer/LSTM forecaster (continuous dynamics)
4. Hybrid (continuous → discrete tokens)

Evaluation metrics:
- Reconstruction quality (MSE, per-keypoint error)
- Temporal coherence (velocity/acceleration consistency)
- Representation quality (clustering, interpretability)
- Behavioral dynamics preservation
- Downstream task performance (if applicable)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import FlyKeypointDataset


class ModelComparator:
    """
    Compare discrete vs continuous behavior models.

    Workflow:
        1. Load all trained models
        2. Extract representations on same test set
        3. Compute reconstruction metrics
        4. Compute representation metrics
        5. Analyze behavioral dynamics
        6. Generate comparison report
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.models = {}
        self.results = {}

    def add_model(self, name: str, model: nn.Module, model_type: str):
        """Add a model to compare."""
        self.models[name] = {
            'model': model.to(self.device),
            'type': model_type,
        }
        model.eval()

    def compute_reconstruction_metrics(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate reconstruction quality for all models.

        Metrics:
        - MSE (overall)
        - MSE per keypoint
        - MSE per body part (wings, legs, etc.)
        - Temporal consistency (velocity/acceleration errors)
        """
        results = {}

        for name, model_dict in self.models.items():
            model = model_dict['model']
            model_type = model_dict['type']

            print(f"\nEvaluating reconstruction: {name}")

            mse_total = []
            mse_per_keypoint = [[] for _ in range(24)]
            velocity_error = []
            acceleration_error = []

            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0].to(self.device)
                    else:
                        x = batch.to(self.device)

                    # Get reconstruction based on model type
                    if model_type == 'vqvae':
                        x_recon, _, _, _ = model(x)
                    elif model_type in ['vae', 'beta_vae', 'annealed_vae']:
                        x_recon, _ = model(x)
                    elif model_type in ['transformer', 'lstm']:
                        # For forecasters, only evaluate second half
                        context_len = model.context_length
                        x_context = x[:, :, :context_len]
                        x_future = x[:, :, context_len:]
                        x_recon = model(x_context)
                        x = x_future  # Compare only forecasted part
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")

                    # Overall MSE
                    mse = torch.mean((x - x_recon) ** 2).item()
                    mse_total.append(mse)

                    # Per-keypoint MSE
                    # Reshape: (batch, 48, T) → (batch, T, 24, 2)
                    batch_size, _, seq_len = x.shape
                    x_kp = x.transpose(1, 2).reshape(batch_size, seq_len, 24, 2)
                    x_recon_kp = x_recon.transpose(1, 2).reshape(batch_size, seq_len, 24, 2)

                    for kp_idx in range(24):
                        kp_mse = torch.mean((x_kp[:, :, kp_idx, :] - x_recon_kp[:, :, kp_idx, :]) ** 2).item()
                        mse_per_keypoint[kp_idx].append(kp_mse)

                    # Temporal consistency: velocity and acceleration errors
                    # Velocity: difference between consecutive frames
                    x_vel = x[:, :, 1:] - x[:, :, :-1]
                    x_recon_vel = x_recon[:, :, 1:] - x_recon[:, :, :-1]
                    vel_err = torch.mean((x_vel - x_recon_vel) ** 2).item()
                    velocity_error.append(vel_err)

                    # Acceleration: second derivative
                    x_acc = x_vel[:, :, 1:] - x_vel[:, :, :-1]
                    x_recon_acc = x_recon_vel[:, :, 1:] - x_recon_vel[:, :, :-1]
                    acc_err = torch.mean((x_acc - x_recon_acc) ** 2).item()
                    acceleration_error.append(acc_err)

            # Aggregate results
            results[name] = {
                'mse_total': np.mean(mse_total),
                'mse_std': np.std(mse_total),
                'mse_per_keypoint': [np.mean(kp) for kp in mse_per_keypoint],
                'velocity_error': np.mean(velocity_error),
                'acceleration_error': np.mean(acceleration_error),
            }

            print(f"  MSE: {results[name]['mse_total']:.6f} ± {results[name]['mse_std']:.6f}")
            print(f"  Velocity error: {results[name]['velocity_error']:.6f}")
            print(f"  Acceleration error: {results[name]['acceleration_error']:.6f}")

        return results

    def compute_representation_metrics(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate representation quality.

        Metrics:
        - Latent space structure (for continuous models)
        - Codebook utilization (for discrete models)
        - Clustering quality (silhouette score, Davies-Bouldin)
        - Representation dimensionality (intrinsic dimension)
        """
        results = {}

        for name, model_dict in self.models.items():
            model = model_dict['model']
            model_type = model_dict['type']

            print(f"\nEvaluating representations: {name}")

            latents = []
            discrete_codes = []

            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0].to(self.device)
                    else:
                        x = batch.to(self.device)

                    # Extract representations
                    if model_type == 'vqvae':
                        _, _, _, encodings = model(x)
                        discrete_codes.extend(encodings.cpu().numpy().flatten())
                    elif model_type in ['vae', 'beta_vae', 'annealed_vae']:
                        if hasattr(model, 'get_latent_codes'):
                            z = model.get_latent_codes(x, use_mean=True)
                        else:
                            mu, _ = model.encode(x)
                            z = mu
                        # Flatten temporal dimension
                        batch_size = z.size(0)
                        z_flat = z.reshape(batch_size, -1)
                        latents.append(z_flat.cpu().numpy())
                    elif model_type in ['transformer', 'lstm']:
                        # For forecasters, use encoder output
                        if hasattr(model, 'encoder'):
                            z = model.encoder(x)
                            batch_size = z.size(0)
                            z_flat = z.reshape(batch_size, -1)
                            latents.append(z_flat.cpu().numpy())

            # Analyze discrete codes
            if discrete_codes:
                discrete_codes = np.array(discrete_codes)
                unique_codes = np.unique(discrete_codes)
                code_counts = np.bincount(discrete_codes)
                code_usage = len(unique_codes)
                code_entropy = -np.sum((code_counts / code_counts.sum()) * np.log(code_counts / code_counts.sum() + 1e-10))

                results[name] = {
                    'type': 'discrete',
                    'num_codes_total': len(code_counts),
                    'num_codes_used': code_usage,
                    'code_usage_ratio': code_usage / len(code_counts),
                    'code_entropy': code_entropy,
                }

                print(f"  Codebook usage: {code_usage}/{len(code_counts)} ({results[name]['code_usage_ratio']:.2%})")
                print(f"  Code entropy: {code_entropy:.2f}")

            # Analyze continuous latents
            elif latents:
                latents = np.concatenate(latents, axis=0)

                # Subsample for clustering metrics (expensive)
                if len(latents) > 5000:
                    indices = np.random.choice(len(latents), 5000, replace=False)
                    latents_sample = latents[indices]
                else:
                    latents_sample = latents

                # Intrinsic dimensionality (variance explained)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(50, latents.shape[1]))
                pca.fit(latents_sample)
                var_explained_90 = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.9) + 1
                var_explained_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.95) + 1

                # Clustering quality (K-means on PCA)
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=64, random_state=42)
                cluster_labels = kmeans.fit_predict(pca.transform(latents_sample))

                silhouette = silhouette_score(pca.transform(latents_sample), cluster_labels)
                davies_bouldin = davies_bouldin_score(pca.transform(latents_sample), cluster_labels)

                results[name] = {
                    'type': 'continuous',
                    'latent_dim': latents.shape[1],
                    'intrinsic_dim_90': var_explained_90,
                    'intrinsic_dim_95': var_explained_95,
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin,
                }

                print(f"  Latent dim: {latents.shape[1]}")
                print(f"  Intrinsic dim (90% var): {var_explained_90}")
                print(f"  Intrinsic dim (95% var): {var_explained_95}")
                print(f"  Silhouette score: {silhouette:.4f}")
                print(f"  Davies-Bouldin score: {davies_bouldin:.4f}")

        return results

    def analyze_behavioral_dynamics(
        self,
        dataloader: DataLoader,
        save_dir: str = 'dynamics_analysis',
    ) -> Dict[str, Dict]:
        """
        Analyze how well models preserve behavioral dynamics.

        Metrics:
        - Transition statistics (for discrete codes)
        - Temporal autocorrelation (for continuous latents)
        - Behavioral motif preservation
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        results = {}

        for name, model_dict in self.models.items():
            model = model_dict['model']
            model_type = model_dict['type']

            print(f"\nAnalyzing dynamics: {name}")

            if model_type == 'vqvae':
                # Analyze discrete code sequences
                all_sequences = []

                with torch.no_grad():
                    for batch in dataloader:
                        if isinstance(batch, (list, tuple)):
                            x = batch[0].to(self.device)
                        else:
                            x = batch.to(self.device)

                        _, _, _, encodings = model(x)
                        all_sequences.extend(encodings.cpu().numpy())

                all_sequences = np.array(all_sequences)

                # Transition matrix
                num_codes = model.num_embeddings
                transition_matrix = np.zeros((num_codes, num_codes))

                for seq in all_sequences:
                    for i in range(len(seq) - 1):
                        transition_matrix[seq[i], seq[i+1]] += 1

                # Normalize
                row_sums = transition_matrix.sum(axis=1, keepdims=True)
                transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

                # Transition entropy
                transition_entropy = -np.nansum(transition_matrix * np.log(transition_matrix + 1e-10)) / num_codes

                results[name] = {
                    'transition_matrix': transition_matrix,
                    'transition_entropy': transition_entropy,
                }

                print(f"  Transition entropy: {transition_entropy:.4f}")

                # Plot transition matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(transition_matrix, cmap='viridis', cbar=True)
                plt.title(f"Transition Matrix: {name}")
                plt.xlabel("Next Code")
                plt.ylabel("Current Code")
                plt.tight_layout()
                plt.savefig(f"{save_dir}/{name}_transition_matrix.png", dpi=150)
                plt.close()

        return results

    def generate_report(self, output_file: str = 'comparison_report.json'):
        """Generate comprehensive comparison report."""
        report = {
            'models': list(self.models.keys()),
            'results': self.results,
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

        print(f"\nComparison report saved to {output_file}")

        # Print summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        for name in self.models.keys():
            print(f"\n{name}:")
            if 'reconstruction' in self.results:
                print(f"  MSE: {self.results['reconstruction'][name]['mse_total']:.6f}")
            if 'representation' in self.results:
                if self.results['representation'][name]['type'] == 'discrete':
                    print(f"  Code usage: {self.results['representation'][name]['code_usage_ratio']:.2%}")
                else:
                    print(f"  Intrinsic dim: {self.results['representation'][name]['intrinsic_dim_90']}")


def run_full_comparison(
    model_configs: List[Tuple[str, nn.Module, str]],
    dataloader: DataLoader,
    output_dir: str = 'comparison_results',
):
    """
    Run full comparison pipeline.

    Args:
        model_configs: List of (name, model, model_type) tuples
        dataloader: Test dataloader
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    comparator = ModelComparator()

    # Add all models
    for name, model, model_type in model_configs:
        comparator.add_model(name, model, model_type)

    # Run evaluations
    print("\n" + "="*80)
    print("RECONSTRUCTION METRICS")
    print("="*80)
    recon_results = comparator.compute_reconstruction_metrics(dataloader)
    comparator.results['reconstruction'] = recon_results

    print("\n" + "="*80)
    print("REPRESENTATION METRICS")
    print("="*80)
    repr_results = comparator.compute_representation_metrics(dataloader)
    comparator.results['representation'] = repr_results

    print("\n" + "="*80)
    print("BEHAVIORAL DYNAMICS")
    print("="*80)
    dynamics_results = comparator.analyze_behavioral_dynamics(
        dataloader,
        save_dir=os.path.join(output_dir, 'dynamics'),
    )
    comparator.results['dynamics'] = dynamics_results

    # Generate report
    comparator.generate_report(os.path.join(output_dir, 'comparison_report.json'))

    return comparator
