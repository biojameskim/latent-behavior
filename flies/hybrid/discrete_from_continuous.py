"""
Hybrid approach: Train continuous model → Extract discrete tokens

This implements your idea:
"Train model first and then extract discrete tokens from the model directly.
The discrete representations/tokens we extract from the model will have richer
representations of behavior that we can then input to VQ-VAE to get behavior codes."

Two strategies:
1. Cluster continuous latents from VAE/forecaster
2. Train VQ-VAE on top of continuous latents (hierarchical)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Optional
import pickle


class DiscreteTokenExtractor:
    """
    Extract discrete behavior tokens from continuous latent representations.

    Workflow:
        1. Train continuous model (VAE or forecaster)
        2. Extract continuous latents for all training data
        3. Cluster latents to create discrete vocabulary
        4. Use cluster IDs as behavior tokens

    This creates a discrete representation that:
    - Preserves dynamics learned by continuous model
    - May have richer semantics than direct VQ-VAE on keypoints
    - Can be used for downstream analysis (Markov models, etc.)
    """

    def __init__(
        self,
        continuous_model: nn.Module,
        num_clusters: int = 512,
        clustering_method: str = 'kmeans',
        use_pca: bool = False,
        pca_dims: int = 32,
    ):
        """
        Args:
            continuous_model: Trained VAE or forecaster
            num_clusters: Number of discrete behavior tokens
            clustering_method: 'kmeans' or 'minibatch_kmeans'
            use_pca: Whether to apply PCA before clustering
            pca_dims: PCA dimensions if use_pca=True
        """
        self.model = continuous_model
        self.num_clusters = num_clusters
        self.clustering_method = clustering_method
        self.use_pca = use_pca
        self.pca_dims = pca_dims

        self.clusterer = None
        self.pca = None
        self.is_fitted = False

    def extract_continuous_latents(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        use_mean: bool = True,
    ) -> np.ndarray:
        """
        Extract continuous latent representations for all data.

        Args:
            dataloader: DataLoader with behavior sequences
            device: Device to run inference on
            use_mean: For VAE, use mean μ instead of sampling

        Returns:
            latents: Array of shape (N, latent_dim) or (N, latent_dim * T)
        """
        self.model.eval()
        self.model.to(device)

        all_latents = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                # Extract latents based on model type
                if hasattr(self.model, 'get_latent_codes'):
                    # VAE with get_latent_codes method
                    latents = self.model.get_latent_codes(x, use_mean=use_mean)
                elif hasattr(self.model, 'encode'):
                    # VAE with encode method
                    mu, logvar = self.model.encode(x)
                    latents = mu if use_mean else self.model.reparameterize(mu, logvar)
                elif hasattr(self.model, 'encoder'):
                    # Generic encoder
                    latents = self.model.encoder(x)
                else:
                    raise ValueError("Model must have get_latent_codes, encode, or encoder method")

                # Flatten temporal dimension: (batch, latent_dim, T) → (batch, latent_dim * T)
                batch_size = latents.size(0)
                latents_flat = latents.reshape(batch_size, -1)

                all_latents.append(latents_flat.cpu().numpy())

        # Concatenate all batches
        all_latents = np.concatenate(all_latents, axis=0)  # (N, latent_dim * T)

        print(f"Extracted {all_latents.shape[0]} latent vectors of dimension {all_latents.shape[1]}")

        return all_latents

    def fit_clustering(self, latents: np.ndarray, verbose: bool = True) -> 'DiscreteTokenExtractor':
        """
        Fit clustering model on continuous latents.

        Args:
            latents: Array of shape (N, latent_dim * T)
            verbose: Print progress

        Returns:
            self
        """
        if verbose:
            print(f"Fitting {self.clustering_method} with {self.num_clusters} clusters...")

        # Optional PCA dimensionality reduction
        if self.use_pca:
            if verbose:
                print(f"Applying PCA: {latents.shape[1]} → {self.pca_dims} dims")
            self.pca = PCA(n_components=self.pca_dims)
            latents = self.pca.fit_transform(latents)

        # Fit clustering
        if self.clustering_method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.num_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                verbose=1 if verbose else 0,
            )
        elif self.clustering_method == 'minibatch_kmeans':
            self.clusterer = MiniBatchKMeans(
                n_clusters=self.num_clusters,
                random_state=42,
                batch_size=1024,
                max_iter=100,
                verbose=1 if verbose else 0,
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        self.clusterer.fit(latents)
        self.is_fitted = True

        if verbose:
            print("Clustering complete!")

        return self

    def predict_tokens(self, latents: np.ndarray) -> np.ndarray:
        """
        Convert continuous latents to discrete tokens.

        Args:
            latents: Array of shape (N, latent_dim * T)

        Returns:
            tokens: Array of shape (N,) with cluster IDs (0 to num_clusters-1)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_clustering first")

        if self.use_pca:
            latents = self.pca.transform(latents)

        tokens = self.clusterer.predict(latents)
        return tokens

    def encode_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
    ) -> np.ndarray:
        """
        End-to-end: continuous data → continuous latents → discrete tokens.

        Args:
            dataloader: DataLoader with behavior sequences

        Returns:
            tokens: Array of discrete token IDs
        """
        latents = self.extract_continuous_latents(dataloader, device=device)
        tokens = self.predict_tokens(latents)
        return tokens

    def fit_and_encode(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
    ) -> np.ndarray:
        """
        Convenience method: extract latents + fit clustering + encode.

        Args:
            train_dataloader: DataLoader for training data

        Returns:
            tokens: Discrete tokens for training data
        """
        # Extract continuous latents
        latents = self.extract_continuous_latents(train_dataloader, device=device)

        # Fit clustering
        self.fit_clustering(latents)

        # Predict tokens
        tokens = self.predict_tokens(latents)

        return tokens

    def save(self, path: str):
        """Save clustering model and PCA."""
        save_dict = {
            'num_clusters': self.num_clusters,
            'clustering_method': self.clustering_method,
            'use_pca': self.use_pca,
            'pca_dims': self.pca_dims,
            'clusterer': self.clusterer,
            'pca': self.pca,
            'is_fitted': self.is_fitted,
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Saved token extractor to {path}")

    @classmethod
    def load(cls, path: str, continuous_model: nn.Module) -> 'DiscreteTokenExtractor':
        """Load clustering model and PCA."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        extractor = cls(
            continuous_model=continuous_model,
            num_clusters=save_dict['num_clusters'],
            clustering_method=save_dict['clustering_method'],
            use_pca=save_dict['use_pca'],
            pca_dims=save_dict['pca_dims'],
        )
        extractor.clusterer = save_dict['clusterer']
        extractor.pca = save_dict['pca']
        extractor.is_fitted = save_dict['is_fitted']

        return extractor


class HierarchicalVQVAE(nn.Module):
    """
    Hierarchical VQ-VAE: Train VQ-VAE on top of continuous latents.

    Architecture:
        Keypoints → Continuous model (VAE/forecaster) → Continuous latents
        → VQ-VAE (small) → Discrete codes

    This creates a two-level hierarchy:
        Level 1: Continuous dynamics model (captures continuous variations)
        Level 2: Discrete tokenization (captures categorical behavior types)

    The idea: Continuous model learns rich dynamics, then VQ-VAE
    creates discrete vocabulary on top of those learned representations.
    """

    def __init__(
        self,
        continuous_model: nn.Module,
        latent_dim: int = 128,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        freeze_continuous: bool = True,
    ):
        """
        Args:
            continuous_model: Pre-trained VAE or forecaster
            latent_dim: Output dimension of continuous model
            num_embeddings: Size of discrete codebook
            embedding_dim: Dimension of quantized embeddings
            commitment_cost: β for VQ loss
            freeze_continuous: Freeze continuous model weights
        """
        super().__init__()

        self.continuous_model = continuous_model
        self.freeze_continuous = freeze_continuous

        if freeze_continuous:
            # Freeze continuous model (only train quantizer)
            for param in self.continuous_model.parameters():
                param.requires_grad = False

        # Small encoder: continuous latents → quantizer input
        self.pre_quantizer = nn.Sequential(
            nn.Conv1d(latent_dim, embedding_dim, kernel_size=1),
            nn.GroupNorm(1, embedding_dim),
        )

        # Vector quantizer
        from .quantizer import VectorQuantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        # Small decoder: quantized embeddings → continuous latents
        self.post_quantizer = nn.Conv1d(embedding_dim, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: Input keypoints (batch, 48, 150)

        Returns:
            x_recon: Reconstructed keypoints
            vq_loss: Vector quantization loss
            info: Dictionary with intermediate values
        """
        # Extract continuous latents
        if hasattr(self.continuous_model, 'encode'):
            mu, logvar = self.continuous_model.encode(x)
            z_continuous = mu  # Use mean for deterministic encoding
        else:
            z_continuous = self.continuous_model.encoder(x)

        # Prepare for quantization
        z_pre_quant = self.pre_quantizer(z_continuous)

        # Quantize
        z_quant, vq_loss, perplexity, encodings = self.quantizer(z_pre_quant)

        # Reconstruct continuous latents
        z_post_quant = self.post_quantizer(z_quant)

        # Decode to keypoints
        if hasattr(self.continuous_model, 'decode'):
            x_recon = self.continuous_model.decode(z_post_quant)
        else:
            x_recon = self.continuous_model.decoder(z_post_quant)

        info = {
            'z_continuous': z_continuous,
            'z_pre_quant': z_pre_quant,
            'z_quant': z_quant,
            'perplexity': perplexity,
            'encodings': encodings,
        }

        return x_recon, vq_loss, info

    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Extract discrete codes."""
        with torch.no_grad():
            if hasattr(self.continuous_model, 'encode'):
                mu, _ = self.continuous_model.encode(x)
                z_continuous = mu
            else:
                z_continuous = self.continuous_model.encoder(x)

            z_pre_quant = self.pre_quantizer(z_continuous)
            _, _, _, encodings = self.quantizer(z_pre_quant)

        return encodings
