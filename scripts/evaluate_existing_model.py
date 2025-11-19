"""
Quick script to evaluate your existing VQ-VAE models using the new discovery metrics.

Usage:
    python scripts/evaluate_existing_model.py --checkpoint path/to/model.pt --data path/to/data.npz

This will:
1. Load your trained model
2. Extract codes and reconstructions
3. Compute all intrinsic metrics
4. (Optional) Compute extrinsic metrics if labels provided
5. Print summary report
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from flies.evaluation.discovery_metrics import (
    evaluate_discovery_pipeline,
    DiscoveryMetrics
)


def load_checkpoint(checkpoint_path):
    """Load a trained VQ-VAE checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def extract_codes_and_recon(model, data, device='cuda'):
    """
    Extract behavior codes and reconstructions from VQ-VAE.

    Args:
        model: Trained VQ-VAE model
        data: Input keypoint data (n_frames, n_features) or (batch, features, time)
        device: 'cuda' or 'cpu'

    Returns:
        codes: (n_frames,) discrete codes
        reconstruction: (n_frames, n_features)
    """
    model = model.to(device)
    model.eval()

    # Handle different data formats
    if len(data.shape) == 2:
        # (n_frames, n_features) -> (1, n_features, n_frames)
        data = torch.from_numpy(data).float().T.unsqueeze(0)
    elif len(data.shape) == 3:
        # Already (batch, features, time)
        data = torch.from_numpy(data).float()

    data = data.to(device)

    with torch.no_grad():
        # Forward pass
        x_recon, vq_loss, commitment_loss, encodings = model(data)

        # Extract codes (flatten temporal dimension)
        codes = encodings.cpu().numpy().flatten()

        # Extract reconstruction
        # (batch, features, time) -> (n_frames, n_features)
        recon = x_recon[0].transpose(0, 1).cpu().numpy()

    # Original data for comparison
    original = data[0].transpose(0, 1).cpu().numpy()

    return codes, recon, original


def print_metrics_summary(results):
    """Pretty print metrics summary."""
    print("\n" + "=" * 80)
    print("DISCOVERY METRICS SUMMARY")
    print("=" * 80)

    print("\n📊 INTRINSIC METRICS (no labels required)")
    print("-" * 80)

    intrinsic = results['intrinsic']

    print(f"\n  Combined Intrinsic Score:  {intrinsic['combined_score']:.3f}")
    print(f"  (Higher = better overall quality)\n")

    print("  Code Distribution:")
    print(f"    - Entropy (normalized):     {intrinsic['code_entropy_normalized']:.3f}")
    print(f"    - Codebook utilization:     {intrinsic['codebook_utilization']:.1%}")
    print(f"    - Codes used:               {intrinsic['num_codes_used']}/{intrinsic['num_codes_total']}")

    print("\n  Temporal Coherence:")
    print(f"    - Mean bout length:         {intrinsic['mean_bout_length']:.1f} frames")
    print(f"    - Transition rate:          {intrinsic['transition_rate']:.3f}")

    if 'reconstruction_mse' in intrinsic:
        print("\n  Reconstruction Quality:")
        print(f"    - MSE:                      {intrinsic['reconstruction_mse']:.6f}")
        print(f"    - R²:                       {intrinsic['reconstruction_r2']:.3f}")

    if 'cross_split_ari' in intrinsic:
        print("\n  Stability:")
        print(f"    - Cross-split ARI:          {intrinsic['cross_split_ari']:.3f}")

    if 'seed_stability_ari' in intrinsic:
        print(f"    - Seed stability ARI:       {intrinsic['seed_stability_ari']:.3f}")

    if 'extrinsic' in results:
        print("\n" + "-" * 80)
        print("🎯 EXTRINSIC METRICS (validation with ground truth)")
        print("-" * 80)

        extrinsic = results['extrinsic']

        print(f"\n  Rediscovery:")
        print(f"    - ARI with ground truth:    {extrinsic['rediscovery_ari']:.3f}")
        print(f"    - NMI with ground truth:    {extrinsic['rediscovery_nmi']:.3f}")

        if 'classification_accuracy' in extrinsic:
            print(f"\n  Downstream Classification:")
            print(f"    - Accuracy:                 {extrinsic['classification_accuracy']:.1%}")
            print(f"    - F1 (macro):               {extrinsic['classification_f1_macro']:.3f}")
            print(f"    - F1 (weighted):            {extrinsic['classification_f1_weighted']:.3f}")

    if 'forecasting' in results:
        print("\n" + "-" * 80)
        print("🔮 FORECASTING METRICS")
        print("-" * 80)

        forecasting = results['forecasting']

        if 'bigram_next_token_accuracy' in forecasting:
            print(f"\n  Bigram Model:")
            print(f"    - Next token accuracy:      {forecasting['bigram_next_token_accuracy']:.1%}")
            print(f"    - Log-likelihood:           {forecasting['bigram_log_likelihood']:.3f}")

        if 'hmm_log_likelihood' in forecasting:
            print(f"\n  HMM Model:")
            print(f"    - Log-likelihood:           {forecasting['hmm_log_likelihood']:.3f}")
            print(f"    - Perplexity:               {forecasting['hmm_perplexity']:.3f}")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate existing VQ-VAE model')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, help='Path to data file (.npz or .npy)')
    parser.add_argument('--labels', type=str, default=None, help='Path to ground truth labels (optional)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--forecasting', action='store_true', help='Include forecasting metrics')

    args = parser.parse_args()

    # For demo purposes without checkpoint
    if args.checkpoint is None and args.data is None:
        print("Running demo with synthetic data...")
        print("For real evaluation, use: python scripts/evaluate_existing_model.py --checkpoint model.pt --data data.npz\n")

        # Generate synthetic data
        np.random.seed(42)
        n_frames = 1000

        # Simulate discovered codes with realistic properties
        # - ~20 distinct codes
        # - Medium bout length (~15 frames)
        # - Some temporal structure
        codes = np.zeros(n_frames, dtype=int)
        current_code = 0
        pos = 0
        while pos < n_frames:
            bout_length = int(np.random.exponential(15)) + 1
            codes[pos:min(pos + bout_length, n_frames)] = current_code
            current_code = np.random.randint(0, 20)
            pos += bout_length

        # Simulate ground truth for extrinsic evaluation
        labels = np.random.randint(0, 5, size=n_frames)

        # Evaluate
        results = evaluate_discovery_pipeline(
            codes=codes,
            labels=labels,
            include_forecasting=True,
        )

        print_metrics_summary(results)

        print("\n💡 INTERPRETATION GUIDE:")
        print("-" * 80)
        print("Intrinsic Score: 0.7-1.0 = excellent, 0.5-0.7 = good, <0.5 = poor")
        print("Code Entropy: Higher = better balanced usage (max = log(n_codes))")
        print("Bout Length: 10-50 frames typical for behavior analysis")
        print("Transition Rate: 0.05-0.15 = stable, >0.3 = noisy")
        print("ARI: >0.7 = strong match, 0.4-0.7 = moderate, <0.4 = weak")
        print("Classification Accuracy: Compare to baseline (1/n_classes)")
        print("-" * 80)

        return

    # Real evaluation with checkpoint
    # TODO: Load your actual model architecture
    # from flies.models.vqvae import VQVAE
    # model = VQVAE(...)
    # checkpoint = load_checkpoint(args.checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # Load data
    if args.data.endswith('.npz'):
        data_dict = np.load(args.data)
        data = data_dict['data']  # Adjust key as needed
    else:
        data = np.load(args.data)

    # Load labels if provided
    labels = None
    if args.labels:
        labels = np.load(args.labels)

    print("Data shape:", data.shape)
    if labels is not None:
        print("Labels shape:", labels.shape)

    # Extract codes and reconstructions
    # codes, recon, original = extract_codes_and_recon(model, data, device=args.device)

    # For now, use dummy data (replace with actual extraction)
    print("\nWARNING: Replace this with actual model loading!")
    codes = np.random.randint(0, 64, size=len(data))
    recon = data + np.random.randn(*data.shape) * 0.1
    original = data

    # Evaluate
    results = evaluate_discovery_pipeline(
        codes=codes,
        labels=labels,
        reconstruction=recon,
        original_data=original,
        include_forecasting=args.forecasting,
    )

    print_metrics_summary(results)


if __name__ == '__main__':
    main()
