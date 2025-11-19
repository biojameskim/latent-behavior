"""
EXPERIMENT 1: Baseline Evaluation of Existing VQ-VAE Models

Goal: Measure intrinsic metrics on your existing trained VQ-VAEs

This establishes baseline scores before optimization.

Usage:
    python scripts/experiment_1_baseline.py --checkpoint path/to/model.pt --data path/to/data.npz
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from flies.evaluation.discovery_metrics import evaluate_discovery_pipeline


def load_model_and_extract_codes(checkpoint_path, data_path, device='cuda'):
    """
    Load VQ-VAE checkpoint and extract codes.

    TODO: Adapt this to your actual model architecture!
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Loading data: {data_path}")

    # TODO: Replace with your actual model loading code
    # from flies.models.vqvae import VQVAE
    # model = VQVAE(...)
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model = model.to(device)
    # model.eval()

    # Load data
    if data_path.endswith('.npz'):
        data_dict = np.load(data_path)
        # Adjust key based on your data format
        data = data_dict['data'] if 'data' in data_dict else data_dict['keypoints']
    else:
        data = np.load(data_path)

    print(f"Data shape: {data.shape}")

    # TODO: Extract codes and reconstruction from your model
    # with torch.no_grad():
    #     x = torch.from_numpy(data).float().to(device)
    #     x_recon, vq_loss, commitment_loss, codes = model(x)
    #     codes = codes.cpu().numpy().flatten()
    #     reconstruction = x_recon.cpu().numpy()

    # PLACEHOLDER: Generate dummy codes for testing
    print("\n⚠️  WARNING: Using placeholder codes - replace with actual model!")
    n_frames = len(data) if len(data.shape) == 2 else data.shape[0] * data.shape[2]
    codes = np.random.randint(0, 64, size=n_frames)
    reconstruction = data  # placeholder

    return codes, reconstruction, data


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Baseline Evaluation')
    parser.add_argument('--checkpoint', type=str, help='Path to VQ-VAE checkpoint')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output', type=str, default='experiment_1_results.txt', help='Output file')

    args = parser.parse_args()

    print("=" * 80)
    print("EXPERIMENT 1: Baseline Evaluation")
    print("=" * 80)

    # For demo without checkpoint
    if args.checkpoint is None or args.data is None:
        print("\n⚠️  Running in DEMO mode with synthetic data")
        print("For real evaluation: python experiment_1_baseline.py --checkpoint model.pt --data data.npz\n")

        # Generate synthetic data
        np.random.seed(42)
        n_frames = 2000
        n_features = 6

        # Simulate codes with realistic properties
        codes = np.zeros(n_frames, dtype=int)
        pos = 0
        current_code = 0
        while pos < n_frames:
            bout_length = int(np.random.exponential(40)) + 1
            codes[pos:min(pos + bout_length, n_frames)] = current_code
            current_code = np.random.randint(0, 64)
            pos += bout_length

        reconstruction = np.random.randn(n_frames, n_features)
        original = reconstruction + np.random.randn(n_frames, n_features) * 0.1

    else:
        # Load actual model
        codes, reconstruction, original = load_model_and_extract_codes(
            args.checkpoint,
            args.data,
            device=args.device
        )

    print(f"\nData loaded:")
    print(f"  Frames: {len(codes)}")
    print(f"  Unique codes: {len(np.unique(codes))}")

    # Evaluate with discovery metrics
    print("\n" + "=" * 80)
    print("Running Discovery Metrics Evaluation")
    print("=" * 80)

    results = evaluate_discovery_pipeline(
        codes=codes,
        reconstruction=reconstruction,
        original_data=original,
        include_forecasting=True,
    )

    # Print results
    print("\n📊 INTRINSIC METRICS (no labels needed)")
    print("-" * 80)

    intrinsic = results['intrinsic']

    print(f"\n  OVERALL SCORE:           {intrinsic['combined_score']:.3f}")
    print(f"  (Higher = better, range 0-1)\n")

    print("  Code Distribution:")
    print(f"    Entropy (normalized):  {intrinsic['code_entropy_normalized']:.3f}")
    print(f"    Codebook utilization:  {intrinsic['codebook_utilization']:.1%}")
    print(f"    Codes used:            {intrinsic['num_codes_used']}/{intrinsic['num_codes_total']}")

    print("\n  Temporal Coherence:")
    print(f"    Mean bout length:      {intrinsic['mean_bout_length']:.1f} frames")
    print(f"    Transition rate:       {intrinsic['transition_rate']:.3f}")

    if 'reconstruction_mse' in intrinsic:
        print("\n  Reconstruction Quality:")
        print(f"    MSE:                   {intrinsic['reconstruction_mse']:.6f}")
        print(f"    R²:                    {intrinsic['reconstruction_r2']:.3f}")

    if 'forecasting' in results:
        print("\n🔮 FORECASTING METRICS")
        print("-" * 80)
        forecasting = results['forecasting']

        if 'bigram_next_token_accuracy' in forecasting:
            print(f"\n  Next-token accuracy:   {forecasting['bigram_next_token_accuracy']:.1%}")
            print(f"  Log-likelihood:        {forecasting['bigram_log_likelihood']:.3f}")

    # Save results
    output_file = args.output
    with open(output_file, 'w') as f:
        f.write("EXPERIMENT 1: Baseline Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint or 'DEMO'}\n")
        f.write(f"Data: {args.data or 'DEMO'}\n\n")
        f.write("INTRINSIC METRICS:\n")
        f.write("-" * 80 + "\n")
        for key, value in intrinsic.items():
            f.write(f"  {key}: {value}\n")

        if 'forecasting' in results:
            f.write("\nFORECASTING METRICS:\n")
            f.write("-" * 80 + "\n")
            for key, value in results['forecasting'].items():
                f.write(f"  {key}: {value}\n")

    print(f"\n✅ Results saved to {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("\nIntrinsic Score: 0.7-1.0 = excellent, 0.5-0.7 = good, <0.5 = poor")
    print("Code Entropy:    Higher = more balanced usage")
    print("Bout Length:     10-50 frames typical for behaviors")
    print("Transition Rate: 0.05-0.15 = stable, >0.3 = noisy")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Record this baseline score")
    print("2. Run Experiment 2: Random search baseline")
    print("3. Run Experiment 3: Agent optimization")
    print("4. Compare: Can agent beat this baseline?")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
