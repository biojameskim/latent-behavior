"""
EXPERIMENT 2: Random Search Baseline

Goal: Establish baseline by training VQ-VAEs with random hyperparameters

This answers: What's the score distribution from random hyperparameter selection?

Usage:
    python scripts/experiment_2_random_search.py --data path/to/data.npz --n_trials 10
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import json
import random
from datetime import datetime
from flies.evaluation.discovery_metrics import evaluate_discovery_pipeline


def sample_random_config(search_space):
    """Sample random hyperparameter configuration."""
    config = {}
    for param_name, options in search_space.items():
        config[param_name] = random.choice(options)
    return config


def train_vqvae_with_config(config, data, max_epochs=20):
    """
    Train VQ-VAE with given config.

    TODO: Replace with your actual training code!

    Args:
        config: Dict of hyperparameters
        data: Training data
        max_epochs: Max training epochs (keep short for search)

    Returns:
        codes: Discrete behavior codes
        reconstruction: Reconstructed data
    """
    print(f"\n  Training with config: {config}")

    # TODO: Replace with actual VQ-VAE training
    # from flies.models.vqvae import VQVAE
    # model = VQVAE(
    #     input_dim=data.shape[1],
    #     embedding_dim=config['embedding_dim'],
    #     num_embeddings=config['codebook_size'],
    #     commitment_cost=config['commitment_cost'],
    # )
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    #
    # for epoch in range(max_epochs):
    #     loss = train_epoch(model, data, optimizer)
    #     print(f"    Epoch {epoch}: loss={loss:.4f}")
    #
    # codes, reconstruction = extract_codes(model, data)

    # PLACEHOLDER: Generate dummy results
    print("  ⚠️  Using placeholder training - replace with actual VQ-VAE training!")

    n_frames = len(data) if len(data.shape) == 2 else data.shape[0]
    codebook_size = config['codebook_size']

    # Simulate codes with realistic temporal structure
    codes = np.zeros(n_frames, dtype=int)
    pos = 0
    while pos < n_frames:
        bout_length = int(np.random.exponential(30)) + 1
        code = np.random.randint(0, codebook_size)
        codes[pos:min(pos + bout_length, n_frames)] = code
        pos += bout_length

    # Simulate reconstruction (with quality depending on embedding_dim)
    noise_scale = 0.2 / (config['embedding_dim'] / 32)  # Better with higher dim
    reconstruction = data + np.random.randn(*data.shape) * noise_scale

    return codes, reconstruction


def main():
    parser = argparse.ArgumentParser(description='Experiment 2: Random Search')
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of random trials')
    parser.add_argument('--max_epochs', type=int, default=20, help='Max epochs per trial')
    parser.add_argument('--output', type=str, default='experiment_2_results.json', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("EXPERIMENT 2: Random Search Baseline")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Trials: {args.n_trials}")
    print(f"  Max epochs per trial: {args.max_epochs}")
    print(f"  Random seed: {args.seed}")

    # Define search space
    search_space = {
        'codebook_size': [32, 64, 128, 256],
        'embedding_dim': [32, 64, 128, 256],
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'commitment_cost': [0.1, 0.25, 0.5, 1.0],
    }

    print(f"\nSearch space:")
    for param, options in search_space.items():
        print(f"  {param}: {options}")

    # Load data
    if args.data is None or not Path(args.data).exists():
        print("\n⚠️  No data provided, using synthetic data for demo")
        # Generate synthetic data
        n_frames = 2000
        n_features = 6
        data = np.random.randn(n_frames, n_features)
    else:
        print(f"\nLoading data: {args.data}")
        if args.data.endswith('.npz'):
            data_dict = np.load(args.data)
            data = data_dict['data'] if 'data' in data_dict else data_dict['keypoints']
        else:
            data = np.load(args.data)

    print(f"Data shape: {data.shape}")

    # Run random search
    print("\n" + "=" * 80)
    print("Running Random Search")
    print("=" * 80)

    results = []

    for trial in range(args.n_trials):
        print(f"\n--- Trial {trial + 1}/{args.n_trials} ---")

        # Sample random config
        config = sample_random_config(search_space)

        # Train model
        codes, reconstruction = train_vqvae_with_config(
            config,
            data,
            max_epochs=args.max_epochs
        )

        # Evaluate
        metrics = evaluate_discovery_pipeline(
            codes=codes,
            reconstruction=reconstruction,
            original_data=data,
            include_forecasting=True,
        )

        intrinsic_score = metrics['intrinsic']['combined_score']

        print(f"\n  Results:")
        print(f"    Intrinsic score:     {intrinsic_score:.3f}")
        print(f"    Code entropy:        {metrics['intrinsic']['code_entropy_normalized']:.3f}")
        print(f"    Mean bout length:    {metrics['intrinsic']['mean_bout_length']:.1f}")
        print(f"    Reconstruction R²:   {metrics['intrinsic'].get('reconstruction_r2', 0):.3f}")

        # Store results
        results.append({
            'trial': trial + 1,
            'config': config,
            'intrinsic_score': intrinsic_score,
            'metrics': {
                'code_entropy': metrics['intrinsic']['code_entropy_normalized'],
                'bout_length': metrics['intrinsic']['mean_bout_length'],
                'reconstruction_r2': metrics['intrinsic'].get('reconstruction_r2', 0),
                'codebook_utilization': metrics['intrinsic']['codebook_utilization'],
            }
        })

    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    scores = [r['intrinsic_score'] for r in results]

    print(f"\nIntrinsic Score Statistics:")
    print(f"  Mean:   {np.mean(scores):.3f}")
    print(f"  Std:    {np.std(scores):.3f}")
    print(f"  Min:    {np.min(scores):.3f}")
    print(f"  Max:    {np.max(scores):.3f}")
    print(f"  Median: {np.median(scores):.3f}")

    # Best config
    best = max(results, key=lambda x: x['intrinsic_score'])
    print(f"\n🏆 Best Configuration (Trial {best['trial']}):")
    print(f"  Score: {best['intrinsic_score']:.3f}")
    print(f"  Config:")
    for param, value in best['config'].items():
        print(f"    {param}: {value}")

    print(f"\n  Metrics:")
    for metric, value in best['metrics'].items():
        print(f"    {metric}: {value:.3f}")

    # Worst config
    worst = min(results, key=lambda x: x['intrinsic_score'])
    print(f"\n📉 Worst Configuration (Trial {worst['trial']}):")
    print(f"  Score: {worst['intrinsic_score']:.3f}")
    print(f"  Config:")
    for param, value in worst['config'].items():
        print(f"    {param}: {value}")

    # Save results
    output = {
        'experiment': 'random_search',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_trials': args.n_trials,
            'max_epochs': args.max_epochs,
            'search_space': search_space,
            'seed': args.seed,
        },
        'results': results,
        'summary': {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'best_config': best['config'],
            'best_score': best['intrinsic_score'],
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to {args.output}")

    # Next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"\n📊 Random Search Baseline Established:")
    print(f"   Best score:  {best['intrinsic_score']:.3f}")
    print(f"   Mean score:  {np.mean(scores):.3f}")
    print(f"\n🤖 Now run Experiment 3 (Agent optimization)")
    print(f"   Goal: Beat the random search best score of {best['intrinsic_score']:.3f}")
    print(f"\n   python scripts/experiment_3_agent_search.py --data {args.data or 'data.npz'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
