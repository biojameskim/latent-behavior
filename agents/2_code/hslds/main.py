"""
Main execution script for HSLDS behavior discovery pipeline

Usage:
    python main.py --data_path <path_to_mabe22_subset_for_claude.npz> \
                   --epochs 50 \
                   --batch_size 16 \
                   --n_states 12 \
                   --device cuda
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from model import DiscoveryPipeline
from training import train_pipeline
from evaluation import IntrinsicEvaluator, ExtrinsicEvaluator


def load_data(data_path):
    """
    Load the MABe 2022 dataset

    Returns:
        trajectories: (N, Time, Features) numpy array
        labels: (N, Time) numpy array
        metadata: dict
    """
    data = np.load(data_path, allow_pickle=True)

    trajectories = data['trajectories']  # (50, 300, 48)
    labels = data['labels']  # (50, 300)
    keypoint_vocab = data['keypoint_vocabulary']
    metadata = data['metadata'].item() if 'metadata' in data else {}

    print(f"Loaded dataset:")
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Keypoints: {len(keypoint_vocab)}")
    print(f"  Number of unique behaviors: {len(np.unique(labels))}")

    return trajectories, labels, metadata


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training loss curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Total loss
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    # Reconstruction loss
    axes[0, 1].plot(history['reconstruction'])
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].grid(True)

    # Codebook loss
    axes[1, 0].plot(history['codebook'])
    axes[1, 0].set_title('Codebook Utilization Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)

    # Temporal loss
    axes[1, 1].plot(history['temporal'])
    axes[1, 1].set_title('Temporal Coherence Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Change Rate')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training history saved to {save_path}")
    plt.close()


def visualize_codes(model, data, labels, save_path='code_visualization.png'):
    """
    Visualize discovered codes vs ground truth labels
    """
    model.eval()
    with torch.no_grad():
        processed = model.preprocessor.preprocess(data)
        codes = model.encode(processed).cpu().numpy()

    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # Plot first 5 sequences
    n_plot = min(5, data.shape[0])
    fig, axes = plt.subplots(n_plot, 2, figsize=(14, 3*n_plot))

    if n_plot == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_plot):
        # Discovered codes
        axes[i, 0].plot(codes[i], linewidth=0.5)
        axes[i, 0].set_title(f'Sequence {i} - Discovered Codes')
        axes[i, 0].set_ylabel('State ID')
        axes[i, 0].set_xlabel('Frame')
        axes[i, 0].grid(True, alpha=0.3)

        # Ground truth labels
        axes[i, 1].plot(labels_np[i], linewidth=0.5)
        axes[i, 1].set_title(f'Sequence {i} - Ground Truth Labels')
        axes[i, 1].set_ylabel('Behavior ID')
        axes[i, 1].set_xlabel('Frame')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Code visualization saved to {save_path}")
    plt.close()


def print_results(intrinsic_results, extrinsic_results):
    """
    Pretty print evaluation results
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print("\n--- INTRINSIC METRICS (No Ground Truth) ---")
    print(f"  Reconstruction MSE:      {intrinsic_results['reconstruction_mse']:.6f}")
    print(f"  Codebook Usage:          {intrinsic_results['codebook_usage']:.2%}")
    print(f"  Mean Bout Length:        {intrinsic_results['mean_bout_length']:.2f} frames")
    print(f"  MMD Score:               {intrinsic_results['mmd_score']:.6f}")
    print(f"  ACF Error:               {intrinsic_results['acf_error']:.6f}")
    print(f"  Discovery Score:         {intrinsic_results['discovery_score']:.4f}")

    print("\n--- EXTRINSIC METRICS (With Ground Truth) ---")
    print(f"  Adjusted Rand Index:     {extrinsic_results['ari']:.4f}")
    print(f"  Normalized Mutual Info:  {extrinsic_results['nmi']:.4f}")
    print(f"  Homogeneity:             {extrinsic_results['homogeneity']:.4f}")
    print(f"  Completeness:            {extrinsic_results['completeness']:.4f}")
    print(f"  V-Measure:               {extrinsic_results['v_measure']:.4f}")

    print("\n" + "="*60)

    # Interpretation guide
    print("\nINTERPRETATION GUIDE:")
    print("  - Reconstruction MSE:  Lower is better (how well we reconstruct)")
    print("  - Codebook Usage:      Higher is better (are all states used?)")
    print("  - MMD Score:           Lower is better (synthetic ≈ real distribution)")
    print("  - ACF Error:           Lower is better (synthetic ≈ real dynamics)")
    print("  - ARI:                 Higher is better, range [-1, 1], >0.5 is good")
    print("  - Discovery Score:     Higher is better (combined quality metric)")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='HSLDS Behavior Discovery Pipeline')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to mabe22_subset_for_claude.npz')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--n_states', type=int, default=12,
                        help='Number of discrete behavioral states')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimension')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    trajectories, labels, metadata = load_data(args.data_path)

    # Convert to torch tensors
    trajectories_tensor = torch.from_numpy(trajectories).float()
    labels_tensor = torch.from_numpy(labels).long()

    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    print(f"Architecture: Hierarchical Switching Linear Dynamical System")
    print(f"  Input dimension:  {trajectories.shape[-1]}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Number of states: {args.n_states}")

    model = DiscoveryPipeline(
        input_dim=trajectories.shape[-1],  # 48
        latent_dim=args.latent_dim,
        n_states=args.n_states,
        output_dim=121  # After preprocessing: 48 + 48 + 25
    )

    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    model, history = train_pipeline(
        model,
        trajectories_tensor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    # Plot training history
    plot_training_history(history, save_path=output_dir / 'training_history.png')

    # Intrinsic evaluation
    print("\n" + "="*60)
    print("INTRINSIC EVALUATION (No Ground Truth)")
    print("="*60)
    intrinsic_evaluator = IntrinsicEvaluator()
    intrinsic_results = intrinsic_evaluator.evaluate_all(
        model,
        trajectories_tensor,
        device=device
    )

    # Extrinsic evaluation
    print("\n" + "="*60)
    print("EXTRINSIC EVALUATION (With Ground Truth)")
    print("="*60)
    extrinsic_evaluator = ExtrinsicEvaluator()
    extrinsic_results = extrinsic_evaluator.evaluate_with_labels(
        model,
        trajectories_tensor,
        labels_tensor,
        device=device
    )

    # Visualize codes
    visualize_codes(
        model,
        trajectories_tensor[:5].to(device),
        labels_tensor[:5],
        save_path=output_dir / 'code_visualization.png'
    )

    # Print results
    print_results(intrinsic_results, extrinsic_results)

    # Save results
    results = {
        'intrinsic': intrinsic_results,
        'extrinsic': extrinsic_results,
        'history': history,
        'args': vars(args)
    }

    np.save(output_dir / 'results.npy', results)
    print(f"\nResults saved to {output_dir / 'results.npy'}")

    # Save model
    torch.save(model.state_dict(), output_dir / 'model.pth')
    print(f"Model saved to {output_dir / 'model.pth'}")

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
