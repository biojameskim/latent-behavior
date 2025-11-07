"""
Analyze and compare different VQ-VAE training runs.

Usage:
    python analyze_runs.py outputs/group_norm_run outputs/simple_init_fix_run
"""

import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load_checkpoint(path):
    """Load checkpoint and extract metrics."""
    ckpt = torch.load(path, map_location='cpu')
    return {
        'epoch': ckpt['epoch'],
        'metrics': ckpt.get('train_metrics', {}),
        'args': ckpt.get('args', {}),
        'model_type': ckpt.get('model_type', 'unknown')
    }

def analyze_run(run_dir):
    """Analyze all checkpoints in a run directory."""
    run_dir = Path(run_dir)

    # Find checkpoint files and sort numerically by epoch number
    checkpoints = list(run_dir.glob('checkpoint_epoch_*.pt'))

    # Sort by epoch number (not alphabetically!)
    def extract_epoch(path):
        # Extract number from "checkpoint_epoch_10.pt" -> 10
        return int(path.stem.split('_')[-1])

    checkpoints = sorted(checkpoints, key=extract_epoch)

    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return None

    epochs = []
    losses = []
    recon_losses = []
    vq_losses = []
    perplexities = []

    for ckpt_path in checkpoints:
        data = load_checkpoint(ckpt_path)
        epochs.append(data['epoch'])
        metrics = data['metrics']
        losses.append(metrics.get('loss', 0))
        recon_losses.append(metrics.get('recon_loss', 0))
        vq_losses.append(metrics.get('vq_loss', 0))
        perplexities.append(metrics.get('perplexity', 0))

    # Get args from last checkpoint
    last_data = load_checkpoint(checkpoints[-1])

    return {
        'name': run_dir.name,
        'epochs': epochs,
        'loss': losses,
        'recon_loss': recon_losses,
        'vq_loss': vq_losses,
        'perplexity': perplexities,
        'args': last_data['args'],
        'model_type': last_data['model_type'],
        'num_embeddings': last_data['args'].get('num_embeddings', '?'),
        'embedding_dim': last_data['args'].get('embedding_dim', '?'),
        'commitment_cost': last_data['args'].get('commitment_cost', '?')
    }

def print_comparison(runs):
    """Print comparison table."""
    print("\n" + "="*100)
    print("TRAINING RUN COMPARISON")
    print("="*100)

    # Header
    print(f"{'Run':<30} {'Model':<12} {'Codes':<8} {'Embed':<8} {'β':<6} {'Final Loss':<12} {'VQ Loss':<12} {'Perplexity':<15}")
    print("-"*100)

    for run in runs:
        final_loss = run['loss'][-1] if run['loss'] else 0
        final_vq = run['vq_loss'][-1] if run['vq_loss'] else 0
        final_perp = run['perplexity'][-1] if run['perplexity'] else 0
        num_codes = run['num_embeddings']

        perp_pct = f"{final_perp:.1f}/{num_codes} ({100*final_perp/num_codes:.0f}%)"

        print(f"{run['name']:<30} {run['model_type']:<12} {num_codes:<8} {run['embedding_dim']:<8} "
              f"{run['commitment_cost']:<6} {final_loss:<12.2f} {final_vq:<12.2f} {perp_pct:<15}")

    print("="*100)
    print("\nKEY METRICS INTERPRETATION:")
    print("  VQ Loss:    < 10 = Good, 10-50 = Acceptable, > 50 = Poor")
    print("  Perplexity: > 60% of codes = Good, 30-60% = Acceptable, < 30% = Poor utilization")
    print("  Total Loss: Lower is better (but watch for overfitting)")
    print()

def plot_comparison(runs, output_file='comparison.png'):
    """Plot comparison of training runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Total Loss
    ax = axes[0, 0]
    for run in runs:
        ax.plot(run['epochs'], run['loss'], marker='o', label=run['name'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss (Recon + VQ)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: VQ Loss
    ax = axes[0, 1]
    for run in runs:
        ax.plot(run['epochs'], run['vq_loss'], marker='o', label=run['name'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('VQ Loss')
    ax.set_title('Vector Quantization Loss')
    ax.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Target < 10')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Reconstruction Loss
    ax = axes[1, 0]
    for run in runs:
        ax.plot(run['epochs'], run['recon_loss'], marker='o', label=run['name'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss (MSE)')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Perplexity (Codebook Utilization)
    ax = axes[1, 1]
    for run in runs:
        num_codes = run['num_embeddings']
        # Plot as percentage
        perp_pct = [100 * p / num_codes for p in run['perplexity']]
        ax.plot(run['epochs'], perp_pct, marker='o', label=f"{run['name']} ({num_codes} codes)")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Codebook Utilization (%)')
    ax.set_title('Codebook Utilization (Perplexity / Num Codes)')
    ax.axhline(y=60, color='g', linestyle='--', alpha=0.5, label='Target > 60%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

def print_recommendations(runs):
    """Print recommendations based on results."""
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)

    # Find best run
    best_run = min(runs, key=lambda r: r['loss'][-1])

    print(f"\n✅ BEST OVERALL: {best_run['name']}")
    print(f"   - Final loss: {best_run['loss'][-1]:.2f}")
    print(f"   - VQ loss: {best_run['vq_loss'][-1]:.2f}")
    print(f"   - Perplexity: {best_run['perplexity'][-1]:.1f} / {best_run['num_embeddings']}")

    # Check for issues
    issues = []
    for run in runs:
        final_vq = run['vq_loss'][-1]
        final_perp = run['perplexity'][-1]
        num_codes = run['num_embeddings']
        utilization = final_perp / num_codes

        if final_vq > 50:
            issues.append(f"❌ {run['name']}: VQ loss very high ({final_vq:.1f}) - scale mismatch not fixed")
        elif final_vq > 10:
            issues.append(f"⚠️  {run['name']}: VQ loss moderate ({final_vq:.1f}) - could be better")

        if utilization < 0.3:
            issues.append(f"⚠️  {run['name']}: Low codebook utilization ({100*utilization:.0f}%) - consider reducing num_embeddings")

    if issues:
        print("\n⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"   {issue}")

    print("\n📝 NEXT STEPS:")

    # Specific recommendations
    best_vq = best_run['vq_loss'][-1]
    best_util = best_run['perplexity'][-1] / best_run['num_embeddings']

    if best_vq < 10 and best_util > 0.6:
        print("   ✅ Your best run looks great! Continue training or try fine-tuning:")
        print("      - Train for more epochs to see if perplexity improves further")
        print("      - Try visualizing the learned codebook embeddings")
    elif best_vq < 10 and best_util < 0.3:
        print("   ⚠️  VQ loss is good but codebook underutilized. Try:")
        print(f"      - Reduce num_embeddings: {best_run['num_embeddings']} → {int(best_run['num_embeddings'] * 0.5)}")
        print(f"      - Increase embedding_dim: {best_run['embedding_dim']} → {int(best_run['embedding_dim'] * 1.5)}")
        print("      - Train longer (codes may activate over time)")
    elif best_vq > 10:
        print("   ❌ VQ loss still too high. Use GroupNorm model:")
        print("      - python train_unified.py --model_type groupnorm")
        print("      - Or use train_optimized.sh script")

    print("\n   📊 Use train_optimized.sh for recommended hyperparameters:")
    print("      ./train_optimized.sh              # GroupNorm with optimized settings")
    print("      ./train_optimized.sh simple       # Simple model (not recommended)")

    print("="*100 + "\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_runs.py <run_dir1> [run_dir2] [run_dir3] ...")
        print("\nExample:")
        print("  python analyze_runs.py outputs/group_norm_run outputs/simple_init_fix_run")
        sys.exit(1)

    run_dirs = sys.argv[1:]

    print("Analyzing training runs...")
    runs = []
    for run_dir in run_dirs:
        print(f"  Loading {run_dir}...")
        run_data = analyze_run(run_dir)
        if run_data:
            runs.append(run_data)

    if not runs:
        print("No valid runs found!")
        sys.exit(1)

    print_comparison(runs)
    plot_comparison(runs)
    print_recommendations(runs)
