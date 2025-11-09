"""
Analyze and compare VQ-VAE quantization methods.

This script reads training results from all methods and creates
comprehensive comparison plots and tables.

Usage:
    python analyze_comparison.py --results_dir outputs/group_norm_v5
    python analyze_comparison.py --results_dir outputs/group_norm_v5 --methods vq fsq rvq
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_training_history(method_dir: Path) -> Dict:
    """Load training history for a method."""
    history_file = method_dir / 'training_history.json'
    if not history_file.exists():
        return None

    with open(history_file, 'r') as f:
        history = json.load(f)

    return history


def load_all_results(results_dir: Path, methods: Optional[List[str]] = None) -> Dict:
    """
    Load results for all methods.

    Args:
        results_dir: Directory containing method subdirectories
        methods: Optional list of method names to load. If None, loads all found.

    Returns:
        Dict mapping method name to training history
    """
    results_dir = Path(results_dir)
    all_results = {}

    # Auto-detect methods if not specified
    if methods is None:
        methods = []
        for subdir in results_dir.iterdir():
            if subdir.is_dir() and (subdir / 'training_history.json').exists():
                methods.append(subdir.name)

    print(f"Loading results for methods: {methods}")

    for method in methods:
        method_dir = results_dir / method
        if not method_dir.exists():
            print(f"Warning: Directory not found for method '{method}'")
            continue

        history = load_training_history(method_dir)
        if history:
            all_results[method] = history
            print(f"  ✓ Loaded {method}: {len(history.get('train_loss', []))} epochs")
        else:
            print(f"  ✗ No history found for {method}")

    return all_results


def plot_loss_curves(all_results: Dict, save_path: Path):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Define colors for each method
    colors = {
        'vq': '#1f77b4',
        'vq_improved': '#ff7f0e',
        'fsq': '#2ca02c',
        'rvq': '#d62728',
        'lfq': '#9467bd'
    }

    metrics = [
        ('train_loss', 'Training Loss'),
        ('val_loss', 'Validation Loss'),
        ('train_recon', 'Training Reconstruction Loss'),
        ('val_recon', 'Validation Reconstruction Loss')
    ]

    for idx, (metric_key, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for method, history in all_results.items():
            if metric_key in history:
                epochs = np.arange(1, len(history[metric_key]) + 1)
                color = colors.get(method, '#333333')
                ax.plot(epochs, history[metric_key],
                       label=method.upper(), color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'loss_curves.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path / 'loss_curves.png'}")
    plt.close()


def plot_vq_loss_and_perplexity(all_results: Dict, save_path: Path):
    """Plot VQ loss and perplexity curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    colors = {
        'vq': '#1f77b4',
        'vq_improved': '#ff7f0e',
        'fsq': '#2ca02c',
        'rvq': '#d62728',
        'lfq': '#9467bd'
    }

    # VQ Loss
    ax = axes[0]
    for method, history in all_results.items():
        if 'train_vq' in history:
            epochs = np.arange(1, len(history['train_vq']) + 1)
            color = colors.get(method, '#333333')
            ax.plot(epochs, history['train_vq'],
                   label=f'{method.upper()} (train)',
                   color=color, linewidth=2, alpha=0.8)

        # Check both 'val_vq' and 'val_vq_loss' for backward compatibility
        val_vq = history.get('val_vq') or history.get('val_vq_loss')
        if val_vq:
            epochs = np.arange(1, len(val_vq) + 1)
            color = colors.get(method, '#333333')
            ax.plot(epochs, val_vq,
                   label=f'{method.upper()} (val)',
                   color=color, linewidth=2, alpha=0.4, linestyle='--')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('VQ Loss', fontsize=12)
    ax.set_title('VQ Loss (Quantization Quality)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Perplexity
    ax = axes[1]
    for method, history in all_results.items():
        if 'train_perp' in history:
            epochs = np.arange(1, len(history['train_perp']) + 1)
            color = colors.get(method, '#333333')
            ax.plot(epochs, history['train_perp'],
                   label=f'{method.upper()} (train)',
                   color=color, linewidth=2, alpha=0.8)

        # Check both 'val_perp' and 'val_perplexity' for backward compatibility
        val_perp = history.get('val_perp') or history.get('val_perplexity')
        if val_perp:
            epochs = np.arange(1, len(val_perp) + 1)
            color = colors.get(method, '#333333')
            ax.plot(epochs, val_perp,
                   label=f'{method.upper()} (val)',
                   color=color, linewidth=2, alpha=0.4, linestyle='--')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Codebook Perplexity (Usage)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'vq_metrics.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path / 'vq_metrics.png'}")
    plt.close()


def plot_final_comparison_bars(all_results: Dict, save_path: Path):
    """Create bar chart comparing final metrics."""
    methods = list(all_results.keys())

    # Extract final epoch metrics
    final_metrics = {
        'val_loss': [],
        'val_recon': [],
        'val_vq': [],
        'val_perp': []
    }

    for method in methods:
        history = all_results[method]
        # Support both naming conventions
        for metric in final_metrics.keys():
            # Check for both old and new key names
            if metric == 'val_recon':
                data = history.get('val_recon') or history.get('val_recon_loss')
            elif metric == 'val_vq':
                data = history.get('val_vq') or history.get('val_vq_loss')
            elif metric == 'val_perp':
                data = history.get('val_perp') or history.get('val_perplexity')
            else:
                data = history.get(metric)

            if data and len(data) > 0:
                final_metrics[metric].append(data[-1])
            else:
                final_metrics[metric].append(np.nan)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    titles = {
        'val_loss': 'Final Validation Loss (Lower is Better)',
        'val_recon': 'Final Reconstruction Loss (Lower is Better)',
        'val_vq': 'Final VQ Loss (Lower is Better)',
        'val_perp': 'Final Perplexity (Higher is Better)'
    }

    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (metric, title) in enumerate(titles.items()):
        ax = axes[idx // 2, idx % 2]

        values = final_metrics[metric]
        valid_mask = ~np.isnan(values)

        bars = ax.bar(
            [m.upper() for i, m in enumerate(methods) if valid_mask[i]],
            [v for v in values if not np.isnan(v)],
            color=[colors_list[i] for i in range(len(methods)) if valid_mask[i]],
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight best method
        if metric == 'val_perp':
            best_idx = np.nanargmax(values)
        else:
            best_idx = np.nanargmin(values)

        if not np.isnan(values[best_idx]):
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(save_path / 'final_comparison_bars.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path / 'final_comparison_bars.png'}")
    plt.close()


def create_summary_table(all_results: Dict, save_path: Path):
    """Create a summary table of results."""

    # Prepare data
    summary_data = []

    for method, history in all_results.items():
        row = {'Method': method.upper()}

        # Final metrics (support both naming conventions)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            row['Final Val Loss'] = f"{history['val_loss'][-1]:.4f}"
        else:
            row['Final Val Loss'] = 'N/A'

        # Check both 'val_recon' and 'val_recon_loss' for backward compatibility
        val_recon = history.get('val_recon') or history.get('val_recon_loss')
        if val_recon and len(val_recon) > 0:
            row['Final Recon Loss'] = f"{val_recon[-1]:.4f}"
        else:
            row['Final Recon Loss'] = 'N/A'

        # Check both 'val_vq' and 'val_vq_loss' for backward compatibility
        val_vq = history.get('val_vq') or history.get('val_vq_loss')
        if val_vq and len(val_vq) > 0:
            row['Final VQ Loss'] = f"{val_vq[-1]:.4f}"
        else:
            row['Final VQ Loss'] = 'N/A'

        # Check both 'val_perp' and 'val_perplexity' for backward compatibility
        val_perp = history.get('val_perp') or history.get('val_perplexity')
        if val_perp and len(val_perp) > 0:
            row['Final Perplexity'] = f"{val_perp[-1]:.2f}"
        else:
            row['Final Perplexity'] = 'N/A'

        # Best metrics
        if 'val_loss' in history and len(history['val_loss']) > 0:
            best_val_loss = min(history['val_loss'])
            best_epoch = np.argmin(history['val_loss']) + 1
            row['Best Val Loss'] = f"{best_val_loss:.4f} (epoch {best_epoch})"
        else:
            row['Best Val Loss'] = 'N/A'

        # Training epochs
        if 'train_loss' in history:
            row['Epochs'] = len(history['train_loss'])
        else:
            row['Epochs'] = 'N/A'

        summary_data.append(row)

    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, max(3, len(summary_data) * 0.6)))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table_data = []
    headers = ['Method', 'Epochs', 'Final Val Loss', 'Final Recon Loss',
               'Final VQ Loss', 'Final Perplexity', 'Best Val Loss']

    for row in summary_data:
        table_data.append([
            row.get('Method', ''),
            row.get('Epochs', ''),
            row.get('Final Val Loss', ''),
            row.get('Final Recon Loss', ''),
            row.get('Final VQ Loss', ''),
            row.get('Final Perplexity', ''),
            row.get('Best Val Loss', '')
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.12, 0.08, 0.15, 0.15, 0.13, 0.15, 0.22])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    plt.title('Quantization Methods Comparison Summary',
             fontsize=16, fontweight='bold', pad=20)

    plt.savefig(save_path / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path / 'summary_table.png'}")
    plt.close()

    # Also save as text
    with open(save_path / 'summary_table.txt', 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("QUANTIZATION METHODS COMPARISON SUMMARY\n")
        f.write("=" * 120 + "\n\n")

        # Write header
        f.write(f"{'Method':<15} {'Epochs':<8} {'Final Val':<12} {'Final Recon':<13} "
               f"{'Final VQ':<11} {'Final Perp':<13} {'Best Val Loss':<25}\n")
        f.write("-" * 120 + "\n")

        # Write data
        for row in summary_data:
            f.write(f"{row.get('Method', ''):<15} "
                   f"{str(row.get('Epochs', '')):<8} "
                   f"{row.get('Final Val Loss', ''):<12} "
                   f"{row.get('Final Recon Loss', ''):<13} "
                   f"{row.get('Final VQ Loss', ''):<11} "
                   f"{row.get('Final Perplexity', ''):<13} "
                   f"{row.get('Best Val Loss', ''):<25}\n")

        f.write("=" * 120 + "\n")

    print(f"  Saved: {save_path / 'summary_table.txt'}")


def plot_learning_curves_grid(all_results: Dict, save_path: Path):
    """Create a grid of learning curves for each method."""
    n_methods = len(all_results)
    if n_methods == 0:
        return

    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 5 * n_methods))

    if n_methods == 1:
        axes = axes.reshape(1, -1)

    colors = {
        'vq': '#1f77b4',
        'vq_improved': '#ff7f0e',
        'fsq': '#2ca02c',
        'rvq': '#d62728',
        'lfq': '#9467bd'
    }

    for idx, (method, history) in enumerate(all_results.items()):
        color = colors.get(method, '#333333')

        # Loss curves
        ax = axes[idx, 0]
        if 'train_loss' in history:
            epochs = np.arange(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'],
                   label='Train', color=color, linewidth=2, alpha=0.8)
        if 'val_loss' in history:
            epochs = np.arange(1, len(history['val_loss']) + 1)
            ax.plot(epochs, history['val_loss'],
                   label='Val', color=color, linewidth=2, alpha=0.5, linestyle='--')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title(f'{method.upper()} - Loss Curves', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Reconstruction loss
        ax = axes[idx, 1]
        if 'train_recon' in history:
            epochs = np.arange(1, len(history['train_recon']) + 1)
            ax.plot(epochs, history['train_recon'],
                   label='Train Recon', color=color, linewidth=2, alpha=0.8)
        # Check both naming conventions
        val_recon = history.get('val_recon') or history.get('val_recon_loss')
        if val_recon:
            epochs = np.arange(1, len(val_recon) + 1)
            ax.plot(epochs, val_recon,
                   label='Val Recon', color=color, linewidth=2, alpha=0.5, linestyle='--')

        if 'train_vq' in history:
            epochs = np.arange(1, len(history['train_vq']) + 1)
            ax.plot(epochs, history['train_vq'],
                   label='Train VQ', color='red', linewidth=2, alpha=0.6, linestyle='-.')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{method.upper()} - Loss Components', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'learning_curves_grid.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path / 'learning_curves_grid.png'}")
    plt.close()


def main(args):
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"\n{'='*70}")
    print(f"ANALYZING QUANTIZATION METHODS COMPARISON")
    print(f"{'='*70}\n")
    print(f"Results directory: {results_dir}")

    # Load results
    print("\nLoading training results...")
    all_results = load_all_results(results_dir, args.methods)

    if not all_results:
        print("\nNo results found! Make sure training has completed.")
        return

    print(f"\nFound {len(all_results)} methods: {', '.join(all_results.keys())}")

    # Create output directory for plots
    plots_dir = results_dir / 'analysis_plots'
    plots_dir.mkdir(exist_ok=True)
    print(f"\nSaving plots to: {plots_dir}")

    # Generate plots
    print("\nGenerating plots...")

    print("\n1. Loss curves comparison...")
    plot_loss_curves(all_results, plots_dir)

    print("\n2. VQ metrics (loss and perplexity)...")
    plot_vq_loss_and_perplexity(all_results, plots_dir)

    print("\n3. Final metrics comparison bars...")
    plot_final_comparison_bars(all_results, plots_dir)

    print("\n4. Learning curves grid...")
    plot_learning_curves_grid(all_results, plots_dir)

    print("\n5. Summary table...")
    create_summary_table(all_results, plots_dir)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll plots saved to: {plots_dir}")
    print("\nGenerated files:")
    print("  - loss_curves.png: Training/validation loss comparison")
    print("  - vq_metrics.png: VQ loss and perplexity curves")
    print("  - final_comparison_bars.png: Bar chart of final metrics")
    print("  - learning_curves_grid.png: Individual method curves")
    print("  - summary_table.png: Summary table visualization")
    print("  - summary_table.txt: Text summary table")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze and compare VQ-VAE quantization methods'
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing method subdirectories (e.g., outputs/group_norm_v5)'
    )

    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=None,
        help='Specific methods to analyze (default: auto-detect all)'
    )

    args = parser.parse_args()
    main(args)
