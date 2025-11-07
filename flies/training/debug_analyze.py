"""
Quick debug script to see what's being loaded from your runs.
Run this first to diagnose the issue.
"""

import sys
import torch
from pathlib import Path

def debug_run(run_dir):
    """Debug what checkpoints are found."""
    run_dir = Path(run_dir)

    print(f"\n{'='*80}")
    print(f"Analyzing: {run_dir}")
    print(f"{'='*80}")

    # Check if directory exists
    if not run_dir.exists():
        print(f"❌ Directory does not exist!")
        return

    # Find checkpoint files
    checkpoints = sorted(run_dir.glob('checkpoint_epoch_*.pt'))
    print(f"\nFound {len(checkpoints)} checkpoint files:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")

    if not checkpoints:
        print("❌ No checkpoint files found!")
        print("\nLooking for any .pt files:")
        all_pts = list(run_dir.glob('*.pt'))
        for pt in all_pts:
            print(f"  - {pt.name}")
        return

    # Load first and last checkpoint to see structure
    print(f"\n--- First checkpoint: {checkpoints[0].name} ---")
    first = torch.load(checkpoints[0], map_location='cpu')
    print(f"Keys: {list(first.keys())}")
    print(f"Epoch: {first.get('epoch', 'N/A')}")
    if 'train_metrics' in first:
        print(f"Metrics: {first['train_metrics']}")

    print(f"\n--- Last checkpoint: {checkpoints[-1].name} ---")
    last = torch.load(checkpoints[-1], map_location='cpu')
    print(f"Epoch: {last.get('epoch', 'N/A')}")
    if 'train_metrics' in last:
        print(f"Metrics: {last['train_metrics']}")

    # Check for args
    if 'args' in last:
        args = last['args']
        print(f"\nModel config:")
        print(f"  num_embeddings: {args.get('num_embeddings', '?')}")
        print(f"  embedding_dim: {args.get('embedding_dim', '?')}")
        print(f"  commitment_cost: {args.get('commitment_cost', '?')}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_analyze.py <run_dir1> [run_dir2] ...")
        sys.exit(1)

    for run_dir in sys.argv[1:]:
        debug_run(run_dir)
