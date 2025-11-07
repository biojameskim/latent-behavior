"""
Find the best epoch across all your training runs.
This helps identify when your model was actually at its best.

Usage:
    python find_best_epoch.py outputs/stable_training
    python find_best_epoch.py outputs/group_norm_run outputs/stable_training
"""

import sys
import torch
from pathlib import Path

def find_best_epoch(run_dir):
    """Find the epoch with lowest total loss in a run."""
    run_dir = Path(run_dir)

    if not run_dir.exists():
        print(f"❌ Directory does not exist: {run_dir}")
        return None

    # Find all checkpoints and sort numerically
    checkpoints = list(run_dir.glob('checkpoint_epoch_*.pt'))

    if not checkpoints:
        print(f"❌ No checkpoints found in {run_dir}")
        return None

    def extract_epoch(path):
        return int(path.stem.split('_')[-1])

    checkpoints = sorted(checkpoints, key=extract_epoch)

    print(f"\n{'='*80}")
    print(f"Analyzing: {run_dir.name}")
    print(f"{'='*80}")

    # Load all epochs and track metrics
    results = []
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            epoch = ckpt['epoch']
            metrics = ckpt.get('train_metrics', {})

            if not metrics:
                continue

            results.append({
                'epoch': epoch,
                'loss': metrics.get('loss', float('inf')),
                'recon_loss': metrics.get('recon_loss', 0),
                'vq_loss': metrics.get('vq_loss', 0),
                'perplexity': metrics.get('perplexity', 0)
            })
        except Exception as e:
            print(f"⚠️  Failed to load {ckpt_path.name}: {e}")

    if not results:
        print("❌ No valid metrics found")
        return None

    # Find best by different criteria
    best_total_loss = min(results, key=lambda x: x['loss'])
    best_recon_loss = min(results, key=lambda x: x['recon_loss'])
    best_vq_loss = min(results, key=lambda x: x['vq_loss'])
    best_perplexity = max(results, key=lambda x: x['perplexity'])

    # Print all epochs
    print(f"\nAll epochs (found {len(results)} checkpoints):")
    print(f"{'Epoch':<8} {'Total Loss':<12} {'Recon Loss':<12} {'VQ Loss':<12} {'Perplexity':<12}")
    print("-" * 80)
    for r in results:
        num_codes = ckpt.get('args', {}).get('num_embeddings', 64)
        perp_pct = f"{r['perplexity']:.1f}/{num_codes} ({100*r['perplexity']/num_codes:.0f}%)"

        marker = ""
        if r == best_total_loss:
            marker = " ← BEST OVERALL"

        print(f"{r['epoch']:<8} {r['loss']:<12.2f} {r['recon_loss']:<12.2f} "
              f"{r['vq_loss']:<12.2f} {perp_pct:<12}{marker}")

    # Print recommendations
    print(f"\n{'='*80}")
    print("BEST EPOCHS BY DIFFERENT METRICS:")
    print(f"{'='*80}")

    print(f"\n✅ Best Total Loss: Epoch {best_total_loss['epoch']}")
    print(f"   Loss: {best_total_loss['loss']:.2f} "
          f"(Recon: {best_total_loss['recon_loss']:.2f}, VQ: {best_total_loss['vq_loss']:.2f})")

    print(f"\n✅ Best Reconstruction: Epoch {best_recon_loss['epoch']}")
    print(f"   Recon Loss: {best_recon_loss['recon_loss']:.2f}")

    print(f"\n✅ Best VQ Loss: Epoch {best_vq_loss['epoch']}")
    print(f"   VQ Loss: {best_vq_loss['vq_loss']:.2f}")

    print(f"\n✅ Best Codebook Utilization: Epoch {best_perplexity['epoch']}")
    print(f"   Perplexity: {best_perplexity['perplexity']:.2f}")

    # Check for divergence
    first_10_avg = sum(r['vq_loss'] for r in results[:min(10, len(results))]) / min(10, len(results))
    last_10_avg = sum(r['vq_loss'] for r in results[-min(10, len(results)):]) / min(10, len(results))

    print(f"\n{'='*80}")
    print("TRAINING STABILITY:")
    print(f"{'='*80}")
    print(f"Average VQ loss (first 10 epochs): {first_10_avg:.2f}")
    print(f"Average VQ loss (last 10 epochs):  {last_10_avg:.2f}")

    if last_10_avg > first_10_avg * 2:
        print(f"❌ DIVERGENCE DETECTED: VQ loss increased {last_10_avg/first_10_avg:.1f}x")
        print(f"   Training collapsed after early epochs!")
        print(f"   → Use checkpoint from epoch {best_total_loss['epoch']} (best model)")
    elif last_10_avg > first_10_avg * 1.2:
        print(f"⚠️  MILD INSTABILITY: VQ loss increased {last_10_avg/first_10_avg:.1f}x")
        print(f"   Consider stopping earlier or lowering learning rate")
    else:
        print(f"✅ STABLE: VQ loss remained consistent")
        if results[-1]['loss'] == best_total_loss['loss']:
            print(f"   Still improving - could train longer!")
        else:
            print(f"   Best model was at epoch {best_total_loss['epoch']}")

    return best_total_loss

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python find_best_epoch.py <run_dir1> [run_dir2] ...")
        print("\nExample:")
        print("  python find_best_epoch.py outputs/stable_training")
        print("  python find_best_epoch.py outputs/group_norm_run outputs/stable_training")
        sys.exit(1)

    best_models = []
    for run_dir in sys.argv[1:]:
        best = find_best_epoch(run_dir)
        if best:
            best_models.append((run_dir, best))

    # Compare across runs
    if len(best_models) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON ACROSS RUNS:")
        print(f"{'='*80}")

        for run_dir, best in best_models:
            print(f"\n{Path(run_dir).name}:")
            print(f"  Best epoch: {best['epoch']}")
            print(f"  Total loss: {best['loss']:.2f}")
            print(f"  Recon loss: {best['recon_loss']:.2f}")
            print(f"  VQ loss: {best['vq_loss']:.2f}")

        overall_best = min(best_models, key=lambda x: x[1]['loss'])
        print(f"\n🏆 OVERALL WINNER: {Path(overall_best[0]).name}")
        print(f"   Epoch {overall_best[1]['epoch']} with loss {overall_best[1]['loss']:.2f}")
