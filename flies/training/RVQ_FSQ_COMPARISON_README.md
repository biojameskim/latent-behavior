# RVQ and FSQ Comparison Guide

## Quick Start

Run all variations (20 epochs each for quick screening):

```bash
python flies/training/train_rvq_fsq_comparison.py \
    --train_data /home/user/latent-behavior/data/main/train_data.npy \
    --fly_split_file /home/user/latent-behavior/data/main/fly_train_val_split.pkl \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-3 \
    --embedding_dim 128 \
    --num_embeddings 32 \
    --commitment_cost 0.25 \
    --output_dir outputs/rvq_fsq_comparison
```

Run specific methods only:

```bash
python flies/training/train_rvq_fsq_comparison.py \
    --train_data /path/to/data.npy \
    --fly_split_file /path/to/split.pkl \
    --methods rvq_base rvq_deep fsq_base fsq_small \
    --epochs 20
```

## Available Configurations

### RVQ Variations

| Config | # Quantizers | Codebook Size | Shared? | Description |
|--------|-------------|---------------|---------|-------------|
| `rvq_base` | 4 | 32 | No | **Baseline** (original best performer, val loss: 4.74) |
| `rvq_deep` | 8 | 32 | No | More hierarchy for finer details |
| `rvq_large` | 4 | 64 | No | Larger codebooks per stage |
| `rvq_xlarge` | 6 | 64 | No | Maximum capacity (most parameters) |
| `rvq_shared` | 4 | 64 | Yes | Parameter efficient, shared codebook |
| `rvq_shallow_large` | 2 | 128 | No | Fast alternative, fewer stages |

**Recommendation**: Start with `rvq_base`, `rvq_deep`, and `rvq_large` for 20 epochs.

### FSQ Variations

| Config | Levels | Codebook Size | Description |
|--------|--------|---------------|-------------|
| `fsq_base` | [8,5,5,5] | 1000 | **Baseline** (original best FSQ, val loss: 14.17) |
| `fsq_small` | [7,5,5,3] | 525 | More efficient, fewer codes |
| `fsq_minimal` | [6,5,5,3] | 450 | Most efficient |
| `fsq_balanced` | [8,8,8,5] | 2560 | Balanced dimensions |
| `fsq_large` | [8,6,5,5,5] | 6000 | High capacity |
| `fsq_xlarge` | [9,8,7,6,5] | 15120 | Maximum capacity |

**Recommendation**: Start with `fsq_base`, `fsq_small`, and `fsq_large` for 20 epochs.

## Understanding the Results

### Key Metrics

- **Val Loss**: Lower is better (target: < 5.0 for RVQ, < 15.0 for FSQ)
- **Perplexity**: Higher is better (measures codebook utilization)
  - RVQ: Target > 15 (>45% utilization per quantizer)
  - FSQ: Target > 25 (>50% of codes used)
- **Recon Loss**: Main component, measures reconstruction quality
- **VQ Loss**: Codebook/commitment loss (0 for FSQ)

### What to Look For

**RVQ:**
- **Deep vs Shallow**: Does 8 quantizers improve over 4? Or is it just slower?
- **Codebook Size**: Does 64 codes/quantizer help, or is 32 enough?
- **Shared Codebooks**: Can we get similar performance with fewer parameters?

**FSQ:**
- **Capacity**: Is 1000 codes enough? Or do we need 2500+?
- **Efficiency**: Can we get similar loss with fewer codes (e.g., 525)?
- **Dimensions**: Do more balanced dimensions ([8,8,8,5]) help?

## Workflow

### Stage 1: Quick Screening (20 epochs)

Run all configurations for 20 epochs:

```bash
# Run all RVQ variations
python flies/training/train_rvq_fsq_comparison.py \
    --methods rvq_base rvq_deep rvq_large rvq_xlarge rvq_shared rvq_shallow_large \
    --epochs 20 \
    [other args...]

# Run all FSQ variations
python flies/training/train_rvq_fsq_comparison.py \
    --methods fsq_base fsq_small fsq_minimal fsq_balanced fsq_large fsq_xlarge \
    --epochs 20 \
    [other args...]
```

**Time estimate**: ~30-60 minutes per method (depends on dataset size)

### Stage 2: Deep Training (50-100 epochs)

Pick top 2-3 performers and train longer:

```bash
python flies/training/train_rvq_fsq_comparison.py \
    --methods rvq_deep rvq_xlarge fsq_large \
    --epochs 100 \
    --output_dir outputs/rvq_fsq_final \
    [other args...]
```

### Stage 3: Analysis

```bash
python flies/analysis/analyze_comparison.py \
    --results_dir outputs/rvq_fsq_comparison
```

## Expected Performance

Based on initial results:

**RVQ Family** (expected val loss range):
- `rvq_base`: 4.5 - 5.0 ✓ (confirmed)
- `rvq_deep`: 3.5 - 4.5 (hypothesis: better with more stages)
- `rvq_large`: 4.0 - 5.0 (hypothesis: similar or slightly better)
- `rvq_xlarge`: 3.0 - 4.0 (hypothesis: best but slowest)

**FSQ Family** (expected val loss range):
- `fsq_base`: 14.0 - 15.0 ✓ (confirmed)
- `fsq_small`: 15.0 - 18.0 (hypothesis: slightly worse but faster)
- `fsq_large`: 12.0 - 14.0 (hypothesis: best FSQ variant)
- `fsq_xlarge`: 10.0 - 13.0 (hypothesis: best overall FSQ)

## Tips

1. **Monitor perplexity**: If perplexity is low (<10), try larger codebooks
2. **Check training curves**: If loss plateaus early, try longer training or higher LR
3. **Watch for overfitting**: If val loss increases while train decreases, add regularization
4. **Resource management**: RVQ with 8 quantizers × 64 codes is memory intensive
5. **FSQ is stable**: FSQ should converge faster than RVQ (no codebook learning)

## Interpreting Results

### When RVQ Wins
- Hierarchical structure captures behavioral complexity well
- Multi-scale patterns in the data
- Willing to accept longer training time

### When FSQ Wins
- Data has natural "grid-like" structure
- Want stable, reproducible training
- Need fast inference (no codebook lookup overhead)
- Parameter budget is limited

### Recommended Next Steps

After 20-epoch screening:
1. **Pick top performer per family** (best RVQ + best FSQ)
2. **Train 100 epochs** with best configs
3. **Analyze learned representations** (t-SNE of codes, codebook usage)
4. **Test on downstream tasks** (classification, generation, etc.)
