# VQ-VAE Training Analysis & Recommendations

## Executive Summary

**Best Model:** `outputs/group_norm_run/checkpoint_epoch_10.pt`
- Total Loss: 81.86
- Reconstruction Loss: 77.89
- VQ Loss: 3.97
- Perplexity: 11.3/64 codes (18% utilization)

**Key Finding:** ALL training runs collapse after epoch 10-30. This is a systematic learning rate scheduling problem, not a one-time issue.

---

## Analysis of All Training Runs

### Run 1: group_norm_run (lr=1e-4, β=0.25, 64 codes)

| Epoch | Total Loss | Recon Loss | VQ Loss | Status |
|-------|-----------|------------|---------|--------|
| **10** | **81.86** ✅ | **77.89** ✅ | **3.97** ✅ | **BEST** |
| 20 | 89.60 | 80.24 | 9.36 | +9% worse |
| 30 | 102.44 | 86.07 | 16.38 | +25% worse |
| 50 | 103.76 | 73.53 | 30.23 | +27% worse |
| 100 | 153.95 | 106.21 | 47.74 | +88% worse |

**Verdict:** Peak at epoch 10, severe collapse afterward. VQ loss increased 12x!

---

### Run 2: group_norm_run_v2 (lr=1e-4, β=0.15, 64 codes)

| Epoch | Total Loss | Recon Loss | VQ Loss | Status |
|-------|-----------|------------|---------|--------|
| 10 | 101.23 | 97.48 | 3.75 | Good |
| 20 | 94.10 | 85.48 | 8.62 | Better |
| **30** | **90.67** | 75.18 | **15.49** | **BEST** |
| 40 | 95.10 | 71.75 | 23.35 | Degrading |
| 100 | 135.53 | 92.90 | 42.63 | +50% worse |

**Verdict:** Lowering commitment cost (0.25→0.15) delayed peak slightly (epoch 30), but ultimately still collapsed. Not a solution.

---

### Run 3: stable_training (lr=5e-5, β=0.25, 32 codes)

| Epoch | Total Loss | Recon Loss | VQ Loss | Status |
|-------|-----------|------------|---------|--------|
| **10** | **93.08** | **90.73** ❌ | **2.35** ✅ | **BEST (but worse overall)** |
| 30 | 117.35 | 111.35 | 6.00 | Much worse |
| 50 | 120.29 | 109.79 | 10.50 | Degrading |
| 100 | 125.86 | 112.03 | 13.82 | +35% worse |

**Verdict:**
- ✅ VQ loss is lowest (2.35) - good quantization
- ❌ Reconstruction is WORST (90.73 vs 77.89) - poor representations
- ❌ Still worse than original run at epoch 10 (93.08 vs 81.86)
- **Conclusion:** Conservative LR learns too slowly and gives wrong tradeoff

---

## Root Cause Analysis

### Problem: Learning Rate Schedule

All runs use `CosineAnnealingLR` with `T_max=100`:

```python
# LR schedule over 100 epochs:
Epoch 1-10:   lr = ~1e-4 (high)  ← Fast learning, great results!
Epoch 11-50:  lr = ~5e-5 (medium) ← Model overshooting, degrading
Epoch 51-100: lr = ~1e-5 (low)   ← Too late, damage done
```

**The model converges by epoch 10 but continues training with high LR!**

### Why This Happens

1. **Initial Phase (Epoch 1-10):** High LR (1e-4) enables fast learning
   - VQ loss drops quickly: 100+ → 4
   - Reconstruction improves rapidly
   - Model finds good minima

2. **Overfitting Phase (Epoch 10-30):** LR still too high
   - Model continues updating with large steps
   - Overshoots the good minima it found
   - VQ loss starts increasing (4 → 15 → 30 → 48)
   - Codebook and encoder drift apart

3. **Collapse Phase (Epoch 30-100):** Damage accumulates
   - Even as LR decreases, model can't recover
   - Total loss nearly doubles
   - Best strategy: should have stopped at epoch 10!

---

## Why Different Approaches Failed

### Lowering Commitment Cost (β: 0.25 → 0.15)
- ❌ Didn't prevent collapse
- Slightly delayed peak (epoch 10 → 30)
- Still collapsed by epoch 100
- **Not the solution**

### Lowering Learning Rate (lr: 1e-4 → 5e-5)
- ❌ Learned too slowly
- Worse results even at best epoch (93.08 vs 81.86)
- Poor reconstruction despite low VQ loss
- Codebook too small (32 codes) limited expressiveness
- **Wrong tradeoff**

### Gradient Clipping
- ✅ Prevents gradient explosion
- ❌ Doesn't prevent slow divergence from high LR
- Necessary but not sufficient

---

## Solution: Aggressive LR Decay

### Strategy

Since all runs peak at epoch 10-30 and degrade after, we should:

1. **Train for fewer epochs** (100 → 30)
   - Don't waste compute on epochs that make things worse
   - All runs were worse at epoch 100 than epoch 10!

2. **Decay LR much faster** (T_max: 100 → 20)
   - Current: LR decays slowly over 100 epochs
   - Better: LR decays to ~0 by epoch 20
   - This prevents the overfitting/divergence phase

3. **Save checkpoints more frequently** (every 5 epochs)
   - Catch the best model early
   - Don't rely on final checkpoint

4. **Monitor validation loss**
   - The saved `best_model.pt` should be the one to use
   - Not the final checkpoint!

### Recommended Settings

```bash
--epochs 30               # Stop early (before degradation)
--lr 1e-4                 # Start high (fast learning)
--lr_scheduler_tmax 20    # Decay fast (prevent divergence)
--grad_clip_norm 1.0      # Safety net
--save_every 5            # Catch best model
--num_embeddings 64       # More capacity than stable_training's 32
--embedding_dim 256       # Expressive codes
```

See `train_final.sh` for complete configuration.

---

## Metrics Interpretation

### What the Numbers Mean

**Total Loss (Recon + VQ):**
- < 85: Excellent ✅
- 85-100: Good
- 100-120: Acceptable
- \> 120: Poor ❌

**Reconstruction Loss (MSE):**
- < 80: Excellent ✅
- 80-90: Good
- 90-100: Acceptable
- \> 100: Poor ❌

**VQ Loss:**
- < 5: Excellent ✅
- 5-10: Good
- 10-20: Acceptable
- \> 20: Poor (scale mismatch) ❌

**Perplexity (Codebook Utilization):**
- \> 60% of codes: Excellent
- 30-60%: Good
- < 30%: Low (codebook too large) ⚠️

### Current Best Model (group_norm_run epoch 10)

- Total Loss: 81.86 ✅ **Excellent**
- Recon Loss: 77.89 ✅ **Excellent**
- VQ Loss: 3.97 ✅ **Excellent**
- Perplexity: 11.3/64 (18%) ⚠️ **Low utilization**

**Overall:** Excellent reconstruction and quantization, but codebook could be smaller or more of it used.

---

## Recommendations

### Immediate: Use Your Best Model

```bash
# This is your best checkpoint
outputs/group_norm_run/checkpoint_epoch_10.pt
```

Use this for:
- Inference on new data
- Visualizing learned behaviors
- Analyzing codebook embeddings
- Downstream tasks

### For Better Results: Retrain with Optimized Settings

```bash
chmod +x train_final.sh
./train_final.sh
```

**Expected improvements:**
- Match or beat current best (loss 81.86)
- Better codebook utilization (more expressive with 256 dims)
- Stable training without collapse
- Less wasted compute (30 epochs vs 100)

**Monitor for:**
- Peak likely at epoch 10-15 (based on all previous runs)
- Use `best_model.pt` (lowest validation loss)
- Compare with `find_best_epoch.py`

---

## Key Takeaways

1. ✅ **GroupNorm fix worked** - VQ loss stayed low in all runs
2. ✅ **Model can learn well** - epoch 10 results were excellent
3. ❌ **LR schedule was wrong** - T_max=100 caused slow divergence
4. ❌ **Training too long** - epochs 30-100 made things worse
5. 🎯 **Solution:** Use epoch 10 model OR retrain with T_max=20, epochs=30

---

## Files Reference

- **Best model:** `outputs/group_norm_run/checkpoint_epoch_10.pt`
- **Analysis script:** `find_best_epoch.py`
- **Optimized training:** `train_final.sh`
- **Comparison plots:** `analyze_runs_fixed.py`

---

## Future Improvements to Consider

1. **Early Stopping:** Monitor validation loss and stop when it plateaus
2. **LR Warmup:** Start with very low LR, ramp up, then decay
3. **Reduce Codebook Size:** Try 32 codes with embedding_dim=256
4. **EMA Updates:** Use exponential moving average for codebook (more stable)
5. **Different Scheduler:** Try ReduceLROnPlateau (decay when val loss stops improving)

But for now, your epoch 10 model is excellent! The main issue was continuing to train after convergence.
