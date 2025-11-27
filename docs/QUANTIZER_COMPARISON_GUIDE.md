# VQ-VAE Quantizer Comparison Guide

This guide explains the four alternative quantization methods you can now try instead of standard VQ-VAE.

## Overview

Your current VQ-VAE uses standard vector quantization with a learned codebook. Based on your results (not great reconstruction, ~18% codebook utilization), we've implemented **four alternative quantization methods** that may work better for behavioral sequence modeling.

## Available Methods

### 1. **Standard VQ** (`vq`) - Current Baseline
**What it is:** Your current implementation with GroupNorm pre-quantizer normalization.

**Pros:**
- Well-established, proven approach
- Already working reasonably well

**Cons:**
- Low codebook utilization (~18%)
- Training degradation after epoch 10
- Requires careful tuning (commitment cost, learning rate)

**When to use:** As a baseline for comparison

---

### 2. **Improved VQ** (`vq_improved`) - ⭐ Quick Win
**What it is:** Enhanced VQ from the `vector-quantize-pytorch` library with:
- **Lower codebook dimension** (codebook stored in 32D, projected from 256D)
- **Cosine similarity** (direction matters more than magnitude)
- **Dead code expiry** (automatically replaces unused codes)
- **K-means initialization** (better starting point)

**Pros:**
- ✅ Easy drop-in replacement (minimal code changes)
- ✅ Proven to increase codebook usage (Improved VQGAN paper)
- ✅ Should fix your 18% utilization problem
- ✅ More stable training

**Cons:**
- Still uses a learned codebook (some overhead)

**When to use:** **Try this first!** Easiest improvement with high likelihood of success.

**Config:**
```python
method='vq_improved'
kwargs={
    'codebook_dim': 32,         # Codebook in lower dim
    'use_cosine_sim': True,     # Directional similarity
    'threshold_ema_dead_code': 2,  # Replace codes with <2 samples
    'kmeans_init': True,        # Smart initialization
}
```

---

### 3. **Finite Scalar Quantization (FSQ)** (`fsq`) - ⭐⭐ Most Promising
**What it is:** No learned codebook! Simply rounds each dimension to discrete levels.

**How it works:**
- Input: continuous vector `[2.3, -0.7, 1.1, -0.2]`
- FSQ rounds to levels: `[2, -1, 1, 0]` (if levels = [8, 5, 5, 3])
- Implicit codebook size = 8 × 5 × 5 × 3 = 600 codes

**Pros:**
- ✅ **No codebook collapse** - mathematically impossible!
- ✅ **No auxiliary losses** - no commitment loss, EMA updates, etc.
- ✅ **Simpler training** - fewer hyperparameters to tune
- ✅ **Used in video models** - proven for temporal sequences
- ✅ **Deterministic** - same input always gives same output

**Cons:**
- Different paradigm (no learned codes to inspect)
- Codebook size is constrained by level choices

**When to use:** **Strong recommendation!** Eliminates your codebook collapse issues entirely.

**Config:**
```python
method='fsq'
kwargs={
    'levels': [8, 5, 5, 5]  # 1000 implicit codes
    # Or try: [7, 5, 5, 3] = 525 codes (closer to your 32-64 range)
}
```

---

### 4. **Residual VQ (RVQ)** (`rvq`) - ⭐ Best for Hierarchical Behavior
**What it is:** Multiple quantizers in sequence, each refining the previous residual.

**How it works:**
```
Input → VQ1 → residual₁ → VQ2 → residual₂ → VQ3 → residual₃ → VQ4
Final output = VQ1 + VQ2 + VQ3 + VQ4
```

**Pros:**
- ✅ **Hierarchical compression** - natural for behavior (coarse movement → fine adjustments)
- ✅ **Better reconstruction** - captures fine-grained details
- ✅ **Flexible capacity** - add more quantizers if needed
- ✅ **Used in audio models** - Jukebox, SoundStream

**Cons:**
- More complex (4 codebooks instead of 1)
- Slightly longer training time

**When to use:** **Great for behavioral sequences!** Flies have hierarchical motion:
- VQ1: Gross body position
- VQ2: Leg movements
- VQ3: Wing adjustments
- VQ4: Fine details (antennae, etc.)

**Config:**
```python
method='rvq'
kwargs={
    'num_quantizers': 4,  # 4 hierarchical stages
    'kmeans_init': True,
    'threshold_ema_dead_code': 2,
}
# Total effective codebook: 32^4 = 1,048,576 possible combinations!
```

---

### 5. **Lookup Free Quantization (LFQ)** (`lfq`) - MAGVIT-2 SOTA
**What it is:** Binary latents (±1) instead of codebook lookup. Used in state-of-the-art video generation (MAGVIT-2).

**How it works:**
- Encoder outputs continuous vectors
- Project to small dimension (e.g., 16)
- Quantize each dimension to {-1, +1}
- Entropy regularization encourages code diversity

**Pros:**
- ✅ **SOTA for video** - MAGVIT-2 beat all previous methods
- ✅ **No lookup needed** - very fast inference
- ✅ **Good for sequences** - designed for spatiotemporal data

**Cons:**
- Requires smaller embedding dim (16-32 instead of 256)
- Need to tune entropy loss weight

**When to use:** If FSQ and RVQ don't work well. Best for video-like temporal sequences.

**Config:**
```python
method='lfq'
kwargs={
    'lfq_dim': 16,  # Binary latent dimension
    'codebook_size': 64,  # 2^6 codes
    'entropy_loss_weight': 0.1,
}
```

---

## Recommended Experiment Plan

### Quick Test (1-2 hours)
```bash
# Test FSQ and Improved VQ (most likely to help)
python train_comparison.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --methods vq vq_improved fsq \
    --epochs 20 \
    --output_dir outputs/quick_test
```

### Full Comparison (overnight)
```bash
# Test all methods
python train_comparison.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --methods vq vq_improved fsq rvq lfq \
    --epochs 30 \
    --lr_scheduler_tmax 20 \
    --output_dir outputs/full_comparison
```

### What to Look For

After training, compare:

1. **Reconstruction Loss** - Which method reconstructs best?
2. **Codebook Usage** (Perplexity) - Higher = better utilization
3. **Training Stability** - Does loss keep improving or degrade?
4. **VQ Loss** - Lower means encoder-quantizer alignment

Example results summary:
```
Method         Best Val Loss  Codebook Usage  Notes
─────────────────────────────────────────────────────
vq             81.86          18%            Baseline (degrades after epoch 10)
vq_improved    73.21          45%            Better utilization!
fsq            69.34          N/A            No collapse, stable training
rvq            67.89          82%            Best reconstruction
lfq            74.12          51%            Good, but needs tuning
```

---

## Windowing Strategy Recommendations 🎯

### Current Approach: Random Windows
You're currently using **random windows** - extracting 150-frame windows at arbitrary positions.

### Recommended: Event-Based Windows
**Your PI is absolutely right!** Event-based windowing (e.g., starting at walking onset) is much better:

**Why:**
- ✅ **Temporal alignment** - Same behaviors appear at same positions
- ✅ **Reduces variability** - VQ doesn't learn "walking at t=10" AND "walking at t=50"
- ✅ **Better discretization** - Codes become actual "syllables" (walk→turn→stop)
- ✅ **Easier forecasting** - Predictable sequences from aligned starts

**Example:**
```python
# Instead of random windows:
window_start = random.randint(0, sequence_length - window_size)

# Use event-based:
walking_starts = detect_walking_onset(trajectory)
window_start = random.choice(walking_starts)
```

**Events to consider for flies:**
- Walking onset
- Turning onset
- Stopping
- Wing grooming start
- Social interaction start (if multi-fly)

**Implementation tip:** Add event detection to your dataset class:
```python
class FlyBehaviorDataset:
    def __init__(self, ..., event_based=True, event_type='walking'):
        self.event_based = event_based
        self.event_type = event_type
        if event_based:
            self.event_times = self._detect_events()

    def _detect_events(self):
        # Detect walking: speed > threshold for N consecutive frames
        speed = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
        walking_onset = (speed > threshold) & (prev_speed < threshold)
        return np.where(walking_onset)[0]
```

---

## Configuration Files

### Example: Best Practices Config
Save as `configs/fsq_improved.json`:
```json
{
  "model": {
    "input_dim": 48,
    "hidden_dims": [64, 128, 256],
    "embedding_dim": 256,
    "num_embeddings": 32,
    "window_size": 150,
    "quantizer_method": "fsq",
    "quantizer_kwargs": {
      "levels": [7, 5, 5, 3]
    }
  },
  "training": {
    "batch_size": 128,
    "epochs": 30,
    "lr": 0.0001,
    "lr_scheduler_tmax": 20,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0
  },
  "data": {
    "window_size": 150,
    "stride": 150,
    "event_based": true,
    "event_type": "walking"
  }
}
```

---

## Troubleshooting

### FSQ gives weird reconstructions
- Try different level combinations: [8,8,8,5,5] for more capacity
- Ensure encoder outputs span the quantization range

### Improved VQ still has dead codes
- Increase `threshold_ema_dead_code` (try 5 or 10)
- Use larger `codebook_dim` (try 64 instead of 32)

### RVQ is slow
- Reduce `num_quantizers` (try 2 or 3 instead of 4)
- Use `shared_codebook=True` to share codebook across quantizers

### LFQ loss explodes
- Reduce `entropy_loss_weight` (try 0.01)
- Use gradient clipping: `--grad_clip_norm 1.0`

---

## Next Steps

1. **Run quick comparison** (FSQ vs Improved VQ vs baseline)
2. **Implement event-based windowing** for walking onset
3. **Retrain with best method** using optimal settings
4. **Analyze learned behaviors** - visualize what each code represents
5. **Use for forecasting/clustering** - now that you have good discrete codes!

Good luck! Based on similar work in behavioral modeling, **FSQ or Residual VQ** will likely give you the best results. 🚀
