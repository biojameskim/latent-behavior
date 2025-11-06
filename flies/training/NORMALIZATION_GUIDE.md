# VQ-VAE Normalization Guide

## The Problem We're Solving

Your VQ-VAE had **codebook collapse** with these symptoms:
- VQ loss: 640 (should be < 10)
- Poor reconstruction quality
- Low codebook utilization (12/16 codes used)

**Root cause:** Scale mismatch between encoder outputs and codebook embeddings.

---

## Two Solutions: Simple vs Robust

### Solution 1: Minimal Fix (Initialization Only) ⭐ **Start Here**

**What it does:**
- Fixes codebook initialization from `uniform(-1/K, 1/K)` to `normal(0, 1)`
- No architectural changes
- Closest to original VQ-VAE paper

**When to use:**
- ✅ You want the simplest solution
- ✅ You want to stay close to the original paper
- ✅ You're willing to tune hyperparameters if needed

**Files:**
- `vq_vae/quantizer_simple.py` - Just fixed initialization
- `vq_vae/vqvae_simple.py` - No pre-quantizer normalization
- `training/train_simple.py` - Training script
- `training/train_simple.sh` - Recommended hyperparameters

**Usage:**
```bash
cd /home/user/latent-behavior/flies/training
chmod +x train_simple.sh
./train_simple.sh
```

**Expected results:**
- VQ loss: 20-50 (much better than 640!)
- If still unstable, try Solution 2

---

### Solution 2: Robust Fix (Pre-Quantizer Normalization)

**What it does:**
- Adds GroupNorm layer before quantizer
- Normalizes encoder outputs to consistent scale
- Fixes initialization too

**When to use:**
- ✅ Solution 1 didn't work well (VQ loss still > 50)
- ✅ You want maximum robustness
- ✅ You don't mind a small architectural change

**Files:**
- `vq_vae/quantizer.py` - Fixed initialization
- `vq_vae/vqvae.py` - Includes GroupNorm layer
- `training/train.py` - Training script
- `training/train_fixed.sh` - Recommended hyperparameters

**Usage:**
```bash
cd /home/user/latent-behavior/flies/training
chmod +x train_fixed.sh
./train_fixed.sh
```

**Expected results:**
- VQ loss: < 10 (very robust!)
- More stable training
- Less sensitive to hyperparameters

---

## What is GroupNorm?

GroupNorm is a normalization technique:

```python
# For input shape (Batch, Channels, Time)
GroupNorm(num_groups=1, num_channels=embedding_dim)

# Normalizes: x_norm = (x - mean) / sqrt(var + eps)
# Where mean/var computed over (channels, time) per batch item
```

**Why GroupNorm instead of BatchNorm/LayerNorm?**

| Normalization | Normalizes Over | Batch-Size Dependent? | Good For |
|---------------|----------------|----------------------|----------|
| **BatchNorm** | Batch dimension | ✅ Yes (breaks with small batches) | Image classification |
| **LayerNorm** | Channels | ❌ No | Transformers, NLP |
| **InstanceNorm** | Spatial dimensions | ❌ No | Style transfer |
| **GroupNorm** | Channel groups | ❌ No | **VQ-VAE, any batch size** |

GroupNorm works well for temporal sequence models like yours.

---

## Is This Best Practice?

**Short answer:** It's **one good approach**, not universally required.

### What the Literature Says:

**Original VQ-VAE paper (van den Oord et al., 2017):**
- ❌ No pre-quantizer normalization
- ✅ Careful initialization + EMA updates

**VQ-VAE-2 (Razavi et al., 2019):**
- ❌ No pre-quantizer normalization

**Modern implementations:**
- Some add normalization for robustness
- Many don't (stay closer to original)
- Both can work if initialization is correct!

**My recommendation:** Try Solution 1 first (simpler), use Solution 2 if needed (more robust).

---

## Comparison Table

| Aspect | Solution 1 (Simple) | Solution 2 (GroupNorm) |
|--------|---------------------|------------------------|
| **Complexity** | ✅ Minimal change | ⚠️ Adds layer |
| **Closest to paper** | ✅ Yes | ⚠️ No |
| **Robustness** | ⚠️ Good if tuned | ✅ Very robust |
| **VQ loss target** | < 50 | < 10 |
| **Recommended for beginners** | ✅ Yes (try first!) | Use if Solution 1 fails |

---

## Debugging Tips

After training for 10 epochs, check:

```python
# Solution 1 (Simple):
# VQ loss should be 20-50 (not 600!)
# If > 50, switch to Solution 2

# Solution 2 (GroupNorm):
# VQ loss should be < 10
# Encoder outputs should have std ~ 1 after GroupNorm
```

Use `debug_new_model.py` to verify:
```bash
python debug_new_model.py
```

---

## Why Did the Original Code Fail?

```python
# OLD (BROKEN):
self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
# With num_embeddings=16: range = [-0.0625, 0.0625]  ← TOO SMALL!

# Encoder outputs: range ~ [-10, 10] or larger
# Distance between encoder and codebook: HUGE!
# VQ loss: 640+ (trying to pull them together)
```

This was a severe bug - the initialization scaled **inversely** with codebook size, making the problem worse with smaller codebooks!

---

## Summary

1. **Try Solution 1 first** (`train_simple.sh`)
   - Simplest fix
   - Closest to original paper
   - Might be all you need!

2. **If VQ loss still high (>50), use Solution 2** (`train_fixed.sh`)
   - More robust
   - Guaranteed to work
   - Small architectural change

3. **Both fix the root cause** (bad initialization)
   - Solution 1: Minimal approach
   - Solution 2: Defensive approach

4. **Cannot use old checkpoints**
   - Must train from scratch
   - Old codebook is stuck in bad state

---

## Further Reading

- Original VQ-VAE paper: https://arxiv.org/abs/1711.00937
- GroupNorm paper: https://arxiv.org/abs/1803.08494
- VQ-VAE-2: https://arxiv.org/abs/1906.00446
