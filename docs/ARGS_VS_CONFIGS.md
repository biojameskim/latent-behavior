# Command-Line Arguments vs Method-Specific Configs

## The Problem

When you run:
```bash
--embedding_dim 128 --num_embeddings 32
```

Which methods use these values?

## Current Behavior (After Fix)

### `--embedding_dim` (Encoder/Decoder Dimension)
✅ **Used by ALL methods** - This is the architecture dimension (encoder output, decoder input)

| Method | Uses `--embedding_dim`? | Notes |
|--------|------------------------|-------|
| VQ | ✅ Yes (128) | Encoder outputs 128-dim vectors |
| VQ_Improved | ✅ Yes (128) | Same encoder architecture |
| FSQ | ✅ Yes (128) | Same encoder architecture |
| RVQ | ✅ Yes (128) | Same encoder architecture |
| LFQ | ✅ Yes (128) | Projects from 128 → 16 → 128 |

**This is correct!** All methods share the same encoder/decoder architecture.

---

### `--num_embeddings` (Codebook Size)
⚠️ **Only used by methods without their own config override**

| Method | Uses `--num_embeddings`? | Actual Codebook Size | Where It's Set |
|--------|-------------------------|---------------------|----------------|
| **VQ** | ✅ Yes (32) | 32 codes | Command line |
| **VQ_Improved** | ✅ Yes (32) | 32 codes | Command line |
| **FSQ** | ❌ No | ~1000 codes | `levels=[8,5,5,5]` in config |
| **RVQ** | ✅ Yes (32) | 32 codes × 4 quantizers | Command line |
| **LFQ** | ❌ No | 64 codes | `codebook_size=64` in config |

---

## How to Customize Per-Method

### Option 1: Edit QUANTIZER_CONFIGS in train_comparison.py

Example - Give RVQ a larger codebook:

```python
'rvq': {
    'method': 'rvq',
    'kwargs': {
        'num_quantizers': 4,
        'kmeans_init': True,
        'threshold_ema_dead_code': 2,
        'shared_codebook': False
    },
    'codebook_size_override': 64  # ← Set this! RVQ will use 64 instead of --num_embeddings
},
```

### Option 2: Change FSQ Levels

```python
'fsq': {
    'method': 'fsq',
    'kwargs': {
        'levels': [7, 5, 5, 3]  # ← Change this! 7×5×5×3 = 525 codes
    },
    'codebook_size_override': None
},
```

### Option 3: Change LFQ Codebook Size

```python
'lfq': {
    'method': 'lfq',
    'kwargs': {
        'lfq_dim': 16,
        'codebook_size': 128,  # ← Change this! LFQ will use 128 codes (2^7)
        'entropy_loss_weight': 0.1,
        'diversity_gamma': 1.0
    },
    'codebook_size_override': None
},
```

---

## Your Current Setup

With your command:
```bash
--embedding_dim 128 --num_embeddings 32
```

You'll get:

| Method | Codebook Size | Encoder Dim | Comments |
|--------|--------------|-------------|----------|
| VQ | 32 | 128 | ✓ Uses your --num_embeddings |
| VQ_Improved | 32 | 128 | ✓ Uses your --num_embeddings |
| FSQ | ~1000 | 128 | ✓ Ignores --num_embeddings (uses levels) |
| RVQ | 32 per quantizer | 128 | ✓ Uses your --num_embeddings |
| LFQ | 64 | 128 | ✓ Ignores --num_embeddings (uses config) |

This is a **fair comparison** because:
- VQ, VQ_Improved, RVQ all use same codebook size (32)
- FSQ and LFQ use their recommended defaults
- All share the same architecture (embedding_dim=128)

---

## Recommendations

### For Your First Run
✅ **Keep current settings** - It's a fair comparison with appropriate defaults

### If VQ/VQ_Improved perform poorly
Try giving them more capacity:
```python
'vq_improved': {
    ...
    'codebook_size_override': 64  # Larger codebook
}
```

### If RVQ performs poorly
Try:
1. More quantizers: `'num_quantizers': 8`
2. Larger per-quantizer codebook: `'codebook_size_override': 64`

### If FSQ needs adjustment
Change levels for different codebook size:
- Smaller: `[7, 5, 5, 3]` = 525 codes
- Larger: `[8, 8, 8, 5]` = 2560 codes

---

## Summary

✅ **`--embedding_dim`** applies to ALL methods (encoder/decoder architecture)
⚠️ **`--num_embeddings`** applies to VQ, VQ_Improved, RVQ only
🎯 **FSQ and LFQ** use their own method-specific codebook configs

Your command is correct and will give you a fair comparison! 🎉
