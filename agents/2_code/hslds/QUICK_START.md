# HSLDS Quick Start Guide

## Three Commands to Success

### 1. Install (30 seconds)
```bash
cd agents/2_code/hslds
pip install -r requirements.txt
```

### 2. Verify (10 seconds)
```bash
python test_installation.py
```
Expected: `✓ ALL TESTS PASSED`

### 3. Train (10-15 minutes on GPU)
```bash
python main.py --data_path YOUR_DATA.npz --device cuda
```
Replace `YOUR_DATA.npz` with actual path to `mabe22_subset_for_claude.npz`

---

## What You Get

After running, check `./output/`:
- ✅ **training_history.png** - Did losses decrease?
- ✅ **code_visualization.png** - Do codes look reasonable?
- ✅ **Console metrics** - Is Discovery Score > 100?

---

## Good vs Bad Results

### ✅ Good Results
```
Reconstruction MSE:      0.45      ← Low
Codebook Usage:          83%       ← High
Mean Bout Length:        34 frames ← Reasonable
MMD Score:               0.32      ← Low
Discovery Score:         156       ← High
ARI:                     0.52      ← Good alignment
```

### ❌ Bad Results (with fixes)
```
Codebook Usage:          15%       ← TOO LOW
  Fix: python main.py --n_states 8 ...

Reconstruction MSE:      8.3       ← TOO HIGH
  Fix: python main.py --epochs 100 --latent_dim 64 ...

Mean Bout Length:        2 frames  ← Flickering
  Fix: Edit training.py, increase delta to 1.0
```

---

## Command Options

```bash
python main.py \
    --data_path PATH        # Required: your .npz file
    --epochs 50             # More = better (try 100)
    --batch_size 16         # Lower if CUDA OOM
    --n_states 12           # Fewer = coarser behaviors
    --device cuda           # or 'cpu' (slower)
    --output_dir ./output   # Where to save results
```

---

## Troubleshooting One-Liners

```bash
# CUDA out of memory?
python main.py --data_path ... --batch_size 8 --device cuda

# No GPU?
python main.py --data_path ... --device cpu

# Faster testing?
python main.py --data_path ... --epochs 20 --device cpu

# Better results?
python main.py --data_path ... --epochs 100 --batch_size 32 --latent_dim 64
```

---

## File Guide

**Must read**:
- [INSTRUCTIONS.md](INSTRUCTIONS.md) - Full quick start
- [hslds/README.md](hslds/README.md) - Architecture overview

**Reference**:
- [hslds/EXECUTION_GUIDE.md](hslds/EXECUTION_GUIDE.md) - Step-by-step walkthrough
- [FINAL_SUMMARY.md](FINAL_SUMMARY.md) - Complete overview

**Deep dive**:
- [hslds/ARCHITECTURE.md](hslds/ARCHITECTURE.md) - Mathematical details
- [hslds/IMPLEMENTATION_SUMMARY.md](hslds/IMPLEMENTATION_SUMMARY.md) - Technical breakdown

---

## Expected Timeline

| Task | GPU | CPU |
|------|-----|-----|
| Install | 30s | 30s |
| Test | 10s | 10s |
| Train (50 epochs) | 10-15 min | 1-2 hours |
| **Total** | **~15 min** | **~2 hours** |

---

## Support

1. Installation issues? → [hslds/README.md](hslds/README.md) Troubleshooting
2. Poor results? → [hslds/EXECUTION_GUIDE.md](hslds/EXECUTION_GUIDE.md) Hyperparameter Tuning
3. Understanding metrics? → [FINAL_SUMMARY.md](FINAL_SUMMARY.md) Expected Performance

---

**Status**: Ready to run ✅
