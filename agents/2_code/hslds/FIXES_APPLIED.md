# HSLDS Model Fixes - Addressing Codebook Collapse

## Problem Diagnosis

The original model suffered from severe **codebook collapse** and **degenerate segmentation**:
- Model converged to using only 1 state out of 12
- Temporal change rate dropped to 0.0 (no state transitions)
- Codebook loss reached maximum (2.48 ≈ log(12))
- While reconstruction MSE improved, the model learned a trivial solution

**Root Cause**: Graph encoder lacked temporal context and loss function forced uniform state usage, leading to collapse into single-state solution.

---

## Fixes Implemented

### Fix #1: Added Temporal Convolution to Encoder ✅

**File**: `model.py` (lines 325-326, 350-354)

**Changes**:
```python
# Added in __init__:
self.temporal_conv = nn.Conv1d(48, 48, kernel_size=5, padding=2, groups=48)

# Modified encode_to_latent():
x_positions_t = x_positions.permute(0, 2, 1)  # (B, 48, T)
x_temporal = self.temporal_conv(x_positions_t)  # (B, 48, T)
x_temporal = x_temporal.permute(0, 2, 1)  # (B, T, 48)
```

**Impact**:
- Each frame now has a receptive field of ±2 frames (167ms at 30Hz)
- Captures temporal context critical for distinguishing behaviors
- Enables encoder to differentiate "standing still" vs "grooming" with similar poses

---

### Fix #2: Updated Loss Weights ✅

**File**: `training.py` (lines 120-123)

**Changes**:
```python
# Old weights:
alpha=1.0, beta=0.25, gamma=0.1, delta=0.5

# New weights:
alpha=1.0  # Reconstruction (unchanged)
beta=0.5   # INCREASED from 0.25 - stronger commitment
gamma=0.0  # REMOVED from 0.1 - let model find natural # of states
delta=1.0  # INCREASED from 0.5 - penalize flickering more
```

**Impact**:
- Removing `gamma` allows model to find natural number of states (not forced uniform)
- Increasing `delta` penalizes rapid state changes more heavily
- Increasing `beta` encourages stable code assignments

---

### Fix #3: Added Gumbel Temperature Annealing ✅

**File**: `model.py` (lines 336, 412-413) and `training.py` (lines 85-86)

**Changes**:
```python
# In model __init__:
self.current_epoch = 0

# In forward():
epoch_fraction = min(self.current_epoch / 50, 1.0)
temperature = max(1.0 - 0.9 * epoch_fraction, 0.1)  # 1.0 → 0.1

# In training loop:
model.current_epoch = epoch
```

**Impact**:
- Anneals temperature from 1.0 (hot, exploratory) to 0.1 (cold, committed)
- Prevents premature collapse to single state
- Allows 50 epochs of exploration before forcing hard decisions

---

### Fix #4: Reduced Number of States ✅

**File**: `train_hslds.sh` (lines 82, 90)

**Changes**:
```bash
# Old:
--n_states 12

# New:
--n_states 6
```

**Impact**:
- Reduces overfitting with only 50 training sequences
- Matches closer to true behavior count (3 ground truth labels)
- Easier for model to learn with smaller discrete space

---

## Expected Improvements

### Training Dynamics

**Early Epochs (1-20):**
- Temporal flickering (healthy exploration)
- Codebook loss moderate (0.5-1.5)
- Temperature high (1.0 → 0.5)

**Mid Epochs (20-50):**
- Gradual stabilization
- Temperature annealing (0.5 → 0.1)
- States crystallizing

**Late Epochs (50-100):**
- Stable state assignments
- Temperature cold (0.1)
- Refinement of boundaries

### Metrics Targets

| Metric | Old (Failed) | Target (Fixed) |
|--------|--------------|----------------|
| Codebook Usage | <10% (1 state) | 50-100% (3-6 states) |
| Temporal Change Rate | 0.0000 | 0.1-0.3 |
| Mean Bout Length | ∞ (never switch) | 20-100 frames |
| Reconstruction MSE | 0.64 | 0.5-0.8 |
| Codebook Loss | 2.48 (collapsed) | 0.5-1.5 |
| Temporal Loss | 0.0 | 0.1-0.3 |

### Failure Modes

**Should NOT see:**
- CODEBOOK_COLLAPSE after epoch 30
- DEGENERATE_SEGMENTATION after epoch 40
- Temporal change rate < 0.05

**May see (acceptable):**
- TEMPORAL_FLICKERING in epochs 1-15 (exploring)
- POOR_RECONSTRUCTION (trading off MSE for structure)

---

## Model Architecture Changes

### Before (Problematic):
```
Raw Keypoints → Graph Encoder → Latent → Switching → States
                 (no temporal context)
```

### After (Fixed):
```
Raw Keypoints → Temporal Conv → Graph Encoder → Latent → Switching → States
                 (167ms context)    (spatial)             (annealed)
```

---

## Parameter Count

No change in total parameters (~115,000):
- Temporal conv adds: 48 * 5 = 240 parameters
- All other components unchanged

---

## Testing

To verify fixes work:

```bash
cd /share/j_sun/jjk297/repos/latent-behavior/agents/2_code/hslds
python test_installation.py
```

Expected output:
```
✓ ALL TESTS PASSED
✓ Model instantiated (115,249 parameters)  # Slight increase from 115,009
```

---

## Running Training

Submit to SLURM:
```bash
cd /share/j_sun/jjk297/repos/latent-behavior/slurm
sbatch train_hslds.sh
```

Expected runtime: ~30-45 minutes on RTX 2080 Ti

---

## Validation Criteria

Model is **scientifically useful** if:

1. **Structure Discovery**:
   - Uses 3-6 states out of 6 (50-100% usage)
   - Mean bout length: 20-100 frames (0.66-3.3 seconds)
   - Temporal change rate: 0.1-0.3

2. **Generative Quality**:
   - MMD score < 0.5 (distribution match)
   - ACF error < 0.1 (temporal dynamics match)
   - Generates complete sequences without errors

3. **Interpretability**:
   - Discovered states show clear temporal persistence
   - State transitions are sparse and meaningful
   - Can answer: "When does behavior X occur?"

---

## If Still Fails

If model still shows collapse after these fixes:

**Diagnostic**:
1. Check epoch where collapse starts
2. Review temporal loss curve
3. Inspect state usage entropy

**Additional fixes to try**:
1. Increase `delta` to 2.0 (even stronger temporal penalty)
2. Add sticky prior bias initialization to 5.0 (in SwitchingPolicy)
3. Reduce learning rate to 5e-4
4. Add state usage regularization (penalize unused states)

---

**Implementation Date**: 2025-11-29
**Status**: All fixes applied ✅
**Ready for Testing**: Yes
