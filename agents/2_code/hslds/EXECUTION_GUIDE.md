# HSLDS Execution Guide

Complete step-by-step instructions for running the behavior discovery pipeline.

## Step 1: Installation

### Environment Setup

```bash
# Navigate to the HSLDS directory
cd agents/2_code/hslds

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run test script
python test_installation.py
```

You should see:
```
✓ All imports successful
✓ Model instantiated (XXX,XXX parameters)
✓ Preprocessing works (output shape: torch.Size([2, 50, 121]))
✓ Forward pass works (recon: torch.Size([2, 50, 121]), codes: torch.Size([2, 50]))
✓ Generation works (synthetic shape: torch.Size([2, 50, 121]))
✓ Loss computation works (total: X.XXXX)
✓ ALL TESTS PASSED - Installation successful!
```

If you see errors, check [README.md](README.md) Troubleshooting section.

---

## Step 2: Prepare Data

Ensure you have the dataset file:
```
mabe22_subset_for_claude.npz
```

This file should contain:
- `trajectories`: (50, 300, 48) - Raw keypoint data
- `labels`: (50, 300) - Ground truth labels (not used until validation)
- `keypoint_vocabulary`: List of keypoint names
- `metadata`: Dataset information

---

## Step 3: Training

### Quick Start (Default Settings)

```bash
python main.py \
    --data_path /path/to/mabe22_subset_for_claude.npz \
    --epochs 50 \
    --batch_size 16 \
    --device cuda
```

**Expected runtime**: ~10-15 minutes on GPU, ~1-2 hours on CPU

### Recommended Settings for Best Results

```bash
python main.py \
    --data_path /path/to/mabe22_subset_for_claude.npz \
    --epochs 100 \
    --batch_size 32 \
    --n_states 12 \
    --latent_dim 32 \
    --lr 0.001 \
    --device cuda \
    --output_dir ./output/hslds_baseline \
    --seed 42
```

### Training Output

You'll see output like this:

```
============================================================
LOADING DATA
============================================================
Loaded dataset:
  Trajectories shape: (50, 300, 48)
  Labels shape: (50, 300)
  Keypoints: 24
  Number of unique behaviors: X

============================================================
INITIALIZING MODEL
============================================================
Architecture: Hierarchical Switching Linear Dynamical System
  Input dimension:  48
  Latent dimension: 32
  Number of states: 12
  Total parameters: XXX,XXX

============================================================
TRAINING
============================================================
Training on 50 sequences for 50 epochs
Device: cuda

Epoch 1/50: 100%|████████| 4/4 [00:XX<00:00, X.XXit/s, loss=X.XXXX, recon=X.XXXX]
Epoch 1/50 - Loss: X.XXXX, Recon: X.XXXX, Codebook: X.XXXX, Temporal: X.XXXX

[Potential warnings about failure modes appear here]
...
```

### Monitoring Training

Watch for these warnings:

- `[WARNING] Failure modes detected: ['CODEBOOK_COLLAPSE']`
  → Not all states are being used - see troubleshooting

- `[WARNING] Failure modes detected: ['TEMPORAL_FLICKERING']`
  → States changing too rapidly - increase temporal loss weight

- `[WARNING] Failure modes detected: ['POOR_RECONSTRUCTION']`
  → Model not learning features - check learning rate

---

## Step 4: Evaluation

After training completes, evaluation runs automatically:

```
============================================================
INTRINSIC EVALUATION (No Ground Truth)
============================================================
Running intrinsic evaluation...
  - Computing reconstruction MSE...
  - Analyzing code usage...
  - Computing temporal statistics...
  - Generating synthetic data...
  - Computing MMD score...
  - Computing ACF error...

============================================================
EXTRINSIC EVALUATION (With Ground Truth)
============================================================
Running extrinsic evaluation (with ground truth)...

============================================================
EVALUATION RESULTS
============================================================

--- INTRINSIC METRICS (No Ground Truth) ---
  Reconstruction MSE:      X.XXXXXX
  Codebook Usage:          XX%
  Mean Bout Length:        XX.XX frames
  MMD Score:               X.XXXXXX
  ACF Error:               X.XXXXXX
  Discovery Score:         XXX.XXXX

--- EXTRINSIC METRICS (With Ground Truth) ---
  Adjusted Rand Index:     X.XXXX
  Normalized Mutual Info:  X.XXXX
  Homogeneity:             X.XXXX
  Completeness:            X.XXXX
  V-Measure:               X.XXXX
```

---

## Step 5: Interpreting Results

### Intrinsic Metrics (Model Quality Without Labels)

**Reconstruction MSE**:
- **Good**: < 1.0
- **Acceptable**: 1.0 - 5.0
- **Poor**: > 5.0
- *Lower is better* - measures how well the model reconstructs input

**Codebook Usage**:
- **Good**: > 0.7 (70%+ of states used)
- **Acceptable**: 0.5 - 0.7
- **Poor**: < 0.5 (codebook collapse)
- *Higher is better* - ensures all behavioral states are utilized

**Mean Bout Length**:
- **Expected range**: 20-50 frames (0.66-1.66 seconds at 30 fps)
- Too low (< 10): Temporal flickering
- Too high (> 100): Under-segmentation

**MMD Score**:
- **Good**: < 0.5
- **Acceptable**: 0.5 - 2.0
- **Poor**: > 2.0
- *Lower is better* - synthetic data matches real data distribution

**ACF Error**:
- **Good**: < 0.1
- **Acceptable**: 0.1 - 0.3
- **Poor**: > 0.3
- *Lower is better* - synthetic data matches real temporal dynamics

**Discovery Score**:
- **Good**: > 100
- **Acceptable**: 50 - 100
- **Poor**: < 50
- *Higher is better* - combined quality metric

### Extrinsic Metrics (Comparison to Human Labels)

**Adjusted Rand Index (ARI)**:
- **Excellent**: > 0.6
- **Good**: 0.4 - 0.6
- **Moderate**: 0.2 - 0.4
- **Poor**: < 0.2
- *Higher is better* - agreement with human annotations

**Important**: Low ARI doesn't mean failure! The model may discover valid alternative segmentations. Check intrinsic metrics and visualizations.

**Normalized Mutual Information (NMI)**:
- Range: 0 to 1
- **Good**: > 0.5
- Measures information overlap between discovered codes and labels

**V-Measure**:
- Harmonic mean of homogeneity and completeness
- **Good**: > 0.5

---

## Step 6: Analyzing Outputs

### Generated Files

All outputs are saved to `--output_dir` (default: `./output`):

1. **training_history.png**
   - 4 panels showing loss curves
   - Check that losses are decreasing
   - Reconstruction should stabilize

2. **code_visualization.png**
   - Top row: Discovered codes for 5 sequences
   - Bottom row: Ground truth labels
   - Visual comparison of segmentation

3. **results.npy**
   - All metrics and training history
   - Load with: `results = np.load('results.npy', allow_pickle=True).item()`

4. **model.pth**
   - Trained model weights
   - Load with: `model.load_state_dict(torch.load('model.pth'))`

### Loading and Using Trained Model

```python
import torch
from model import DiscoveryPipeline

# Load model
model = DiscoveryPipeline(
    input_dim=48,
    latent_dim=32,
    n_states=12,
    output_dim=121
)
model.load_state_dict(torch.load('output/model.pth'))
model.eval()

# Encode new data
new_data = torch.randn(1, 300, 48)  # Your new trajectory
processed = model.preprocessor.preprocess(new_data)
codes = model.encode(processed)

print(f"Discovered behavioral sequence: {codes[0]}")

# Generate new behaviors
synthetic = model.generate(n_samples=10, length=300)
print(f"Generated {synthetic.shape[0]} synthetic trajectories")
```

---

## Step 7: Hyperparameter Tuning

If results are unsatisfactory, try these adjustments:

### For Low Codebook Usage (< 0.5)

```bash
# Option 1: Increase codebook utilization loss
# Edit training.py, line ~80: gamma=0.2 (instead of 0.1)

# Option 2: Reduce number of states
python main.py --n_states 8 --data_path ...

# Option 3: Increase latent dimension
python main.py --latent_dim 64 --data_path ...
```

### For High Temporal Flickering

```bash
# Option 1: Increase temporal coherence loss
# Edit training.py, line ~80: delta=1.0 (instead of 0.5)

# Option 2: Reduce learning rate
python main.py --lr 0.0005 --data_path ...
```

### For Poor Reconstruction (High MSE)

```bash
# Option 1: Train longer
python main.py --epochs 100 --data_path ...

# Option 2: Increase model capacity
python main.py --latent_dim 64 --data_path ...

# Option 3: Reduce commitment loss
# Edit training.py, line ~80: beta=0.1 (instead of 0.25)
```

### For Low ARI but Good Intrinsic Metrics

This is **not necessarily a problem**! The model may have discovered a valid alternative behavioral segmentation. Examine the visualizations:

```bash
# Increase number of states to match label count
python main.py --n_states 16 --data_path ...

# Or accept that the model found different (but valid) structure
```

---

## Step 8: Comparing to Alternative Architectures

To implement other architectures (VQ-VAE, Transformer) for comparison:

1. Create new directory: `agents/2_code/vqvae/`
2. Implement `DiscoveryPipeline` interface with:
   - `encode(x)` → codes
   - `decode(codes)` → reconstructed
   - `forward(x)` → (reconstructed, codes)
   - `generate(n_samples, length)` → synthetic

3. Use **same** `discovery_loss` function
4. Use **same** `IntrinsicEvaluator` and `ExtrinsicEvaluator`
5. Compare discovery scores

---

## Quick Reference: Common Commands

```bash
# Minimal run (CPU, fast)
python main.py --data_path data.npz --epochs 20 --device cpu

# Standard run (GPU, balanced)
python main.py --data_path data.npz --epochs 50 --device cuda

# High-quality run (GPU, slow but best results)
python main.py --data_path data.npz --epochs 100 --batch_size 32 \
    --latent_dim 64 --device cuda --output_dir ./output/best

# Hyperparameter search
for n_states in 8 12 16 20; do
    python main.py --data_path data.npz --n_states $n_states \
        --output_dir ./output/states_${n_states}
done

# Test installation
python test_installation.py

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Support & Troubleshooting

See [README.md](README.md) for detailed troubleshooting.

Common issues:
- **CUDA OOM**: Reduce batch size
- **Import errors**: Check torch-geometric installation
- **Low scores**: Try hyperparameter tuning above
- **Slow training**: Use GPU (`--device cuda`)

---

## Expected Timeline

- Installation: 5-10 minutes
- Testing: 1 minute
- Training (50 epochs, GPU): 10-15 minutes
- Training (50 epochs, CPU): 1-2 hours
- Evaluation: 2-5 minutes
- Total (GPU): ~20-30 minutes
- Total (CPU): ~1.5-2.5 hours
