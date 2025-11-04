# VQ-VAE Training Guide

## Overview

This directory contains the training script for the VQ-VAE model on fly behavior sequences. The training process learns a discrete codebook of behavior "syllables" by reconstructing temporal sequences of fly keypoint trajectories.

## Training Script: `train.py`

### Quick Start

```bash
# Navigate to training directory
cd /share/j_sun/jjk297/repos/latent-behavior/flies/training

# Basic training with default parameters
python train.py \
    --train_data ../../../data/fly_data/fly_group_train.npy \
    --val_data ../../../data/fly_data/fly_group_val.npy \
    --augment_rotation

# Training with custom hyperparameters
python train.py \
    --train_data ../../../../data/fly_data/fly_group_train.npy \
    --window_size 150 \
    --stride 75 \
    --hidden_dims 64 128 256 \
    --num_embeddings 128 \
    --embedding_dim 256 \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.0 \
    --beta1 0.9 \
    --beta2 0.99 \
    --output_dir ./outputs \
    --augment_rotation
```

## Command-Line Arguments

### Data Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_data` | str | **required** | Path to training .npy file |
| `--val_data` | str | `None` | Path to validation .npy file (optional). If omitted you can still create a validation loader by providing fly-level filters. |
| `--window_size` | int | `150` | Number of frames per window |
| `--stride` | int | `150` | Stride for sliding windows (use `window_size` for non-overlapping) |
| `--fly_split_file` | str | `None` | JSON file containing fly-level splits (e.g., output of `generate_fly_splits`). |
| `--train_split_name` | str | `train` | Key inside the split JSON for the training set. |
| `--val_split_name` | str | `val` | Key inside the split JSON for the validation set. |
| `--train_fly_filter` | str | `None` | Path to JSON list of fly records overriding the training split. |
| `--val_fly_filter` | str | `None` | Path to JSON list overriding the validation split. |

**Note on stride**:
- `stride = window_size` → non-overlapping windows (recommended for clean train/val split)
- `stride < window_size` → overlapping windows (more training data but potential leakage)

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_dim` | int | `48` | Input dimension (24 keypoints × 2 coordinates) |
| `--hidden_dims` | int list | `[64, 128, 256]` | Hidden dimensions for encoder/decoder layers |
| `--embedding_dim` | int | `512` | Latent embedding dimension |
| `--num_embeddings` | int | `512` | Codebook size (number of behavior syllables) |
| `--num_residual_blocks` | int | `2` | Number of residual blocks per layer |
| `--commitment_cost` | float | `0.25` | Beta coefficient for commitment loss |

**Model architecture notes**:
- `hidden_dims=[64, 128, 256]` → 3 downsampling layers with auto-computed strides
  - For window_size=150: strides=[5,3,2] → 150→30→10→5 (exact reconstruction)
  - Strides automatically factorize sequence length for any window size
- `num_embeddings=512` → learns 512 discrete behavior patterns
- `embedding_dim=512` → richness of each codebook entry representation

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | `32` | Training batch size |
| `--epochs` | int | `100` | Number of training epochs |
| `--lr` | float | `1e-3` | Learning rate for Adam optimizer |
| `--num_workers` | int | `4` | Number of DataLoader workers |
| `--augment_rotation` | flag | `False` | Apply random rotation augmentation for rotation-invariant behavior codes |

### Output Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | `./outputs` | Directory for saving checkpoints |
| `--save_every` | int | `10` | Save checkpoint every N epochs |

## Training Process

### What Happens During Training

1. **Data Loading**:
   - Loads fly trajectories from .npy files
   - Splits into individual fly sequences (removes multi-fly context)
   - Filters out flies with NaN tracking values
   - Creates sliding windows of size `window_size`
   - Transposes to `(batch, features, time)` format for Conv1d

2. **Model Forward Pass**:
   ```
   Input (B, 48, 150)
      ↓ Encoder (Conv1d with strides=[5,3,2] + residual blocks)
   Latent (B, 512, 5)
      ↓ Vector Quantizer (nearest neighbor lookup)
   Quantized (B, 512, 5)
      ↓ Decoder (Upsample+Conv with strides=[2,3,5] + residual blocks)
   Reconstruction (B, 48, 150)

   *If `--augment_rotation` is enabled, each training batch is randomly rotated around the origin before encoding so the learned codes become rotation-invariant.*
   ```

3. **Loss Computation**:
   - **Reconstruction Loss**: MSE between input and reconstruction
   - **VQ Loss**: Codebook loss + commitment loss
   - **Total Loss**: `recon_loss + vq_loss`

4. **Metrics Tracked**:
   - **Loss**: Total training/validation loss averaged over every batch in the epoch.
   - **Recon Loss**: Mean-squared error between the input window and its reconstruction.
   - **VQ Loss**: Sum of codebook and commitment losses returned by the vector quantizer.
   - **Perplexity**: Codebook usage computed from the empirical distribution of chosen code indices (higher = more codes actively used).

### Model Checkpoints

The training script saves three types of checkpoints:

1. **Best Model** (`best_model.pt`):
   - Saved when validation loss improves
   - Contains: model weights, optimizer state, best val loss, hyperparameters

2. **Periodic Checkpoints** (`checkpoint_epoch_N.pt`):
   - Saved every `--save_every` epochs
   - Contains: model weights, optimizer state, training metrics, hyperparameters

3. **Final Model** (`final_model.pt`):
   - Saved at end of training
   - Contains: model weights, optimizer state, hyperparameters

### Loading a Checkpoint

```python
import torch
from vq_vae.vqvae import VQVAE

# Load checkpoint
checkpoint = torch.load('outputs/best_model.pt')

# Recreate model with saved hyperparameters
args = checkpoint['args']
model = VQVAE(
    input_dim=args['input_dim'],
    hidden_dims=args['hidden_dims'],
    embedding_dim=args['embedding_dim'],
    num_embeddings=args['num_embeddings'],
    sequence_length=args['window_size'],  # Use window_size from training
    num_residual_blocks=args['num_residual_blocks'],
    commitment_cost=args['commitment_cost']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract behavior codes
codes = model.encode(x)  # (batch, reduced_time_steps)
```

## Hyperparameter Tuning Guide

### Starting Point (Recommended Defaults)

```bash
python train.py \
    --train_data path/to/train.npy \
    --val_data path/to/val.npy \
    --window_size 150 \
    --stride 150 \
    --hidden_dims 64 128 256 \
    --num_embeddings 512 \
    --embedding_dim 512 \
    --num_residual_blocks 2 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-3
```

### Tuning Strategies

**If reconstruction quality is poor**:
- Increase model capacity: `--hidden_dims 128 256 512`
- Add more residual blocks: `--num_residual_blocks 3`
- Increase embedding dim: `--embedding_dim 1024`
- Train longer: `--epochs 200`

**If codebook usage is low** (perplexity < 100):
- Increase codebook size: `--num_embeddings 1024`
- Decrease commitment cost: `--commitment_cost 0.1`
- Increase embedding dim: `--embedding_dim 1024`

**If training is slow**:
- Decrease batch size: `--batch_size 16`
- Use fewer workers: `--num_workers 2`
- Reduce model size: `--hidden_dims 32 64 128`

**If overfitting** (train loss << val loss):
- Use non-overlapping windows: `--stride 150`
- Add more data or augmentation
- Reduce model capacity: `--hidden_dims 32 64 128`

## Expected Training Time

Approximate training times on different hardware:

- **CPU only**: ~10-20 hours for 100 epochs (not recommended)
- **Single GPU (RTX 3090)**: ~2-4 hours for 100 epochs
- **Single GPU (A100)**: ~1-2 hours for 100 epochs

## Monitoring Training

### Log Output

The training script logs progress every 100 batches:

```
2025-01-15 10:30:45 - __main__ - INFO - Epoch 1 [0/125] | Loss: 0.8234 | Recon: 0.7891 | VQ: 0.0343 | Perplexity: 64.32
2025-01-15 10:31:12 - __main__ - INFO - Epoch 1 [100/125] | Loss: 0.5672 | Recon: 0.5401 | VQ: 0.0271 | Perplexity: 128.45
...
2025-01-15 10:32:00 - __main__ - INFO - Train | Loss: 0.5234 | Recon: 0.4982 | VQ: 0.0252 | Perplexity: 156.78
2025-01-15 10:32:15 - __main__ - INFO - Val   | Loss: 0.5489 | Recon: 0.5231 | VQ: 0.0258 | Perplexity: 142.91
2025-01-15 10:32:15 - __main__ - INFO - Saved best model to outputs/best_model.pt
```

### What to Look For

**Good training signs**:
- Loss decreases steadily
- Perplexity increases over time (more codebook usage)
- Val loss tracks train loss (not overfitting)
- VQ loss stabilizes around 0.02-0.05

**Warning signs**:
- Perplexity stays very low (< 50) → codebook collapse
- Val loss much higher than train loss → overfitting
- Loss not decreasing after many epochs → learning rate too low or model too small

## Common Issues and Solutions

### Issue: "CUDA out of memory"
**Solution**:
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Reduce model size: `--hidden_dims 32 64 128`
- Reduce window size: `--window_size 120` (use values with many factors for auto-stride computation)

### Issue: Codebook collapse (low perplexity)
**Solution**:
- Increase codebook size: `--num_embeddings 1024`
- Decrease commitment cost: `--commitment_cost 0.1`
- Increase learning rate: `--lr 5e-3`

### Issue: Poor reconstruction quality
**Solution**:
- Increase model capacity: `--hidden_dims 128 256 512`
- Train longer: `--epochs 200`
- Check data preprocessing (NaNs, normalization)

### Issue: Training too slow
**Solution**:
- Use GPU if available
- Increase batch size: `--batch_size 64`
- Use more workers: `--num_workers 8`
- Reduce dataset size (fewer windows per fly)

## Fly-Level Splits & Reconstructions

- Generate reproducible train/validation fly lists with `generate_fly_splits` in `flies/data/prepare_data.py`:
  ```bash
  python -c "from flies.data.prepare_data import generate_fly_splits; generate_fly_splits('data/fly_data/fly_group_train.npy', val_fraction=0.1, seed=42, save_path='data/fly_data/fly_splits.json')"
  ```
- Point `train.py` at the saved JSON via `--fly_split_file data/fly_data/fly_splits.json`. The script keeps whole `(sequence_id, fly_idx)` pairs in a single split and will reuse `--train_data` for validation if you only supply filters.
- Override or fine-tune splits with `--train_fly_filter` / `--val_fly_filter`, each expecting a JSON array of `{"sequence_id": "...", "fly_idx": N}`.
- When you need to trace reconstructions back to their source, request metadata from the dataloaders by calling `create_dataloaders(..., train_include_metadata=True, val_include_metadata=True)`. Each sample then returns `(window, metadata_dict)`.
- After training, load your checkpoint, run the evaluation dataset through the model to produce reconstructions, and use the utilities in `flies/visualization/reconstruction.py` to stitch windows back into full fly trajectories or arena-wide overlays for qualitative inspection.

## Next Steps After Training

1. **Evaluate Model**:
   - Check reconstruction quality on validation set
   - Visualize reconstructed trajectories vs. original
   - Examine codebook usage distribution

2. **Extract Behavior Codes**:
   ```python
   codes = model.encode(sequences)  # (N, T) discrete codes
   ```

3. **Analyze Behavior Syllables**:
   - Cluster similar codes
   - Visualize prototypical behaviors for each code
   - Analyze temporal patterns (e.g., code transitions)
   - Study social context effects on syllable usage

4. **Downstream Analysis**:
   - Build Markov models on code sequences
   - Identify behavioral motifs
   - Compare behavior across experimental conditions
