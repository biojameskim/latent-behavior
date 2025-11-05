# Dataset Documentation

## Overview

This directory contains data preprocessing and dataset utilities for fly behavior analysis with VQ-VAE. The pipeline converts raw multi-fly tracking data into individual fly trajectories with sliding windows suitable for temporal sequence modeling.

## Data Flow

```
Raw .npy file (multi-fly sequences)
    ↓ preprocessing.py: load_and_preprocess_for_vqvae()
Individual fly trajectories (list of dicts)
    ↓ dataset.py: FlyKeypointDataset
Sliding windows in tensor format
    ↓ dataset.py: create_dataloaders()
PyTorch DataLoaders ready for training
```

## Modules

### 1. `preprocessing.py`

Contains data loading and preprocessing logic.

#### `load_and_preprocess_for_vqvae(data_file, allowed_fly_ids=None)`

Loads raw fly tracking data and prepares it for VQ-VAE training. If you pass a set of `(sequence_id, fly_idx)` tuples via `allowed_fly_ids`, the function keeps only those trajectories while maintaining the NaN filtering.

**Input format**: `.npy` file containing:
```python
{
    'sequences': {
        'sequence_id_1': {
            'keypoints': np.array (n_frames, 11, 24, 2)  # 11 flies, 24 keypoints, x/y coords
        },
        'sequence_id_2': { ... },
        ...
    }
}
```

**Processing steps**:
1. Splits multi-fly sequences into individual fly trajectories
   - Rationale: Learn individual behavior syllables rather than group behaviors
   - Social context can be analyzed post hoc by examining syllable sequences
2. Removes flies with any NaN values (dropped entirely, not imputed)
   - Ensures clean training data without tracking artifacts

**Output**: List of dicts, each containing:
```python
{
    'keypoints': np.array (n_frames, 24, 2),  # Single fly trajectory
    'sequence_id': str,                        # Original sequence ID
    'fly_idx': int                             # Fly index in original data (0-10)
}
```

**Example usage**:
```python
from data.preprocessing import load_and_preprocess_for_vqvae

trajectories = load_and_preprocess_for_vqvae('path/to/fly_group_train.npy')
print(f"Loaded {len(trajectories)} fly trajectories")
print(f"Each trajectory shape: {trajectories[0]['keypoints'].shape}")
# Output: (4500, 24, 2) for full-length sequences

# Keep only selected flies
allowed = {('ABC123', 0), ('XYZ789', 4)}
subset = load_and_preprocess_for_vqvae('path/to/fly_group_train.npy', allowed_fly_ids=allowed)
```

**Statistics** (example from training set):
- Input: 426 sequences with 11 flies each = 4,686 total flies
- Output: 4,025 flies (removed 661 with NaN values)
- All kept trajectories: 4,500 frames with no NaNs

### 2. `dataset.py`

PyTorch Dataset class and DataLoader utilities.

#### `FlyKeypointDataset`

Creates sliding windows from fly trajectories for batch training.

**Constructor arguments**:
- `all_fly_trajectories`: List of dicts from `load_and_preprocess_for_vqvae()`
- `window_size`: Number of frames per window (default: 150)
- `stride`: Stride for sliding windows (default: 75)
- `include_metadata`: Whether to return `(window, metadata_dict)` (default: `False`)

**Windowing behavior**:
- Creates overlapping or non-overlapping windows based on stride
- `stride = window_size` → non-overlapping (recommended)
- `stride < window_size` → overlapping (more data but potential leakage)
- Skips trajectories shorter than `window_size`

**Output shape**: `(batch, features, time)` = `(B, 48, window_size)`
- 48 features = 24 keypoints × 2 coordinates (x, y)
- Transposed to `(features, time)` for Conv1d compatibility

**Methods**:
- `__len__()`: Returns total number of windows
- `__getitem__(idx)`: Returns window tensor (or `(window, metadata)` if `include_metadata=True`)
- `get_metadata(idx)`: Returns dict with `sequence_id`, `fly_idx`, `window_idx`

**Example usage**:
```python
from data.preprocessing import load_and_preprocess_for_vqvae
from data.dataset import FlyKeypointDataset

# Load and preprocess
trajectories = load_and_preprocess_for_vqvae('train.npy')

# Create dataset with non-overlapping windows
dataset = FlyKeypointDataset(
    trajectories,
    window_size=150,
    stride=150,           # Non-overlapping
    include_metadata=True
)

print(f"Total windows: {len(dataset)}")
window, metadata = dataset[0]
print(window.shape)       # (48, 150)
print(metadata)           # {'sequence_id': 'ABC123', 'fly_idx': 3, 'window_idx': 0}
```

#### `create_dataloaders()`

Convenience function to create train/val DataLoaders in one call.

**Arguments**:
- `train_data_file`: Path to training .npy file
- `val_data_file`: Path to validation .npy file (optional; if omitted you can still create a validation loader by passing `val_fly_ids`) 
- `window_size`: Window size for Dataset (default: 150)
- `stride`: Stride for Dataset (default: 75)
- `batch_size`: Batch size for DataLoader (default: 32)
- `num_workers`: Number of DataLoader workers (default: 4)
- `train_fly_ids`, `val_fly_ids`: Optional iterables of `(sequence_id, fly_idx)` pairs to filter each split.
- `train_include_metadata`, `val_include_metadata`: Forwarded to `FlyKeypointDataset` to return metadata alongside windows.

**Returns**:
- If `val_data_file` provided: `(train_loader, val_loader)`
- If no validation data: `train_loader` only

**Example usage**:
```python
from data.dataset import create_dataloaders

# Create loaders using fly-level filters stored in JSON
import json

split_json = json.load(open('data/fly_data/fly_splits.json'))
train_ids = {(item['sequence_id'], item['fly_idx']) for item in split_json['train']}
val_ids = {(item['sequence_id'], item['fly_idx']) for item in split_json['val']}

train_loader, val_loader = create_dataloaders(
    train_data_file='data/fly_group_train.npy',
    val_data_file=None,  # reuse the same file; val_fly_ids selects the held-out flies
    window_size=150,
    stride=150,
    batch_size=32,
    num_workers=4,
    train_fly_ids=train_ids,
    val_fly_ids=val_ids,
    train_include_metadata=True,
    val_include_metadata=True,
)

windows, metadata = next(iter(train_loader))
print(windows.shape)               # (32, 48, 150)
print(metadata['sequence_id'][0])  # e.g., 'ABC123'
```

### 3. `prepare_data.py`

Helper utilities that wrap preprocessing/dataset functionality.

- `generate_fly_splits(data_file, val_fraction=0.1, seed=0, save_path=None)`:
  - Builds reproducible `{train, val}` fly lists by shuffling `(sequence_id, fly_idx)` pairs.
  - Ensures entire flies stay in a single split (no leakage across windows).
  - Optionally saves a JSON file compatible with `train.py --fly_split_file`.
- `create_windows` / `prepare_dataset`: Lower-level window generators mainly used for diagnostics or exporting windowed datasets.

## Data Shape Transformations

Understanding the shape changes through the pipeline:

```
Raw data:        (n_frames, 11, 24, 2)      # Multi-fly sequences
    ↓ Split by fly
Single fly:      (n_frames, 24, 2)          # One fly's trajectory
    ↓ Create windows
Windows:         (n_windows, window_size, 24, 2)  # Sliding windows
    ↓ Flatten coordinates
Flattened:       (n_windows, window_size, 48)     # 24*2 = 48 features
    ↓ Transpose for Conv1d
Dataset output:  (n_windows, 48, window_size)     # (features, time)
    ↓ DataLoader batching
Batch:           (batch_size, 48, window_size)    # Ready for model
```

**Why transpose to (features, time)?**
- PyTorch Conv1d expects input shape: `(batch, channels, length)`
- Our "channels" are the 48 keypoint coordinates
- Our "length" is the temporal sequence (window_size frames)

## Dataset Statistics

### Window Size Considerations

For `window_size=150` at 30 fps:
- 150 frames = 5 seconds of behavior
- Long enough to capture complete behavioral motifs
- Short enough to avoid concatenating multiple distinct behaviors

**Window size and VQ-VAE architecture**:
- The VQ-VAE automatically computes strides to evenly divide the sequence length
- For `window_size=150` with 3 layers: strides=[5,3,2] → 150→30→10→5→10→30→150
- **Choose window sizes with many factors** (e.g., 120, 144, 150, 160, 192, 200)
- Avoid prime numbers or values with few factors (e.g., 97, 101, 127)
- The model will warn if the sequence length is incompatible with the architecture

### Stride Recommendations

**Non-overlapping** (`stride = window_size = 150`):
- **Pros**: Clean train/val split, no data leakage, faster training
- **Cons**: Less total training data
- **Recommended for**: Final training runs, published results

**50% overlap** (`stride = 75`):
- **Pros**: 2x more training windows
- **Cons**: Adjacent windows highly correlated, validation leakage
- **Recommended for**: Early experimentation if data is limited

**No overlap recommended** for VQ-VAE:
- Discrete codebook benefits from diverse, independent examples
- Overlapping windows yield nearly identical encodings
- Can lead to overfitting on temporal continuity rather than learning distinct syllables

### Expected Dataset Sizes

With 4,025 flies, 4,500 frames each, `window_size=150`:

| Stride | Windows per fly | Total windows | Storage (approx) |
|--------|-----------------|---------------|------------------|
| 150 (non-overlap) | 30 | ~120,750 | ~1.8 GB |
| 75 (50% overlap) | 60 | ~241,500 | ~3.6 GB |
| 50 (66% overlap) | 89 | ~358,225 | ~5.4 GB |

## Data Quality Checks

### Handling NaN Values

The preprocessing removes entire fly trajectories with any NaN values because:
1. **VQ-VAE is sensitive to input quality**: NaNs would disrupt training
2. **Imputation is risky**: Could introduce artificial patterns the model learns
3. **Sufficient data**: Even after removing 14% of flies, we have 4,025 clean trajectories

**Alternative strategies** (not implemented):
- Per-frame filtering: Keep good frames from flies with partial NaNs
- Imputation: Interpolate missing keypoints (risky for learning)
- Masking: Train model to handle missing data (complex)

### Trajectory Length Filtering

The `_create_windows()` method skips trajectories shorter than `window_size`:
```python
if num_frames < self.window_size:
    return None  # Skip this trajectory
```

This ensures all windows have the full temporal context for learning.

## Memory Considerations

### Dataset Loading Strategy

The `FlyKeypointDataset` loads **all windows into memory** during initialization:
```python
self.windows = torch.FloatTensor(np.array(self.windows))
```

**Pros**:
- Fast training (no I/O bottleneck)
- Simple implementation

**Cons**:
- High memory usage (~2-6 GB depending on stride)

**If memory is limited**, consider:
1. Lazy loading: Load windows on-demand in `__getitem__()`
2. Memory mapping: Use `np.memmap` for large arrays
3. Reduce stride (fewer windows)
4. Reduce window_size

### DataLoader Workers

`num_workers=4` parallelizes data loading:
- **More workers**: Faster data loading, higher memory usage
- **Fewer workers**: Lower memory, potential CPU bottleneck
- **Rule of thumb**: Start with `num_workers = min(4, num_cpu_cores)`

## Integration with Training

The dataset integrates seamlessly with the training script:

```python
# In train.py
from data.dataset import create_dataloaders

train_loader, val_loader = create_dataloaders(
    train_data_file=args.train_data,
    val_data_file=args.val_data,
    window_size=args.window_size,
    stride=args.stride,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        # batch shape: (batch_size, 48, window_size)
        x_recon, vq_loss, perplexity, _, _ = model(batch)
        # ... compute loss and backprop
```

## Extending the Dataset

### Adding Data Augmentation

Potential augmentations for fly behavior:
```python
def __getitem__(self, idx):
    window = self.windows[idx]

    # Time reversal (if behavior is time-symmetric)
    if random.random() < 0.5:
        window = torch.flip(window, dims=[1])

    # Small temporal shifts
    # Gaussian noise (careful not to break behavior structure)

    return window
```

### Normalization (Implemented)

**All trajectories are automatically normalized to a canonical reference frame** in `preprocessing.py`:

```python
def normalize_trajectory(keypoints):
    """
    Normalize to canonical frame:
    - Center: Initial body position (keypoint 19) at (0, 0)
    - Rotation: Initial orientation facing upward (positive y)
    """
    # Uses keypoint 19 (ellipse_center) for translation
    # Uses keypoint 20 (ellipse_orientation) for rotation
```

**Benefits**:
- Makes behavior codes rotation-invariant
- More efficient than rotation augmentation
- Removes arbitrary spatial variance (absolute position/orientation in arena)
- All flies start in consistent reference frame for learning

**Note**: This is applied during data loading in `load_and_preprocess_for_vqvae()`, so all downstream training/validation/inference automatically uses normalized coordinates

## Troubleshooting

### Issue: "Dataset is empty"
**Cause**: All trajectories are shorter than `window_size`
**Solution**: Reduce `window_size` or check data loading

### Issue: "Out of memory during dataset creation"
**Cause**: Too many windows loaded into RAM
**Solution**: Increase `stride`, reduce dataset size, or implement lazy loading

### Issue: "DataLoader is slow"
**Cause**: Too few workers or I/O bottleneck
**Solution**: Increase `num_workers` or use SSD storage

### Issue: "Validation loss is suspiciously close to training loss"
**Cause**: Overlapping windows causing data leakage
**Solution**: Use `stride = window_size` for non-overlapping windows
