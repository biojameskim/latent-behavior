# Architecture and Implementation Clarifications

## Question 1: Conv1d vs Conv2d - Do the Quantizers Work?

### Short Answer: **YES, they work perfectly with Conv1d**

### Explanation

**The key insight:** The quantizers (VQ, FSQ, RVQ, LFQ) operate on **embeddings**, not on convolution layers. They don't care whether the embeddings came from Conv1d, Conv2d, or a fully connected layer.

**How it works in our implementation:**

```
Your Conv1d Encoder → (B, C, T) embeddings → Quantizer → (B, C, T) quantized → Your Conv1d Decoder
                      ↑                       ↑
                    (B=batch,              (operates on
                     C=channels,            embeddings,
                     T=time)                architecture-
                                           agnostic)
```

**Shape handling:**

1. **Our encoder outputs:** `(Batch, Channels=256, Time=5)`
2. **Quantizers expect:** `(Batch, Sequence=5, Features=256)` or `(B, T, C)`
3. **Solution:** Permute before quantization, permute back after

```python
# In unified_quantizer.py - already implemented!
def _forward_vq_improved(self, z):
    # z comes in as (B, C, T) from our Conv1d encoder
    z = z.permute(0, 2, 1)  # → (B, T, C) for quantizer

    z_q, indices, loss = self.quantizer(z)  # Quantizer operates on (B, T, C)

    z_q = z_q.permute(0, 2, 1)  # → (B, C, T) for our Conv1d decoder
    return z_q, indices, loss, perplexity
```

### Why the Library Uses Conv2d in Examples

The library's examples use Conv2d because:
- **Images are 2D** (height, width)
- Their **encoders/decoders** are Conv2d for images
- But the **quantizers themselves** work with any shape

**For us:**
- **Sequences are 1D** (time)
- Our **encoders/decoders** are Conv1d for sequences
- The **same quantizers** work perfectly

### Verification

You can verify this works by checking the shape transformations:

```python
# Test with your actual architecture
from flies.vq_vae import UnifiedVQVAE

model = UnifiedVQVAE(
    input_dim=48,
    hidden_dims=[64, 128, 256],
    embedding_dim=256,
    num_embeddings=32,
    sequence_length=150,
    quantizer_method='fsq',
    quantizer_kwargs={'levels': [8, 5, 5, 5]}
)

# Test forward pass
import torch
test_input = torch.randn(4, 48, 150)  # (B, features, time)
output, vq_loss, perplexity, _, indices = model(test_input)

print(f"Input:  {test_input.shape}")  # (4, 48, 150)
print(f"Output: {output.shape}")      # (4, 48, 150) ✓ Same shape!
print(f"Success!")
```

### Bottom Line

✅ **The implementation is correct**
✅ **Conv1d encoder/decoder + any quantizer works**
✅ **Permutation handles shape differences**
✅ **No changes needed**

---

## Question 2: Can I Use Fly Split File?

### Short Answer: **YES!**

The `train_comparison.py` script fully supports fly split files, exactly like the original `train.py`.

### Usage

**Option A: Using fly split JSON file**
```bash
python flies/training/train_comparison.py \
    --train_data data/train.npy \
    --fly_split_file data/fly_splits.json \
    --train_split_name train \
    --val_split_name val \
    --methods vq fsq rvq
```

**Option B: Using separate val_data**
```bash
python flies/training/train_comparison.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --methods vq fsq rvq
```

**Option C: Both (fly split takes precedence for filtering)**
```bash
python flies/training/train_comparison.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --fly_split_file data/fly_splits.json \
    --methods vq fsq rvq
```

### Arguments Available

All the same fly-level split arguments from `train.py`:

- `--fly_split_file`: Path to JSON file with train/val/test splits
- `--train_split_name`: Key in split file for training (default: 'train')
- `--val_split_name`: Key in split file for validation (default: 'val')
- `--train_fly_filter`: Override train flies with separate JSON
- `--val_fly_filter`: Override val flies with separate JSON

### Implementation Details

The fly filtering is handled at lines 356-373 in `train_comparison.py`:

```python
# Load fly filters
train_fly_ids, val_fly_ids = load_fly_filters(
    fly_split_file=args.fly_split_file,
    train_split_name=args.train_split_name,
    val_split_name=args.val_split_name,
    train_filter_path=args.train_fly_filter,
    val_filter_path=args.val_fly_filter,
)

# Create dataloaders with filtering
if args.val_data or val_fly_ids is not None:
    train_loader, val_loader = create_dataloaders(
        train_data_file=args.train_data,
        val_data_file=args.val_data,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_fly_ids=train_fly_ids,  # ← Fly filtering applied here
        val_fly_ids=val_fly_ids        # ← Fly filtering applied here
    )
```

---

## Question 3: Windowing - I Was Wrong!

### Correction: You're Using Sequential Non-Overlapping Windows

**What you're actually doing:**
```python
# With window_size=150, stride=150
# Sequence: [0, 1, 2, ..., 449]
Window 1: [0:150]    # frames 0-149
Window 2: [150:300]  # frames 150-299
Window 3: [300:450]  # frames 300-449
# Non-overlapping, sequential splits
```

This is **NOT random** - it's systematic chunking. My apologies for the confusion!

### Current Approach is Fine for Initial Comparison

**For your initial run**, this sequential windowing is perfectly reasonable:
- ✅ Simple and reproducible
- ✅ No data leakage
- ✅ Good for comparing quantizer performance
- ✅ Easier to implement than event detection

### Event-Based Windowing for Later

**After initial comparison**, you can implement event-based windowing. Here's a more rigorous implementation:

```python
# File: flies/data/event_detection.py
import numpy as np
from typing import List, Tuple

class BehaviorEventDetector:
    """Rigorous event detection for fly behavior."""

    def __init__(
        self,
        keypoint_names: List[str],
        fps: float = 30.0,
        pixel_to_mm: float = 0.1  # Calibration: pixels to mm
    ):
        """
        Args:
            keypoint_names: List of keypoint names in order
            fps: Frames per second
            pixel_to_mm: Conversion factor from pixels to mm
        """
        self.keypoint_names = keypoint_names
        self.fps = fps
        self.pixel_to_mm = pixel_to_mm

        # Find body center index
        if 'body_center' in keypoint_names:
            self.body_idx = keypoint_names.index('body_center')
        else:
            self.body_idx = 0  # Default to first keypoint

    def detect_walking_onset(
        self,
        keypoints: np.ndarray,
        speed_threshold_mm_s: float = 5.0,  # mm/s
        min_duration_s: float = 0.2,        # seconds
        min_gap_s: float = 0.5              # min time between events
    ) -> np.ndarray:
        """
        Detect walking onset with rigorous criteria.

        Args:
            keypoints: (T, num_keypoints, 2) array
            speed_threshold_mm_s: Minimum speed to count as walking (mm/s)
            min_duration_s: Minimum duration to count as walking (seconds)
            min_gap_s: Minimum gap between walking bouts (seconds)

        Returns:
            onset_frames: (N,) array of frame indices where walking starts
        """
        # Extract body center positions
        positions = keypoints[:, self.body_idx, :]  # (T, 2)

        # Convert to mm
        positions_mm = positions * self.pixel_to_mm

        # Compute velocity and speed
        dt = 1.0 / self.fps
        velocity = np.diff(positions_mm, axis=0) / dt  # (T-1, 2) in mm/s
        speed = np.linalg.norm(velocity, axis=1)  # (T-1,) in mm/s

        # Pad to original length
        speed = np.concatenate([[0], speed])  # (T,)

        # Smooth speed to reduce noise (moving average)
        window = int(0.1 * self.fps)  # 100ms window
        if window > 0:
            speed = np.convolve(speed, np.ones(window)/window, mode='same')

        # Detect walking (speed above threshold)
        walking = speed > speed_threshold_mm_s

        # Find walking bout starts and ends
        walking_padded = np.concatenate([[False], walking, [False]])
        starts = np.where(np.diff(walking_padded.astype(int)) == 1)[0]
        ends = np.where(np.diff(walking_padded.astype(int)) == -1)[0]

        # Filter by minimum duration
        min_frames = int(min_duration_s * self.fps)
        durations = ends - starts
        valid_mask = durations >= min_frames

        starts = starts[valid_mask]
        ends = ends[valid_mask]

        # Filter by minimum gap between bouts
        min_gap_frames = int(min_gap_s * self.fps)
        if len(starts) > 1:
            gaps = starts[1:] - ends[:-1]
            # Merge bouts that are too close
            merge_mask = gaps < min_gap_frames

            # Keep only well-separated bouts
            keep_indices = [0]  # Always keep first
            for i in range(1, len(starts)):
                if gaps[i-1] >= min_gap_frames:
                    keep_indices.append(i)

            starts = starts[keep_indices]

        return starts

    def detect_turning(
        self,
        keypoints: np.ndarray,
        angle_threshold_deg: float = 45.0,
        window_frames: int = 5,
        min_gap_s: float = 0.3
    ) -> np.ndarray:
        """
        Detect sharp turns in trajectory.

        Args:
            keypoints: (T, num_keypoints, 2) array
            angle_threshold_deg: Minimum angle change (degrees)
            window_frames: Window to compute angle over
            min_gap_s: Minimum gap between turn events

        Returns:
            turn_frames: (N,) array of frame indices where turns occur
        """
        positions = keypoints[:, self.body_idx, :]

        # Compute heading direction
        velocities = np.diff(positions, axis=0)
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])

        # Compute angular velocity over window
        angle_changes = np.abs(np.diff(angles, n=window_frames))

        # Wrap to [-π, π]
        angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
        angle_changes_deg = np.degrees(angle_changes)

        # Find turns
        turn_mask = angle_changes_deg > angle_threshold_deg
        turn_frames = np.where(turn_mask)[0] + window_frames  # Offset by window

        # Filter by minimum gap
        if len(turn_frames) > 1:
            min_gap_frames = int(min_gap_s * self.fps)
            gaps = np.diff(turn_frames)
            keep_mask = np.concatenate([[True], gaps >= min_gap_frames])
            turn_frames = turn_frames[keep_mask]

        return turn_frames

    def validate_events(
        self,
        event_frames: np.ndarray,
        sequence_length: int,
        window_size: int
    ) -> np.ndarray:
        """
        Validate that events leave room for full windows.

        Args:
            event_frames: (N,) array of event frame indices
            sequence_length: Total sequence length
            window_size: Window size for training

        Returns:
            valid_events: (M,) array of valid event indices
        """
        # Event must leave room for full window
        valid_mask = event_frames + window_size <= sequence_length

        # Event must not be too close to start (optional)
        # valid_mask &= event_frames >= min_start_offset

        return event_frames[valid_mask]
```

### When to Switch to Event-Based

**Switch to event-based windowing when:**
1. ✅ You've run initial comparison and chosen best quantizer
2. ✅ You have good calibration (pixel-to-mm conversion)
3. ✅ You've validated event detection on sample sequences
4. ✅ You want to study behavior syllables (not just reconstruction)

### Example: Event-Based Dataset

```python
# File: flies/data/event_dataset.py
from torch.utils.data import Dataset
import numpy as np
import torch

class EventBasedFlyDataset(Dataset):
    """Dataset using event-based windowing."""

    def __init__(
        self,
        data_file: str,
        window_size: int = 150,
        event_type: str = 'walking',
        detector_params: dict = None
    ):
        self.data = np.load(data_file, allow_pickle=True)
        self.window_size = window_size

        # Initialize event detector
        from flies.data.event_detection import BehaviorEventDetector
        keypoint_names = ['body_center', ...]  # Your keypoint names
        self.detector = BehaviorEventDetector(keypoint_names)

        # Detect events for all sequences
        self.windows = self._detect_events(event_type, detector_params or {})

    def _detect_events(self, event_type, params):
        windows = []

        for seq_idx, sequence in enumerate(self.data):
            # sequence shape: (T, num_keypoints, 2) or flattened (T, features)
            # Reshape if needed
            if sequence.ndim == 2:
                # Assuming 24 keypoints × 2 coords = 48 features
                keypoints = sequence.reshape(len(sequence), -1, 2)
            else:
                keypoints = sequence

            # Detect events
            if event_type == 'walking':
                events = self.detector.detect_walking_onset(keypoints, **params)
            elif event_type == 'turning':
                events = self.detector.detect_turning(keypoints, **params)
            else:
                raise ValueError(f"Unknown event type: {event_type}")

            # Validate events
            valid_events = self.detector.validate_events(
                events, len(sequence), self.window_size
            )

            # Add to windows list
            for event_frame in valid_events:
                windows.append((seq_idx, event_frame))

        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.windows[idx]
        sequence = self.data[seq_idx]

        # Extract window
        window = sequence[start_frame:start_frame + self.window_size]

        # Flatten if needed
        if window.ndim == 3:
            window = window.reshape(len(window), -1)

        # Convert to (features, time) for Conv1d
        window = torch.FloatTensor(window).transpose(0, 1)

        return window
```

---

## Summary

### Question 1: Architecture ✅
- **Quantizers work with Conv1d** - they operate on embeddings, not convolutions
- **Implementation is correct** - permutation handles shape differences
- **No changes needed**

### Question 2: Fly Split File ✅
- **Yes, fully supported** in `train_comparison.py`
- **Use `--fly_split_file`** just like in original `train.py`
- **Example:**
  ```bash
  python flies/training/train_comparison.py \
      --train_data data/train.npy \
      --fly_split_file data/fly_splits.json \
      --methods vq fsq rvq
  ```

### Question 3: Windowing ✅
- **Your current approach** (sequential, non-overlapping) is fine for initial comparison
- **Not random** - my mistake, sorry for confusion!
- **Event-based detection** provided above for later use
- **Recommendation:** Run initial comparison with current windowing, then switch to event-based if needed

---

## Recommended Next Steps

1. **Run initial comparison with current setup:**
   ```bash
   python flies/training/train_comparison.py \
       --train_data <path> \
       --fly_split_file <path> \
       --methods vq vq_improved fsq \
       --epochs 20
   ```

2. **Analyze results** and choose best quantizer

3. **(Optional) Implement event-based windowing** using the rigorous detector above

4. **Retrain with event-based windows** if behavior syllables are important
