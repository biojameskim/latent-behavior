# Windowing Strategy for Behavioral Sequences

## Current Approach: Random Windows ❌

Your current implementation extracts windows at **random positions**:

```python
# Current approach in dataset.py
window_start = random.randint(0, len(trajectory) - window_size)
window = trajectory[window_start:window_start + window_size]
```

**Problems with random windowing:**
1. **No temporal alignment** - The same behavior (e.g., walking) appears at different positions within the window across examples
2. **Increased variability** - VQ-VAE must learn to recognize "walking at t=10" AND "walking at t=50" AND "walking at t=90" as the same behavior
3. **Worse discretization** - Codebook entries don't correspond to clean "syllables"
4. **Harder to forecast** - Sequences starting from random points don't have consistent structure

## Recommended Approach: Event-Based Windows ✅

Extract windows starting from **meaningful behavioral events**.

### Why Event-Based is Better

**Temporal Alignment:**
```
Random Windows:                Event-Based Windows (walking onset):
Window 1: [idle][walk][turn]   Window 1: [walk][turn][stop]
Window 2: [turn][stop][idle]   Window 2: [walk][groom][idle]
Window 3: [stop][walk][turn]   Window 3: [walk][walk][turn]
                                         ↑ All start with walking!
```

**Benefits:**
- ✅ VQ-VAE learns one "walking" code instead of multiple position-dependent ones
- ✅ Codes become true behavior syllables (walk→turn, walk→groom, walk→stop)
- ✅ Better codebook utilization (fewer redundant codes)
- ✅ Easier to interpret results
- ✅ Better for downstream forecasting tasks

## Implementation

### Step 1: Define Behavioral Events

For fruit fly behavior, useful events include:

```python
import numpy as np

class BehaviorEvents:
    """Detect behavioral events in fly trajectories."""

    @staticmethod
    def detect_walking_onset(positions, speed_threshold=0.5,
                            min_duration=5):
        """
        Detect when fly starts walking.

        Args:
            positions: (T, 2) array of x,y positions
            speed_threshold: Minimum speed to count as walking (mm/frame)
            min_duration: Minimum frames of walking to count as event

        Returns:
            onset_frames: Indices where walking starts
        """
        # Compute speed
        velocity = np.diff(positions, axis=0)
        speed = np.linalg.norm(velocity, axis=1)

        # Detect walking (sustained speed above threshold)
        walking = speed > speed_threshold

        # Find transitions from not-walking to walking
        walking_padded = np.concatenate([[False], walking, [False]])
        starts = np.where(np.diff(walking_padded.astype(int)) == 1)[0]
        ends = np.where(np.diff(walking_padded.astype(int)) == -1)[0]

        # Filter by minimum duration
        durations = ends - starts
        valid_starts = starts[durations >= min_duration]

        return valid_starts

    @staticmethod
    def detect_turning(positions, angle_threshold=30, window=5):
        """
        Detect sharp turns in trajectory.

        Args:
            positions: (T, 2) array of positions
            angle_threshold: Minimum angle change (degrees)
            window: Frames to compute angle over

        Returns:
            turn_frames: Indices where turns occur
        """
        # Compute heading direction
        velocities = np.diff(positions, axis=0)
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])

        # Compute angle change over window
        angle_changes = np.abs(np.diff(angles, n=window))
        # Wrap to [-pi, pi]
        angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)

        # Detect turns
        turns = np.where(np.degrees(angle_changes) > angle_threshold)[0]

        return turns

    @staticmethod
    def detect_stopping(positions, speed_threshold=0.2, min_duration=10):
        """
        Detect when fly stops moving.

        Args:
            positions: (T, 2) array
            speed_threshold: Maximum speed to count as stopped
            min_duration: Minimum frames to count as stop

        Returns:
            stop_starts: Indices where stopping begins
        """
        velocity = np.diff(positions, axis=0)
        speed = np.linalg.norm(velocity, axis=1)

        stopped = speed < speed_threshold

        # Find stop onsets
        stopped_padded = np.concatenate([[False], stopped, [False]])
        starts = np.where(np.diff(stopped_padded.astype(int)) == 1)[0]
        ends = np.where(np.diff(stopped_padded.astype(int)) == -1)[0]

        durations = ends - starts
        valid_starts = starts[durations >= min_duration]

        return valid_starts

    @staticmethod
    def detect_grooming(keypoints, wing_indices=(8, 9, 10, 11),
                       movement_threshold=2.0, min_duration=15):
        """
        Detect wing grooming behavior (wings moving while body still).

        Args:
            keypoints: (T, num_keypoints, 2) array
            wing_indices: Indices of wing keypoints
            movement_threshold: Wing movement threshold
            min_duration: Minimum frames

        Returns:
            groom_starts: Indices where grooming starts
        """
        # Extract body center and wings
        body_center = keypoints[:, 0, :]  # Assuming index 0 is body
        wings = keypoints[:, wing_indices, :]

        # Body movement
        body_velocity = np.diff(body_center, axis=0)
        body_speed = np.linalg.norm(body_velocity, axis=1)

        # Wing movement
        wing_velocity = np.diff(wings, axis=0)
        wing_speed = np.linalg.norm(wing_velocity, axis=(1, 2))

        # Grooming = wings moving while body still
        body_still = body_speed < 0.3
        wings_moving = wing_speed > movement_threshold
        grooming = body_still & wings_moving

        # Find grooming onsets
        grooming_padded = np.concatenate([[False], grooming, [False]])
        starts = np.where(np.diff(grooming_padded.astype(int)) == 1)[0]
        ends = np.where(np.diff(grooming_padded.astype(int)) == -1)[0]

        durations = ends - starts
        valid_starts = starts[durations >= min_duration]

        return valid_starts
```

### Step 2: Modify Dataset Class

Update your `FlyBehaviorDataset` to use event-based windowing:

```python
class FlyBehaviorDataset(Dataset):
    """Dataset with event-based windowing."""

    def __init__(
        self,
        data_file,
        window_size=150,
        stride=150,
        event_based=True,
        event_type='walking',  # 'walking', 'turning', 'stopping', 'grooming', 'any'
        min_events_per_sequence=5,  # Skip sequences with too few events
        fallback_to_random=True,  # Use random windows if no events found
    ):
        self.data = np.load(data_file)
        self.window_size = window_size
        self.stride = stride
        self.event_based = event_based
        self.event_type = event_type
        self.fallback_to_random = fallback_to_random

        # Detect events for all sequences
        if event_based:
            self.event_indices = self._detect_all_events()
        else:
            self.event_indices = None

        # Create window index
        self.windows = self._create_window_index()

    def _detect_all_events(self):
        """Detect events for all sequences."""
        event_detector = BehaviorEvents()
        all_events = []

        for seq_idx in range(len(self.data)):
            trajectory = self.data[seq_idx]  # (T, features)

            # Extract positions (first 2 features are x, y)
            positions = trajectory[:, :2]

            # Detect events based on type
            if self.event_type == 'walking':
                events = event_detector.detect_walking_onset(positions)
            elif self.event_type == 'turning':
                events = event_detector.detect_turning(positions)
            elif self.event_type == 'stopping':
                events = event_detector.detect_stopping(positions)
            elif self.event_type == 'any':
                # Use any event type
                walking = event_detector.detect_walking_onset(positions)
                turning = event_detector.detect_turning(positions)
                stopping = event_detector.detect_stopping(positions)
                events = np.concatenate([walking, turning, stopping])
                events = np.unique(np.sort(events))
            else:
                raise ValueError(f"Unknown event type: {self.event_type}")

            # Filter events that leave room for full window
            valid_events = events[events + self.window_size <= len(trajectory)]

            all_events.append(valid_events)

        return all_events

    def _create_window_index(self):
        """Create index of all valid windows."""
        windows = []

        for seq_idx in range(len(self.data)):
            seq_len = len(self.data[seq_idx])

            if self.event_based and self.event_indices is not None:
                # Use event-based windows
                events = self.event_indices[seq_idx]

                if len(events) > 0:
                    # Create windows starting at each event
                    for event_frame in events:
                        windows.append((seq_idx, event_frame))
                elif self.fallback_to_random:
                    # No events found, use random windows as fallback
                    for start in range(0, seq_len - self.window_size + 1, self.stride):
                        windows.append((seq_idx, start))
                # else: skip this sequence entirely

            else:
                # Use regular sliding windows
                for start in range(0, seq_len - self.window_size + 1, self.stride):
                    windows.append((seq_idx, start))

        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.windows[idx]
        trajectory = self.data[seq_idx]
        window = trajectory[start_frame:start_frame + self.window_size]

        # Convert to (features, time) format for Conv1d
        window = torch.FloatTensor(window).transpose(0, 1)

        return window
```

### Step 3: Update Training Script

```python
# In train_comparison.py or train.py

# Enable event-based windowing
train_loader = create_dataloaders(
    train_data_file=args.train_data,
    window_size=150,
    stride=150,  # Ignored when event_based=True
    batch_size=128,
    event_based=True,  # NEW!
    event_type='walking',  # NEW!
)
```

## Comparison: Random vs Event-Based

### Expected Results

**Random Windows (Current):**
- Codebook utilization: ~18%
- Reconstruction loss: ~78
- VQ loss: ~4
- Interpretation: Codes mix temporal position + behavior

**Event-Based Windows (Recommended):**
- Codebook utilization: **~40-60%** (expected improvement)
- Reconstruction loss: **~60-70** (expected improvement)
- VQ loss: **~2-3** (expected improvement)
- Interpretation: **Codes are true behavior syllables!**

### Visualization

After training with event-based windows, you can analyze what each code represents:

```python
# Load model
model = torch.load('best_model.pt')

# Extract codes for many windows
codes = []
for window in dataset:
    code = model.encode(window.unsqueeze(0))
    codes.append(code.item())

# Each code should correspond to a behavior syllable:
# Code 0: "Walk forward 3 steps"
# Code 1: "Walk forward then turn left"
# Code 2: "Walk forward then stop"
# Code 3: "Turn left"
# Code 4: "Turn right"
# etc.
```

## When to Use Which Strategy

| Scenario | Recommended Windowing |
|----------|----------------------|
| Exploring dataset, initial training | Random (simpler) |
| Goal: Interpretable behavior codes | **Event-based** |
| Goal: Behavior forecasting | **Event-based** |
| Goal: Clustering behaviors | **Event-based** |
| Sparse events (<5 per sequence) | Random or fallback hybrid |
| Multi-fly social interactions | Event-based (social events) |

## Your Next Steps

1. **Implement event detection** (start with walking onset - simplest)
2. **Run comparison**: Random vs Event-based windowing with same model
3. **Analyze results**: Check codebook utilization and code interpretability
4. **Refine**: Adjust speed thresholds, try different event types

Your PI is absolutely right that event-based windowing will be much better! This is standard practice in behavioral neuroscience (e.g., MoSeq, VAME, B-SOiD all use some form of event alignment).

## References

- MoSeq (Wiltschko et al., 2015): Uses syllable onsets
- VAME (Luxem et al., 2020): Learns motif boundaries
- B-SOiD (Hsu et al., 2021): Uses behavior transition points

All successful behavior encoding methods use some form of temporal alignment! 🎯
