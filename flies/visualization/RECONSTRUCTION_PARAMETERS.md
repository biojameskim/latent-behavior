# Reconstruction Visualization Parameters Guide

## Quick Reference

When you run:
```bash
python visualize_reconstructions.py \
    --checkpoint ../training/outputs/rvq_deep/best_model.pt \
    --data_file ../../data/fly_data/fly_group_train.npy \
    --output_dir rvq_recon_viz
```

## The Four Key Parameters

### 1. `--window_overlays` (default: 5)

**What it controls**: How many **individual window comparisons** to save as images

**What a "window" is**:
- One behavior snippet (e.g., 150 frames)
- The model processes data in these windows
- Each window is independently encoded → quantized → decoded

**What gets visualized**:
- Original (blue) vs Reconstructed (orange) fly poses
- Side-by-side comparison for quality inspection
- Shows detailed reconstruction quality on individual segments

**Example**:
```bash
--window_overlays 5  # Default: saves 5 window comparison images
--window_overlays 20 # Saves 20 window comparison images
```

**Output files**:
```
rvq_recon_viz/window_overlays/
├── window_0000.png  # 1st window comparison
├── window_0001.png  # 2nd window comparison
├── window_0002.png  # 3rd window comparison
├── window_0003.png  # 4th window comparison
└── window_0004.png  # 5th window comparison (if --window_overlays 5)
```

**Each image shows**:
- Multiple frames from one window (controlled by `--window_frames`)
- Blue skeleton = original behavior
- Orange skeleton = reconstructed behavior
- Closer overlap = better reconstruction

---

### 2. `--window_frames` (default: [0, -1])

**What it controls**: Which **frames within each window** to show

**Why it matters**:
- Windows have 150 frames
- Showing all 150 would be overwhelming
- So we pick representative frames

**Default behavior** `[0, -1]`:
- Frame 0 = first frame of window
- Frame -1 = last frame of window (Python negative indexing)
- Shows beginning and end of behavior snippet

**Example values**:
```bash
--window_frames 0 -1           # Default: first and last frame
--window_frames 0 74 149       # First, middle, last frame
--window_frames 0 50 100 149   # Four snapshots across window
```

**How it works**:
```
Window with 150 frames: [0, 1, 2, ..., 148, 149]

--window_frames 0 -1
→ Shows frames 0 and 149

--window_frames 0 74 149
→ Shows frames 0, 74, and 149
```

**Output appearance**:
```
window_0000.png:
┌─────────┬─────────┬─────────┐
│ Frame 0 │ Frame 74│ Frame149│  ← Same window, different time points
│  Blue   │  Blue   │  Blue   │  ← Original
│ Orange  │ Orange  │ Orange  │  ← Reconstructed
└─────────┴─────────┴─────────┘
```

---

### 3. `--max_sequences` (default: 5)

**What it controls**: How many **full trajectory arena views** to save

**What a "sequence" is**:
- A complete recording session (one video file)
- Contains multiple flies in an arena
- Thousands of frames long (e.g., 4500 frames)
- Windows are stitched back together to reconstruct full trajectories

**What gets visualized**:
- Full arena view with ALL flies
- Original (solid, high opacity) vs Reconstructed (transparent, low opacity)
- Shows how well stitched windows recreate continuous behavior

**Example**:
```bash
--max_sequences 5  # Default: saves 5 sequence arena views
--max_sequences 10 # Saves 10 sequence arena views
--max_sequences -1 # Saves ALL sequences (could be many!)
```

**Output files**:
```
rvq_recon_viz/sequence_overlays/
├── sequence_abc123_frame_0.png      # Sequence abc123, frame 0
├── sequence_abc123_frame_100.png    # Sequence abc123, frame 100
├── sequence_def456_frame_0.png      # Sequence def456, frame 0
└── ...
```

**Each image shows**:
- Overhead view of the arena
- All flies visible (if multiple flies tracked)
- Original poses (solid/opaque)
- Reconstructed poses (transparent) overlaid

---

### 4. `--sequence_frames` (default: [0])

**What it controls**: Which **frames from the full sequence** to show in arena view

**Why it matters**:
- Sequences are long (4500 frames = 150 seconds at 30fps)
- Can't show all frames as images
- Pick representative snapshots

**Default behavior** `[500]`:
- Shows frame 500 (avoids frame 0 centering artifact)
- Frame 0 typically has all flies centered at origin due to data preprocessing

**Example values**:
```bash
--sequence_frames 500            # Default: single snapshot at frame 500
--sequence_frames 0              # First frame (all flies centered at origin)
--sequence_frames 500 1500 2500  # Three snapshots: early, middle, late
--sequence_frames 100 500 1000 1500 2000  # Five snapshots across sequence
```

**How it works**:
```
Sequence with 4500 frames: [0, 1, 2, ..., 4498, 4499]

--sequence_frames 500
→ Shows frame 500 only (default)

--sequence_frames 500 1500 2500
→ Shows frames 500, 1500, 2500
→ Creates 3 images per sequence
```

**Output files per sequence**:
```bash
# With --sequence_frames 500 1500 2500 and 2 sequences:
sequence_abc123_frame_500.png
sequence_abc123_frame_1500.png
sequence_abc123_frame_2500.png
sequence_def456_frame_500.png
sequence_def456_frame_1500.png
sequence_def456_frame_2500.png
```

---

## Complete Example

```bash
python visualize_reconstructions.py \
    --checkpoint ../training/outputs/rvq_deep/best_model.pt \
    --data_file ../../data/fly_data/fly_group_train.npy \
    --output_dir rvq_recon_viz \
    --window_overlays 10 \          # Save 10 window comparisons
    --window_frames 0 74 149 \      # Show 3 frames per window
    --max_sequences 5 \             # Save 5 sequence arena views
    --sequence_frames 500 1500 2500 # Show 3 frames per sequence (avoids frame 0)
```

**This produces**:

```
rvq_recon_viz/
├── window_overlays/
│   ├── window_0000.png  ← 3 frames (0, 74, 149) from 1st window
│   ├── window_0001.png  ← 3 frames from 2nd window
│   ├── ...
│   └── window_0009.png  ← 3 frames from 10th window
│
└── sequence_overlays/
    ├── seq1_frame_500.png    ← Arena view, frame 500
    ├── seq1_frame_1500.png   ← Arena view, frame 1500
    ├── seq1_frame_2500.png   ← Arena view, frame 2500
    ├── seq2_frame_500.png
    ├── seq2_frame_1500.png
    ├── seq2_frame_2500.png
    └── ... (3 images × 5 sequences = 15 images)
```

**Total images**: 10 window overlays + 15 sequence overlays = 25 images

---

## Visual Summary

### Window Overlays (Close-up Detail)
```
┌──────────────────────────────────────┐
│  Window 0000: Frames 0, 74, 149      │
│  ┌─────┐  ┌─────┐  ┌─────┐          │
│  │  0  │  │ 74  │  │ 149 │          │
│  │ 🦟  │  │ 🦟  │  │ 🦟  │  Blue    │
│  │ 🦟  │  │ 🦟  │  │ 🦟  │  Orange  │
│  └─────┘  └─────┘  └─────┘          │
│  One behavior window, detailed view  │
└──────────────────────────────────────┘
```

### Sequence Overlays (Arena View)
```
┌──────────────────────────────────────┐
│  Sequence abc123: Frame 0            │
│  ┌─────────────────────────────┐    │
│  │  🦟  🦟      🦟              │    │
│  │      🦟         🦟   🦟      │    │
│  │  🦟        🦟                │    │
│  │         🦟          🦟  🦟   │    │
│  └─────────────────────────────┘    │
│  All flies in arena, one time point  │
└──────────────────────────────────────┘
```

---

## Parameter Cheat Sheet

| Parameter | What | Default | Common Values |
|-----------|------|---------|---------------|
| `--window_overlays` | # of window images to save | 5 | 5, 10, 20, 50 |
| `--window_frames` | Frames to show per window | [0, -1] | [0, -1], [0, 74, 149], [0, 50, 100, 149] |
| `--max_sequences` | # of sequence arena views | 5 | 5, 10, -1 (all) |
| `--sequence_frames` | Frames to show per sequence | [500] | [500], [500, 1500], [100, 500, 1000] |

---

## Tips

### For Quick Quality Check
```bash
# Minimal: just look at a few windows
--window_overlays 5 \
--window_frames 0 -1 \
--max_sequences 0  # Skip sequence overlays
```

### For Comprehensive Analysis
```bash
# Look at many windows and sequences
--window_overlays 50 \
--window_frames 0 37 74 111 149 \  # 5 snapshots per window
--max_sequences 10 \
--sequence_frames 0 500 1000 1500 2000 2500
```

### For Debugging Specific Issues
```bash
# Focus on temporal progression
--window_overlays 10 \
--window_frames 0 30 60 90 120 149 \  # Dense temporal sampling
--max_sequences 3
```

---

## What Actually Gets Reconstructed

**Important**: The script reconstructs **ALL validation windows**, not just what's visualized!

| Stage | What Happens |
|-------|-------------|
| **1. Reconstruction** | ALL validation windows are reconstructed (could be 1000s) |
| **2. Stitching** | Windows stitched into full trajectories (up to 4500 frames/fly) |
| **3. Visualization** | Only a **sample** is saved as images (controlled by parameters above) |

So even with `--window_overlays 5`, the model still reconstructs everything. The parameters just control what you **see** as images.

---

## Where to Find Your Results

After running the script, look in your `--output_dir`:

```bash
cd rvq_recon_viz/

# Window comparisons (detailed, close-up)
ls window_overlays/
# → window_0000.png, window_0001.png, ...

# Sequence arena views (big picture)
ls sequence_overlays/
# → sequence_id_frame_0.png, sequence_id_frame_100.png, ...
```

Open the PNG files to visually inspect reconstruction quality!
