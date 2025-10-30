# Visualization Docs

Utilities for rendering MABe22 fly pose sequences live here. All scripts assume the
processed dataset lives at `../../../../data/fly_data/fly_group_train.npy` and use a
headless Matplotlib backend so they can run on servers without displays.

## Key Files
- `plot_mabe_flies.py` – Core plotting library. Exposes `plot_frame`, `plot_flies`,
  `plot_fly`, and `plot_trajectory` helpers plus the `KEYPOINT_NAMES` list and arena
  constants. Handles NaN-padded flies, trims derived keypoints when requested, and
  draws the arena boundary and skeleton edges for each pose.
- `create_video.py` – Command-line tool for generating per-frame PNGs and optionally
  assembling them into MP4 (via `ffmpeg`) or GIF (via Pillow). Useful flags include
  `--sequence`, frame range arguments, `--step` for subsampling, `--format` (`frames`,
  `video`, or `gif`), and `--fps`. See documentation in [create_video.py](create_video.py) for more control in the arguments.

  Example:
  ```
  python create_video.py --sequence 01FJRKCP4GE1W1DFX51C --start 0 --end 1000 --step 2 --format video
  ```
 - Sequence is the sequence_id
 - start and end specify the start and end frames
 - step is stride
 - format can be video, gif, or frames
 - can also specify desired fps and dpi with `--fps` and `--dpi`
  
- `example_plotting.py` – End-to-end showcase of the plotting API. Generates sample
  frame renders, trajectory overlays, pose snapshots, and diagnostic counts of valid
  keypoints. Outputs land in `test_plots/` and `animation_frames/`.

## Output Directories
- `animation_frames/` – Cache of PNG frames produced by `example_plotting.py` and
  `create_video.py`. Subdirectories group frames by sequence ID when requested.
- `animations/` – Destination for MP4/GIF files assembled by `create_video.py`.
- `test_plots/` – Gallery of static figures created by the example script (grid
  layouts, leg-tip overlays, trajectory plots, etc.).

## Dependencies
- Python packages: `numpy`, `matplotlib`, and `Pillow` (only required for GIF export).
- External tools: `ffmpeg` for MP4 creation.

## Quick Start
1. Load the dataset and explore keypoint labels:
   ```python
   import numpy as np
   from plot_mabe_flies import plot_frame

   data = np.load('../../../../data/fly_data/fly_group_train.npy', allow_pickle=True).item()
   fig, ax = plot_frame(data, sequence_id=list(data['sequences'])[0], frame_idx=0)
   fig.savefig('test_frame.png', dpi=150, bbox_inches='tight')
   ```
2. Batch-render frames or create a clip with `create_video.py` using the command above.
3. Run `python example_plotting.py` to reproduce the bundled diagnostic figures.
