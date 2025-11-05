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
  python create_video.py \
  --sequence 01FJRKCP4GE1W1DFX51C \
  --start 0 \
  --end 1000 \
  --step 2 \
  --format video
  ```
  - Sequence is the sequence_id
  - start and end specify the start and end frames
  - step is stride
  - format can be video, gif, or frames
  - can also specify desired fps and dpi with `--fps` and `--dpi`
  
- `example_plotting.py` – End-to-end showcase of the plotting API. Generates sample
  frame renders, trajectory overlays, pose snapshots, and diagnostic counts of valid
  keypoints. Outputs land in `test_plots/` and `animation_frames/`.
- `reconstruction.py` – Utilities for turning VQ-VAE reconstructions back into stitched fly trajectories and arena overlays. Given original windows, reconstructed windows, and the metadata emitted by `FlyKeypointDataset(include_metadata=True)`, it provides:
  - `group_windows_by_fly`, `stitch_fly_windows`, `assemble_sequences` to rebuild `(4500, 24, 2)` fly tracks and `(4500, 11, 24, 2)` arena tensors.
  - `plot_window_overlay` for side-by-side window diagnostics.
  - `plot_sequence_overlay` for arena-wide comparisons at a chosen frame.
- `visualize_reconstructions.py` – Command-line wrapper that automates the full qualitative evaluation workflow.
  ```bash
  python visualize_reconstructions.py \
      --data_file ../../../data/fly_data/fly_group_train.npy \
      --checkpoint ../training/outputs/<run>/best_model.pt \
      --fly_split_file ../data/fly_data/fly_splits.json \
      --val_split_name val \
      --output_dir recon_viz
  ```
  **What it does**
  1. Loads the specified checkpoint, recreating the VQ-VAE with the saved hyperparameters.
  2. Filters the `.npy` data to the held-out flies using `--fly_split_file` or `--val_fly_filter`.
  3. Runs inference with `FlyKeypointDataset(include_metadata=True)` to collect originals, reconstructions, and metadata.
  4. Stitches windows back into full-length trajectories (`reconstruction.py`) and writes plots to disk.

  **What the options control**
  - `--window_overlays 4 --window_frames 0 74 149` → render four held-out windows; within each window overlay the reconstruction on the first, middle, and last frame (0, 74, 149 for a 150-frame clip).
  - `--sequence_frames 1200 2400 --max_sequences 3` → after windows are stitched back together, draw arena overlays for global frames 1,200 and 2,400, limited to the first three validation sequences.
  - `--save_pt stitched.pt` → persist the stitched originals/reconstructions (plus metadata) via `torch.save` for downstream analysis.

  **Output layout**
  ```
  reconstruction_viz/
  ├── window_overlays/
  │   ├── window_0000.png
  │   └── …
  ├── sequence_overlays/
  │   ├── <sequence_id>_frame_0.png
  │   └── …
  └── stitched.pt  # only if --save_pt was provided
  ```
  Originals are drawn with opaque skeletons; reconstructions reuse the same color but lighter, with “x” markers to highlight discrepancies.
- `visualize_codebook_embeddings.py` – Decodes every codebook entry (and optional user-supplied code sequences) through the trained decoder to catalogue the learned behavior “syllables.”
  ```bash
  python visualize_codebook_embeddings.py \
      --checkpoint ../training/outputs/<run>/best_model.pt \
      --output_dir codebook_viz \
      --frame_indices 0 74 149
  ```
  Generates figures in `codebook_viz/codebook_embeddings/`, one per embedding, showing the synthetic poses at the requested frame indices when a window is filled entirely with that code. Pass `--codes "[[12,42,17,17,12]]"` (JSON list) to inspect composite sequences.

## Output Directories
- `animation_frames/` – Cache of PNG frames produced by `example_plotting.py` and
  `create_video.py`. Subdirectories group frames by sequence ID when requested.
- `animations/` – Destination for MP4/GIF files assembled by `create_video.py`.
- `test_plots/` – Gallery of static figures created by the example script (grid
  layouts, leg-tip overlays, trajectory plots, etc.).
- `recon_viz/` - Reconstruction visualizations

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
