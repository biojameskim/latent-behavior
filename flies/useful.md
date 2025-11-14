# Useful details

Some useful commands I've been using:

## 1. Training the vanilla VQ-VAE model + some variations
I used `train_comparison.py` first to identify which method family works best, then used `train_rvq_fsq_comparison.py` to optimize RVQ/FSQ configurations since they were the best.

### General model training
```
python train_comparison.py \
--train_data ../../../../data/fly_data/fly_group_train.npy \
--val_data ../../../../data/fly_data/fly_group_train.npy \
--fly_split_file ../data/fly_data/fly_split.json \
\

--methods vq vq_improved fsq rvq lfq

\
--window_size 150 \
--stride 75 \
\
--embedding_dim 128 \
--num_embeddings 32 \
--num_residual_blocks 2 \
--commitment_cost 0.25 \
\
--batch_size 128 \
--epochs 20 \
--lr 1e-4 \
--weight_decay 1e-5 \
--lr_scheduler_tmax 15 \
\
--output_dir ./outputs/comparison \
--save_every 5
```
- **Purpose:** General comparison script for all 5 quantization methods
- **`--methods` available:** `vq`, `vq_improved`, `fsq`, `rvq`, `lfq`
- **Default hyperparameters:** More general (e.g., embedding_dim=256, lr=1e-4, batch_size=128, epochs=100)

### Fine-tuning between the RVQ/FSQ models 
- I was using this to compare different versions of rvq and fsq because they were the best out of all the VQ-VAE variations I trained. But I've also just been using this script to train the rvq_deep (the best one) which you can see I'm controlling using the `--methods` flag.
```
python train_rvq_fsq_comparison.py \
    --train_data ../../../../data/fly_data/fly_group_train.npy \
    --val_data ../../../../data/fly_data/fly_group_train.npy \
    --fly_split_file ../data/fly_data/fly_split.json \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-4 \
    --embedding_dim 128 \
    --num_embeddings 32 \
    --commitment_cost 0.25 \
    --output_dir outputs/rvq_fsq_comparison \
    --methods rvq_deep
```
- **Purpose:** Focused comparison of RVQ and FSQ variations only
- **`--methods` available:** `rvq_base`, `rvq_deep`, `rvq_large`, `rvq_xlarge`, `rvq_shared`, `rvq_shallow_large`, `fsq_base`, `fsq_small`, `fsq_large`, `fsq_balanced`, `fsq_minimal`, `fsq_xlarge`
- **Default hyperparameters:** Different defaults (e.g., embedding_dim=128, lr=1e-3, batch_size=32, epochs=20)

## 2. Generating figures for model training results
```
python analyze_comparison.py \
    --results_dir ../training/outputs/rvq_fsq_comparison
```

## 3. Visualizations
```
python visualize_reconstructions.py \
	--checkpoint ../training/outputs/rvq_fsq_comparison/rvq_deep/best_model.pt \
	--data_file ../../../../data/fly_data/fly_group_train.npy \
	--output_dir rvq_recon_viz_denorm \
	--window_overlays 10 \
	--window_frames 0 74 149 \
	--max_sequences 5 \
	--sequence_frames 0 500 1000 \
	--denormalize
```
- The `--denormalize` flag here is important. When creating the vq codebook, I want all the flies to start at (0,0) facing up so that they're normalized. But when reconstructing the visualizations, the multi-fly arena layout is meaningless because the spatial relationships between the flies is meaningless. So it's good to denormalize this for visualization. You can remove this flag, and you'll just get an egocentric view. Could be useful but don't try to interpret the spatial relationships between flies then.

There's **two types of visualizations**, each controlled by different parameters:

---

### **3a. Window Overlays (Close-up, per-fly snippets)**

Short, detailed clips (each = 150 frames) of **one fly** vs its reconstruction.

* **`window_overlays`**

  * *What it controls:* How many snippet windows to save.
  * *Example:* `window_overlays = 5` → save 5 different 150-frame windows.
  * *What you see:* Close-up reconstruction quality on individual behaviors.

* **`window_frames`**

  * *What it controls:* Which frames **within each 150-frame window** to visualize.
  * *Why needed:* Too many frames to save all 150.
  * *Example:* `window_frames = [0, 75, 149]` → show first, middle, last frame.
  * *What you see:* Snapshot comparisons (ground truth vs reconstruction) for each chosen frame.

**Together:**
`window_overlays = 5` and `window_frames = [0, 75, 149]` →
**5 windows × 3 frames each = 15 close-up comparison images**

---

### **3b. Sequence Overlays (Full-arena, all flies)**

Large-scale overview of the **entire 4500-frame trajectory**, showing **all flies** moving in the arena.

* **`max_sequences`**

  * *What it controls:* How many full-trajectory sequences to visualize.
  * *Example:* `max_sequences = 5` → show 5 different arenas.

* **`sequence_frames`**

  * *What it controls:* Which frame indices from each 4500-frame sequence to snapshot.
  * *Why needed:* Saving all 4500 frames is impossible/too heavy.
  * *Example:* `sequence_frames = [0, 1000, 2000]` → snapshot 3 points in time.
  * *What you see:* Arena-wide fly positions at chosen timestamps.

**Together:**
`max_sequences = 5` and `sequence_frames = [0, 1000, 2000]` →
**5 sequences × 3 frames each = 15 arena-wide trajectory images**

---
### Video Generation
- You can also create videos with the visualizations generated.
- `create_overlay_video.py` – Generate videos showing VQ-VAE reconstruction overlays over time. Combines the reconstruction overlay functionality with video creation to visualize how reconstruction quality evolves throughout a sequence.
  ```bash
  # Create sequence overlay video (full arena view)
  python create_overlay_video.py \
    --data_file ../../../../data/fly_data/fly_group_train.npy \
    --checkpoint ../training/outputs/rvq_fsq_comparison/rvq_deep/best_model.pt \
  	--output_dir ./overlay_videos
    --overlay_type sequence \
    --sequence_id 01FJRKCP4GE1W1DFX51C \
    --start_frame 0 \
    --end_frame 1000 \
    --frame_step 15 \
    --fps 10 \
    --denormalize

  # Create window overlay video (individual fly)
  python create_overlay_video.py \
	--data_file ../../../../data/fly_data/fly_group_train.npy \
	--checkpoint ../training/outputs/rvq_fsq_comparison/rvq_deep/best_model.pt \
	--output_dir ./overlay_videos
	--overlay_type window \
	--sequence_id 01FJRKCP4GE1W1DFX51C \
	--fly_idx 0 \
	--start_frame 0 \
	--end_frame 1000 \
	--frame_step 15 \
	--fps 10
  ```
  **What it does under the hood**
  1. Loads VQ-VAE checkpoint and runs inference on validation data.
  2. Generates overlay frames at regular intervals (e.g., every 15 frames via `--frame_step`).
  3. Creates video using FFmpeg with configurable framerate (`--fps`).
  4. Supports both sequence overlays (full arena with all flies) and window overlays (individual fly windows).

  **Useful flags**
  - `--overlay_type {sequence,window}` → Choose between full arena view or individual fly window.
  - `--frame_step 15` → Generate overlay every N frames (default: 15).
  - `--fps 10` → Video framerate (default: 10).
  - `--denormalize` → For sequence overlays, show true spatial coordinates instead of ego-centric.
  - `--keep_frames` → By default, the individual frame PNG files are automatically deleted after the video is created. Use this flag if you want to keep them.
  - `--dpi 150` → Resolution for frame images (default: 150).

  **Output**
  - Video file named `<sequence_id>_sequence_overlay.mp4` or `<sequence_id>_fly<N>_window_overlay.mp4` in the output directory.
 
  **Sample Sequences**
  - Here's 10 sequence ids to try out:
		```
		01FJRKCP4GE1W1DFX51C
		0ARFX7NW5OPBHY1YD7BO
		0DGRG61QOBG0YIPOQ8OW
		0E9JINHV8YPSXX940XZQ
		0JJ7UPKK5NRBFKBV4RBW
		0K9A0NEHPW5E793L8SSS
		0LS4847OD3QIDR0DAEHN
		0POUSR2V31YYWWMU1AXO
		0TSEI0MP7TWUIZ3LEJH1
		0V0ZDPL65RH0YLNL80VZ
		```
---

### Codebook Visualizations
**For Vanilla/Improved VQ-VAE:** 
(This decodes individual codes from the single codebook).

	```
	python visualize_codebook_embeddings.py \
	    --checkpoint ../training/outputs/vanilla_run/best_model.pt \
	    --output_dir codebook_viz
	```

**For RVQ (Residual Vector Quantization):**

	```
	python visualize_rvq_codebook.py \
	    --checkpoint ../training/outputs/rvq_run/best_model.pt \
	    --output_dir rvq_viz \
	    --mode all \
	    --num_samples 10
	```
- **individual** - Each quantizer in isolation (diagonal matrix pattern)
	```
	Q0: [code, 0, 0, 0]
	Q1: [0, code, 0, 0]
	Q2: [0, 0, code, 0]
	Q3: [0, 0, 0, code]
	```
- **combinations** - Random full code combinations
	- `[a, b, c, d]` with random codes at each position.
	- Shows what diverse behaviors the full codebook can produce

- **cumulative** - Progressive refinement showing all quantizations building up:
	```
	Q0: [code, 0, 0, 0] (coarse)
	Q0+Q1: [code, code, 0, 0] (add detail)
	Q0+Q1+Q2: [code, code, code, 0] (more detail)
	Full: [code, code, code, code] ← All quantizers applied!
	```

- **ablations** - Removing quantizers (backward)
	```
	Full: [a, b, c, d]
	-Q3: [a, b, c, 0]
	-Q2&Q3: [a, b, 0, 0]
	Q0 only: [a, 0, 0, 0]
	```
- **all** - Runs all four modes above
